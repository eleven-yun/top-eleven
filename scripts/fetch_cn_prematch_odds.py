#!/usr/bin/env python3
"""
Fetch China Lottery 竞彩足球 pre-match odds history from 500.com.

For each match on the kaijiang page, fetches the full odds history
(opening → closing) for all three outcomes (home/draw/away) by calling
the undocumented readpl AJAX API.

Output: data/raw/china_lottery/prematch_odds_raw.jsonl
Schema:
  source_site, zid, issue_no, play_type_raw, play_type,
  date_local, league_name_raw, kickoff_local,
  home_team_raw, away_team_raw, handicap_line_raw,
  opening_win, opening_draw, opening_lost, opening_time,
  closing_win, closing_draw, closing_lost, closing_time,
  odds_history   (list of {win, draw, lost, time})
"""

import argparse
import json
import re
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import requests

KAIJIANG_URL = "https://zx.500.com/jczq/kaijiang.php"
OUTPUT_FILE = Path("data/raw/china_lottery/prematch_odds_raw.jsonl")
DEFAULT_START = "2024-08-01"
DEFAULT_END = "2026-06-01"
REQUEST_DELAY = 0.5  # seconds between readpl API calls

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9",
}

AJAX_HEADERS = {
    **HEADERS,
    "X-Requested-With": "XMLHttpRequest",
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Referer": KAIJIANG_URL,
}

# Map kaijiang playid → internal play type
PLAY_TYPES = {
    1: ("spf", "handicap_1x2", "让球胜平负"),
    2: ("nspf", "fulltime_1x2", "胜平负"),
}

# ── Bot challenge bypass ─────────────────────────────────────────────────────

def _solve_eo_challenge(script: str) -> dict[str, str]:
    """Parse and solve the TencentEdgeOne bot challenge JS to get cookies.

    The challenge sets two cookies:
      __tst_status = sum of three object-literal constants, mod 2^32
      EO_Bot_Ssid  = a standalone constant in the inner cookie-builder function

    'EO_Bot_Ssid=' appears as a literal string in the script; the digit that
    follows it (within ~200 chars) is the EO_Bot_Ssid value.  All other 8-10
    digit numbers in the script are the addends for __tst_status.
    """
    # EO_Bot_Ssid value: the number that follows the literal 'EO_Bot_Ssid=' string
    eo_m = re.search(r'EO_Bot_Ssid.{1,200}?(\d{8,10})', script, re.DOTALL)
    if not eo_m:
        return {}
    eo_value = int(eo_m.group(1))
    # All other 8-10 digit numbers are addends for __tst_status
    all_nums = [int(x) for x in re.findall(r'\b(\d{8,10})\b', script)]
    tst_nums = [n for n in all_nums if n != eo_value]
    if not tst_nums:
        return {}
    tst_value = sum(tst_nums) % (2 ** 32)
    return {
        "__tst_status": str(tst_value),
        "EO_Bot_Ssid": str(eo_value),
    }


def _is_challenge(response: requests.Response) -> bool:
    return len(response.content) < 2000 and b"EO_Bot_Ssid" in response.content


def get_with_bot_bypass(
    session: requests.Session,
    url: str,
    params: dict | None = None,
    headers: dict | None = None,
    timeout: int = 20,
) -> requests.Response:
    """GET with automatic bot-challenge retry.

    If the server returns the TencentEdgeOne JS challenge (< 2 KB, contains
    'EO_Bot_Ssid'), solve the cookie computation, wait for the validation
    window (the JS itself uses a 1.2 s setTimeout), then retry.
    """
    resp = session.get(url, params=params, headers=headers, timeout=timeout)
    if _is_challenge(resp):
        cookies = _solve_eo_challenge(resp.text)
        if cookies:
            session.cookies.update(cookies)
            time.sleep(1.5)  # match the JS challenge redirect delay
            resp = session.get(url, params=params, headers=headers, timeout=timeout)
    return resp


def daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def extract_match_rows(html: str, date_str: str) -> list[dict]:
    """Parse kaijiang page HTML and return list of match metadata with zid."""
    # Decode if bytes
    if isinstance(html, bytes):
        html = html.decode("gb2312", errors="replace")

    rows = []
    # Find rows with issue number + zid
    for row_m in re.finditer(
        r"<tr>(?P<cells>.*?)</tr>",
        html,
        re.DOTALL,
    ):
        cells_html = row_m.group("cells")

        # Must have issue number (e.g. "周四004")
        issue_m = re.search(r"<td[^>]*>(周[一二三四五六日]\d+)</td>", cells_html)
        if not issue_m:
            continue

        # Must have a zid attribute on a td
        zid_m = re.search(r'zid="(\d+)"', cells_html)
        if not zid_m:
            continue

        issue_no = issue_m.group(1).strip()
        zid = zid_m.group(1)

        # stat (SPF or NSPF)
        stat_m = re.search(r'stat="([^"]+)"', cells_html)
        stat = stat_m.group(1).upper() if stat_m else "SPF"
        wtype = "spf" if stat == "SPF" else "nspf"
        play_type = "handicap_1x2" if wtype == "spf" else "fulltime_1x2"
        play_type_raw = "让球胜平负" if wtype == "spf" else "胜平负"

        # League name
        league_m = re.search(r'class="league"[^>]*>([^<]+)</a>', cells_html)
        league = league_m.group(1).strip() if league_m else ""

        # Kickoff time
        kickoff_m = re.search(r'class="eng"[^>]*>(\d{2}-\d{2} \d{2}:\d{2})', cells_html)
        kickoff_raw = kickoff_m.group(1) if kickoff_m else ""
        kickoff_local = ""
        if kickoff_raw:
            year = date_str[:4]
            try:
                cn_dt = datetime.strptime(f"{year}-{kickoff_raw}", "%Y-%m-%d %H:%M")
                utc_dt = cn_dt - timedelta(hours=8)
                kickoff_local = utc_dt.strftime("%Y-%m-%d %H:%M")
            except ValueError:
                kickoff_local = f"{year}-{kickoff_raw}"

        # Home team (in text_r td) and away team (in text_l td)
        home_m = re.search(r'class="text_r"[^>]*>.*?<a[^>]*>([^<]+)</a>', cells_html, re.DOTALL)
        away_m = re.search(r'class="text_l"[^>]*>.*?<a[^>]*>([^<]+)</a>', cells_html, re.DOTALL)
        home_team = home_m.group(1).strip() if home_m else ""
        away_team = away_m.group(1).strip() if away_m else ""

        # Handicap line
        handicap_m = re.search(
            r'class="eng"[^>]*>(?:<span[^>]*>)?([+-]?\d*(?:\.\d+)?|平)(?:</span>)?</td>',
            cells_html,
        )
        handicap_raw = handicap_m.group(1).strip() if handicap_m else ""

        rows.append({
            "zid": zid,
            "issue_no": issue_no,
            "wtype": wtype,
            "play_type": play_type,
            "play_type_raw": play_type_raw,
            "date_local": date_str,
            "league_name_raw": league,
            "kickoff_local": kickoff_local,
            "home_team_raw": home_team,
            "away_team_raw": away_team,
            "handicap_line_raw": handicap_raw,
        })

    return rows


def fetch_odds_history(session: requests.Session, zid: str, date_str: str, wtype: str) -> list[dict]:
    """Call readpl API and return full odds history list, newest first."""
    try:
        resp = get_with_bot_bypass(
            session,
            KAIJIANG_URL,
            params={
                "d": date_str,
                "playid": "1" if wtype == "spf" else "2",
                "step": "readpl",
                "zxid": zid,
                "date": date_str,
                "wtype": wtype,
                "rnd": str(int(time.time() * 1000)),
            },
            headers={
                **AJAX_HEADERS,
                "Referer": f"{KAIJIANG_URL}?d={date_str}&playid={'1' if wtype == 'spf' else '2'}",
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = json.loads(resp.text)
        if isinstance(data, list):
            return data
        return []
    except (requests.RequestException, json.JSONDecodeError, ValueError) as exc:
        print(f"    WARN: readpl failed zid={zid} date={date_str} wtype={wtype}: {exc}")
        return []


def process_history(history: list[dict]) -> dict:
    """Extract opening and closing odds from full history (newest-first order)."""
    if not history:
        return {}
    closing = history[0]   # most recent entry
    opening = history[-1]  # earliest entry
    return {
        "opening_win": _safe_float(opening.get("win")),
        "opening_draw": _safe_float(opening.get("draw")),
        "opening_lost": _safe_float(opening.get("lost")),
        "opening_time": opening.get("time", ""),
        "closing_win": _safe_float(closing.get("win")),
        "closing_draw": _safe_float(closing.get("draw")),
        "closing_lost": _safe_float(closing.get("lost")),
        "closing_time": closing.get("time", ""),
    }


def _safe_float(val) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def fetch_day(session: requests.Session, d: date, seen_zids: set[str], args) -> list[dict]:
    """Fetch all pre-match odds records for a single date."""
    date_str = d.strftime("%Y-%m-%d")
    records = []

    for playid, (wtype, play_type, play_type_raw) in PLAY_TYPES.items():
        if args.play_type and wtype not in args.play_type:
            continue
        try:
            resp = get_with_bot_bypass(
                session,
                KAIJIANG_URL,
                params={"d": date_str, "playid": str(playid)},
                headers=HEADERS,
                timeout=20,
            )
            resp.raise_for_status()
            resp.encoding = "gb2312"
            page_html = resp.text
        except requests.RequestException as exc:
            print(f"  WARN: kaijiang page failed {date_str} playid={playid}: {exc}")
            continue

        rows = extract_match_rows(page_html, date_str)
        if not rows:
            continue

        for row in rows:
            zid = row["zid"]
            cache_key = f"{zid}_{wtype}"
            if cache_key in seen_zids:
                continue

            time.sleep(args.delay)
            history = fetch_odds_history(session, zid, date_str, wtype)
            seen_zids.add(cache_key)

            if not history:
                continue

            odds_info = process_history(history)
            record = {
                "source_site": "500.com",
                "zid": zid,
                "issue_no": row["issue_no"],
                "play_type_raw": row["play_type_raw"],
                "play_type": row["play_type"],
                "date_local": date_str,
                "league_name_raw": row["league_name_raw"],
                "kickoff_local": row["kickoff_local"],
                "home_team_raw": row["home_team_raw"],
                "away_team_raw": row["away_team_raw"],
                "handicap_line_raw": row["handicap_line_raw"],
                **odds_info,
                "odds_history": history,
            }
            records.append(record)

    return records


def main():
    parser = argparse.ArgumentParser(
        description="Fetch China Lottery 竞彩足球 pre-match odds from 500.com"
    )
    parser.add_argument("--start", default=DEFAULT_START, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=DEFAULT_END, help="End date YYYY-MM-DD")
    parser.add_argument("--output", default=str(OUTPUT_FILE), help="Output JSONL path")
    parser.add_argument("--delay", type=float, default=REQUEST_DELAY,
                        help="Seconds between readpl API calls")
    parser.add_argument("--append", action="store_true",
                        help="Append to existing output file (default: overwrite)")
    parser.add_argument("--play-type", nargs="+", choices=["spf", "nspf"],
                        help="Only fetch specific play types (default: both)")
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing zids to skip if appending
    seen_zids: set[str] = set()
    if args.append and out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    zid = rec.get("zid", "")
                    wtype = "spf" if rec.get("play_type") == "handicap_1x2" else "nspf"
                    if zid:
                        seen_zids.add(f"{zid}_{wtype}")
                except json.JSONDecodeError:
                    continue
        print(f"Resuming: {len(seen_zids)} zids already fetched")

    write_mode = "a" if args.append else "w"
    total_records = 0

    session = requests.Session()

    with out_path.open(write_mode, encoding="utf-8") as f:
        for d in daterange(start, end):
            day_records = fetch_day(session, d, seen_zids, args)
            if day_records:
                for rec in day_records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                total_records += len(day_records)
                print(f"  {d}: {len(day_records)} records (total: {total_records})")

    print(f"\nDone. Wrote {total_records} records → {out_path}")


if __name__ == "__main__":
    main()
