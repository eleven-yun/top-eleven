#!/usr/bin/env python3
"""
Fetch China Lottery 竞彩足球 historical SP (payout) odds from 500.com kaijiang pages.

Output: data/raw/china_lottery/odds_raw.jsonl
Schema matches enrich_lottery_odds.py expected format:
  source_site, source_match_id, issue_no, play_type_raw,
  league_name_raw, kickoff_local, home_team_raw, away_team_raw,
  handicap_line_raw, odds_home, odds_draw, odds_away,
  odds_capture_time, odds_stage
"""

import argparse
import json
import re
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import requests

BASE_URL = "https://zx.500.com/jczq/kaijiang.php"
ROOT = Path(__file__).resolve().parent.parent
OUTPUT_FILE = ROOT / "data" / "raw" / "china_lottery" / "odds_raw.jsonl"
DEFAULT_START = "2024-08-01"
DEFAULT_END = "2025-06-01"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9",
    "Referer": "https://zx.500.com/jczq/kaijiang.php",
}
# SP values after the score column:
# <td>&nbsp;</td><td>RESULT</td><td class="eng"><span ...>SP_VALUE</span></td>
SP_BLOCK = re.compile(
    r"<td[^>]*>&nbsp;</td>\s*"
    r"<td[^>]*>([^<]+)</td>\s*"
    r'<td[^>]*class="eng"[^>]*><span[^>]*>([.\d]+)</span></td>',
    re.DOTALL,
)


def daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def fetch_day(d: date) -> list[dict]:
    """Fetch all completed matches for a given date."""
    ds = d.strftime("%Y-%m-%d")
    try:
        r = requests.get(
            BASE_URL,
            params={"d": ds},
            headers=HEADERS,
            timeout=20,
        )
        r.raise_for_status()
        r.encoding = "gb2312"
        return parse_page(r.text, ds)
    except requests.RequestException as exc:
        print(f"  WARN: failed {ds}: {exc}")
        return []


def parse_page(html: str, date_str: str) -> list[dict]:
    """Parse kaijiang HTML page for one date and return list of records."""
    records = []

    def shift_year_safe(dt: datetime, target_year: int) -> datetime:
        """Shift year while handling leap-day edge cases safely."""
        try:
            return dt.replace(year=target_year)
        except ValueError:
            # Handles Feb-29 -> non-leap-year by clamping day downward.
            day = dt.day
            while day > 28:
                day -= 1
                try:
                    return dt.replace(year=target_year, day=day)
                except ValueError:
                    continue
            return dt.replace(year=target_year, month=2, day=28)

    # Isolate the main results table to avoid false matches in login/nav areas
    # The main table starts after the header row
    table_match = re.search(
        r'<tr>\s*<th[^>]*>赛事编号</th>.*?</tbody>',
        html,
        re.DOTALL,
    )
    if table_match:
        table_html = table_match.group(0)
    else:
        table_html = html

    # Split into individual rows
    rows = re.split(r"(?=<tr>\s*<td[^>]*>周[一二三四五六日]\d+</td>)", table_html)

    for row in rows:
        row = row.strip()
        if not row.startswith("<tr>"):
            continue

        # Extract issue number and basic fields
        m = re.search(
            r"<td[^>]*>(周[一二三四五六日]\d+)</td>\s*"
            r"<td[^>]*><a[^>]+>([^<]+)</a></td>\s*"
            r'<td[^>]*class="eng"[^>]*>([^<]+)</td>\s*'
            r"<td[^>]*><a[^>]+>([^<]+)</a></td>\s*"
            r'<td[^>]*class="eng"[^>]*>(?:<span[^>]*>)?([^<]*)(?:</span>)?</td>\s*'
            r"<td[^>]*><a[^>]+>([^<]+)</a></td>\s*"
            r'<td[^>]*class="eng"[^>]*>([^<]+)</td>',
            row,
            re.DOTALL,
        )
        if not m:
            continue

        issue_no = m.group(1).strip()
        issue_id = f"{date_str}-{issue_no}"
        league = m.group(2).strip()
        kickoff_raw = m.group(3).strip()  # e.g. "03-16 07:30"
        home_team = m.group(4).strip()
        handicap_raw = m.group(5).strip()  # e.g. "-1" or "0"
        away_team = m.group(6).strip()
        # score is group 7, we don't need it

        # Convert kickoff local China time (UTC+8) to UTC for matching.
        # Around New Year, kaijiang page date and row MM-DD can cross year boundary,
        # so choose the closest year candidate to the page date.
        page_dt = datetime.strptime(date_str, "%Y-%m-%d")
        year = page_dt.year
        kickoff_cn_str = f"{year}-{kickoff_raw}"  # "YYYY-MM-DD HH:MM" (UTC+8)
        try:
            kickoff_cn_dt = datetime.strptime(kickoff_cn_str, "%Y-%m-%d %H:%M")
            delta_days = abs((kickoff_cn_dt.date() - page_dt.date()).days)
            if delta_days > 180:
                prev_year = shift_year_safe(kickoff_cn_dt, kickoff_cn_dt.year - 1)
                next_year = shift_year_safe(kickoff_cn_dt, kickoff_cn_dt.year + 1)
                kickoff_cn_dt = min(
                    (kickoff_cn_dt, prev_year, next_year),
                    key=lambda x: abs((x.date() - page_dt.date()).days),
                )
            kickoff_utc_dt = kickoff_cn_dt - timedelta(hours=8)
            kickoff_local = kickoff_utc_dt.strftime("%Y-%m-%d %H:%M")
        except ValueError:
            kickoff_local = kickoff_cn_str  # fallback

        # Extract SP values — the row has two SP blocks:
        # 1st: 让球胜平负 (handicap 1X2) — stat="SPF"
        # 2nd: 胜平负 (fulltime 1X2) — stat="NSPF"
        # Each block: <td>&nbsp;</td><td>RESULT</td><td ...><span...>SP</span></td>
        # Get remaining row HTML after main fields
        after_score_idx = m.end()
        remaining = row[after_score_idx:]
        sp_pairs = SP_BLOCK.findall(remaining)

        # sp_pairs: list of (result_label, sp_value)
        # Order is: [SPF_result, SPF_sp], [NSPF_result, NSPF_sp], [big_small?, ...]
        # Some rows may have fewer (if play type not available)
        spf_result = spf_sp = nspf_result = nspf_sp = None
        if len(sp_pairs) >= 1:
            spf_result, spf_sp = sp_pairs[0]
        if len(sp_pairs) >= 2:
            nspf_result, nspf_sp = sp_pairs[1]

        # Build records for each play type
        capture_time = f"{date_str}T23:59:59+08:00"

        # 让球胜平负 (handicap 1X2)
        if spf_sp:
            # For handicap: home = handicap win, draw = handicap draw, away = handicap lose
            # The result tells us which outcome paid out, but we store all SP odds
            # Unfortunately kaijiang only shows the WINNING payout, not all three
            # We need to derive from the result which slot it belongs to
            # Schema expects: odds_home, odds_draw, odds_away
            # We store the known SP in the correct field, NaN for others
            h_sp = float(spf_sp) if spf_sp else None
            result_label = (spf_result or "").strip()

            h_home = h_draw = h_away = None
            if result_label == "胜":
                h_home = h_sp
            elif result_label == "平":
                h_draw = h_sp
            elif result_label == "负":
                h_away = h_sp

            records.append({
                "source_site": "500.com",
                "source_match_id": issue_id,
                "issue_no": issue_id,
                "play_type_raw": "让球胜平负",
                "league_name_raw": league,
                "kickoff_local": kickoff_local,
                "home_team_raw": home_team,
                "away_team_raw": away_team,
                "handicap_line_raw": handicap_raw,
                "odds_home": h_home,
                "odds_draw": h_draw,
                "odds_away": h_away,
                "actual_result": result_label,
                "actual_sp": h_sp,
                "odds_capture_time": capture_time,
                "odds_stage": "postmatch_settlement",
            })

        # 胜平负 (fulltime 1X2)
        if nspf_sp:
            nspf_val = float(nspf_sp)
            result_label = (nspf_result or "").strip()

            n_home = n_draw = n_away = None
            if result_label == "胜":
                n_home = nspf_val
            elif result_label == "平":
                n_draw = nspf_val
            elif result_label == "负":
                n_away = nspf_val

            records.append({
                "source_site": "500.com",
                "source_match_id": issue_id,
                "issue_no": issue_id,
                "play_type_raw": "胜平负",
                "league_name_raw": league,
                "kickoff_local": kickoff_local,
                "home_team_raw": home_team,
                "away_team_raw": away_team,
                "handicap_line_raw": "",
                "odds_home": n_home,
                "odds_draw": n_draw,
                "odds_away": n_away,
                "actual_result": result_label,
                "actual_sp": nspf_val,
                "odds_capture_time": capture_time,
                "odds_stage": "postmatch_settlement",
            })

    return records


def main():
    parser = argparse.ArgumentParser(
        description="Fetch China Lottery 竞彩足球 SP odds from 500.com"
    )
    parser.add_argument("--start", default=DEFAULT_START, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=DEFAULT_END, help="End date YYYY-MM-DD")
    parser.add_argument(
        "--output",
        default=str(OUTPUT_FILE),
        help="Output JSONL path (relative paths resolve from repo root)",
    )
    parser.add_argument("--delay", type=float, default=1.5, help="Seconds between requests")
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing output file (default: overwrite)",
    )
    args = parser.parse_args()

    start_date = date.fromisoformat(args.start)
    end_date = date.fromisoformat(args.end)
    out_arg = Path(args.output)
    out_path = out_arg if out_arg.is_absolute() else (ROOT / out_arg)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if args.append else "w"
    total_written = 0

    print(f"Fetching {args.start} → {args.end} → {out_path}")

    with open(out_path, mode, encoding="utf-8") as fout:
        for d in daterange(start_date, end_date):
            records = fetch_day(d)
            if records:
                for rec in records:
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                total_written += len(records)
                print(f"  {d.isoformat()}: {len(records)} records (total {total_written})")
            else:
                # No matches on this date (normal for weekdays with no lottery)
                print(f"  {d.isoformat()}: 0 records")

            time.sleep(args.delay)

    print(f"\nDone. Wrote {total_written} records to {out_path}")


if __name__ == "__main__":
    main()
