#!/usr/bin/env python3
"""Chronological temperature-scaling diagnostics for prediction JSONL.

Fits temperature on an early calibration slice and evaluates reliability on a
later holdout slice to reduce leakage risk.
"""

import argparse
import json
import math
from pathlib import Path

from cn_lottery_backtest import load_jsonl


def clamp_probs(probs):
    s = sum(max(1e-12, float(p)) for p in probs)
    return [max(1e-12, float(p)) / s for p in probs]


def apply_temperature(probs, temperature):
    if temperature <= 0:
        return clamp_probs(probs)
    logs = [math.log(max(1e-12, float(p))) / temperature for p in probs]
    m = max(logs)
    exps = [math.exp(x - m) for x in logs]
    z = sum(exps)
    return [x / z for x in exps]


def nll(rows, temperature):
    total = 0.0
    n = 0
    for r in rows:
        y = r["true_label"]
        p = apply_temperature(r["probs"], temperature)
        total += -math.log(max(1e-12, p[y]))
        n += 1
    return total / max(1, n)


def brier(rows, temperature):
    total = 0.0
    n = 0
    for r in rows:
        y = r["true_label"]
        p = apply_temperature(r["probs"], temperature)
        total += sum((pi - (1.0 if i == y else 0.0)) ** 2 for i, pi in enumerate(p))
        n += 1
    return total / max(1, n)


def ece(rows, temperature, bins=10):
    bucket_conf = [0.0] * bins
    bucket_acc = [0.0] * bins
    bucket_n = [0] * bins

    for r in rows:
        y = r["true_label"]
        p = apply_temperature(r["probs"], temperature)
        conf = max(p)
        pred = max(range(len(p)), key=lambda i: p[i])
        b = min(bins - 1, int(conf * bins))
        bucket_conf[b] += conf
        bucket_acc[b] += 1.0 if pred == y else 0.0
        bucket_n[b] += 1

    n = sum(bucket_n)
    if n == 0:
        return 0.0

    err = 0.0
    for i in range(bins):
        if bucket_n[i] == 0:
            continue
        avg_conf = bucket_conf[i] / bucket_n[i]
        avg_acc = bucket_acc[i] / bucket_n[i]
        err += (bucket_n[i] / n) * abs(avg_acc - avg_conf)
    return err


def search_temperature(rows, objective="nll", t_min=0.5, t_max=2.0, step=0.01):
    best_t = 1.0
    best_score = float("inf")
    t = t_min
    while t <= t_max + 1e-12:
        if objective == "ece":
            score = ece(rows, t)
        elif objective == "brier":
            score = brier(rows, t)
        else:
            score = nll(rows, t)
        if score < best_score:
            best_score = score
            best_t = round(t, 4)
        t += step
    return best_t, best_score


def main() -> None:
    parser = argparse.ArgumentParser(description="Temperature-scaling diagnostics")
    parser.add_argument("--predictions", default="output/predictions/handicap_label_test.jsonl")
    parser.add_argument("--meta-file", default="data/processed/match_meta.jsonl")
    parser.add_argument("--prematch-file", default="data/processed/prematch_features.jsonl")
    parser.add_argument("--calib-ratio", type=float, default=0.5)
    parser.add_argument("--t-min", type=float, default=0.5)
    parser.add_argument("--t-max", type=float, default=2.0)
    parser.add_argument("--t-step", type=float, default=0.01)
    parser.add_argument(
        "--objective",
        choices=["nll", "ece", "brier"],
        default="ece",
        help="Metric to optimize when selecting best_temperature",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    pred_path = root / args.predictions if not Path(args.predictions).is_absolute() else Path(args.predictions)
    meta_path = root / args.meta_file if not Path(args.meta_file).is_absolute() else Path(args.meta_file)
    prematch_path = (
        root / args.prematch_file if not Path(args.prematch_file).is_absolute() else Path(args.prematch_file)
    )

    preds = load_jsonl(str(pred_path))
    meta_rows = load_jsonl(str(meta_path))
    prematch_rows = load_jsonl(str(prematch_path))
    match_to_dt = {r.get("match_id"): r.get("datetime_utc", "") for r in meta_rows}
    match_to_promoted = {}
    for r in prematch_rows:
        mid = r.get("match_id")
        if not mid:
            continue
        home = int(r.get("home", {}).get("promoted_this_season", 0) or 0)
        away = int(r.get("away", {}).get("promoted_this_season", 0) or 0)
        match_to_promoted[mid] = bool(home or away)

    rows = []
    for p in preds:
        mid = p.get("match_id")
        if mid is None or p.get("true_label") is None or p.get("probs") is None:
            continue
        rows.append(
            {
                "match_id": mid,
                "datetime_utc": match_to_dt.get(mid, ""),
                "promoted_match": bool(match_to_promoted.get(mid, False)),
                "true_label": int(p["true_label"]),
                "probs": [float(x) for x in p["probs"]],
            }
        )

    rows.sort(key=lambda r: r["datetime_utc"])
    split = int(len(rows) * args.calib_ratio)
    calib = rows[:split]
    holdout = rows[split:]

    best_t_nll, calib_nll = search_temperature(calib, "nll", args.t_min, args.t_max, args.t_step)
    best_t_ece, calib_ece = search_temperature(calib, "ece", args.t_min, args.t_max, args.t_step)
    best_t_brier, calib_brier = search_temperature(calib, "brier", args.t_min, args.t_max, args.t_step)

    if args.objective == "nll":
        best_t = best_t_nll
    elif args.objective == "brier":
        best_t = best_t_brier
    else:
        best_t = best_t_ece

    report = {
        "num_rows": len(rows),
        "calib_rows": len(calib),
        "holdout_rows": len(holdout),
        "holdout_promoted_rows": sum(1 for r in holdout if r.get("promoted_match")),
        "holdout_non_promoted_rows": sum(1 for r in holdout if not r.get("promoted_match")),
        "temperature_objective": args.objective,
        "best_temperature": best_t,
        "best_temperature_nll": best_t_nll,
        "best_temperature_ece": best_t_ece,
        "best_temperature_brier": best_t_brier,
        "calib_nll_at_best_t": round(calib_nll, 6),
        "calib_ece_at_best_t": round(calib_ece, 6),
        "calib_brier_at_best_t": round(calib_brier, 6),
        "holdout": {
            "uncalibrated": {
                "nll": round(nll(holdout, 1.0), 6),
                "brier": round(brier(holdout, 1.0), 6),
                "ece": round(ece(holdout, 1.0), 6),
            },
            "temperature_scaled": {
                "nll": round(nll(holdout, best_t), 6),
                "brier": round(brier(holdout, best_t), 6),
                "ece": round(ece(holdout, best_t), 6),
            },
        },
    }

    holdout_promoted = [r for r in holdout if r.get("promoted_match")]
    holdout_non_promoted = [r for r in holdout if not r.get("promoted_match")]
    report["holdout_promoted"] = {
        "uncalibrated": {
            "nll": round(nll(holdout_promoted, 1.0), 6),
            "brier": round(brier(holdout_promoted, 1.0), 6),
            "ece": round(ece(holdout_promoted, 1.0), 6),
        },
        "temperature_scaled": {
            "nll": round(nll(holdout_promoted, best_t), 6),
            "brier": round(brier(holdout_promoted, best_t), 6),
            "ece": round(ece(holdout_promoted, best_t), 6),
        },
    }
    report["holdout_non_promoted"] = {
        "uncalibrated": {
            "nll": round(nll(holdout_non_promoted, 1.0), 6),
            "brier": round(brier(holdout_non_promoted, 1.0), 6),
            "ece": round(ece(holdout_non_promoted, 1.0), 6),
        },
        "temperature_scaled": {
            "nll": round(nll(holdout_non_promoted, best_t), 6),
            "brier": round(brier(holdout_non_promoted, best_t), 6),
            "ece": round(ece(holdout_non_promoted, best_t), 6),
        },
    }

    u = report["holdout"]["uncalibrated"]
    c = report["holdout"]["temperature_scaled"]
    print(f"Rows: total={len(rows)} calib={len(calib)} holdout={len(holdout)}")
    print(f"Best temperature ({args.objective}): {best_t}")
    print(f"  Candidate best_t by NLL={best_t_nll}, ECE={best_t_ece}, Brier={best_t_brier}")
    print("Holdout metrics:")
    print(f"  NLL  {u['nll']:.6f} -> {c['nll']:.6f}")
    print(f"  Brier {u['brier']:.6f} -> {c['brier']:.6f}")
    print(f"  ECE  {u['ece']:.6f} -> {c['ece']:.6f}")
    print(
        "Promoted holdout rows: "
        f"{report['holdout_promoted_rows']} | Non-promoted: {report['holdout_non_promoted_rows']}"
    )

    out = (
        Path(args.output)
        if args.output
        else root / "output" / "backtest" / (pred_path.stem + "_temperature_report.json")
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved temperature report to {out}")


if __name__ == "__main__":
    main()
