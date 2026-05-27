#!/usr/bin/env python3
"""Chronological isotonic-calibration diagnostics for multiclass probabilities."""

import argparse
import json
from pathlib import Path

from sklearn.isotonic import IsotonicRegression

from cn_lottery_backtest import load_jsonl
from calibrate_temperature import brier, ece, nll


def normalize(probs):
    s = sum(max(1e-12, float(p)) for p in probs)
    return [max(1e-12, float(p)) / s for p in probs]


def fit_isotonic(rows, num_classes=3):
    models = []
    for cls in range(num_classes):
        x = []
        y = []
        for r in rows:
            p = float(r["probs"][cls])
            x.append(p)
            y.append(1.0 if int(r["true_label"]) == cls else 0.0)
        m = IsotonicRegression(out_of_bounds="clip")
        m.fit(x, y)
        models.append(m)
    return models


def apply_isotonic_to_rows(rows, models):
    out = []
    for r in rows:
        p = [float(x) for x in r["probs"]]
        cal = [float(models[i].predict([p[i]])[0]) for i in range(len(models))]
        cal = normalize(cal)
        out.append({"true_label": int(r["true_label"]), "probs": cal})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Isotonic calibration diagnostics")
    parser.add_argument("--predictions", default="output/predictions/handicap_label_test.jsonl")
    parser.add_argument("--meta-file", default="data/processed/match_meta.jsonl")
    parser.add_argument("--calib-ratio", type=float, default=0.5)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    pred_path = root / args.predictions if not Path(args.predictions).is_absolute() else Path(args.predictions)
    meta_path = root / args.meta_file if not Path(args.meta_file).is_absolute() else Path(args.meta_file)

    preds = load_jsonl(str(pred_path))
    meta_rows = load_jsonl(str(meta_path))
    match_to_dt = {r.get("match_id"): r.get("datetime_utc", "") for r in meta_rows}

    rows = []
    for p in preds:
        mid = p.get("match_id")
        if mid is None or p.get("true_label") is None or p.get("probs") is None:
            continue
        rows.append(
            {
                "match_id": mid,
                "datetime_utc": match_to_dt.get(mid, ""),
                "true_label": int(p["true_label"]),
                "probs": [float(x) for x in p["probs"]],
            }
        )

    rows.sort(key=lambda r: r["datetime_utc"])
    split = int(len(rows) * args.calib_ratio)
    calib = rows[:split]
    holdout = rows[split:]

    models = fit_isotonic(calib, num_classes=3)
    holdout_uncal = [{"true_label": int(r["true_label"]), "probs": [float(x) for x in r["probs"]]} for r in holdout]
    holdout_iso = apply_isotonic_to_rows(holdout, models)

    report = {
        "num_rows": len(rows),
        "calib_rows": len(calib),
        "holdout_rows": len(holdout),
        "holdout": {
            "uncalibrated": {
                "nll": round(nll(holdout_uncal, 1.0), 6),
                "brier": round(brier(holdout_uncal, 1.0), 6),
                "ece": round(ece(holdout_uncal, 1.0), 6),
            },
            "isotonic_scaled": {
                "nll": round(nll(holdout_iso, 1.0), 6),
                "brier": round(brier(holdout_iso, 1.0), 6),
                "ece": round(ece(holdout_iso, 1.0), 6),
            },
        },
    }

    u = report["holdout"]["uncalibrated"]
    c = report["holdout"]["isotonic_scaled"]
    print(f"Rows: total={len(rows)} calib={len(calib)} holdout={len(holdout)}")
    print("Holdout metrics:")
    print(f"  NLL  {u['nll']:.6f} -> {c['nll']:.6f}")
    print(f"  Brier {u['brier']:.6f} -> {c['brier']:.6f}")
    print(f"  ECE  {u['ece']:.6f} -> {c['ece']:.6f}")

    out = (
        Path(args.output)
        if args.output
        else root / "output" / "backtest" / (pred_path.stem + "_isotonic_report.json")
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved isotonic report to {out}")


if __name__ == "__main__":
    main()
