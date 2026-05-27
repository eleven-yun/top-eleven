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
    parser.add_argument("--prematch-file", default="data/processed/prematch_features.jsonl")
    parser.add_argument("--calib-ratio", type=float, default=0.5)
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

    models = fit_isotonic(calib, num_classes=3)
    holdout_uncal = [{"true_label": int(r["true_label"]), "probs": [float(x) for x in r["probs"]]} for r in holdout]
    holdout_iso = apply_isotonic_to_rows(holdout, models)
    holdout_promoted = [r for r in holdout if r.get("promoted_match")]
    holdout_non_promoted = [r for r in holdout if not r.get("promoted_match")]
    holdout_promoted_uncal = [
        {"true_label": int(r["true_label"]), "probs": [float(x) for x in r["probs"]]} for r in holdout_promoted
    ]
    holdout_non_promoted_uncal = [
        {"true_label": int(r["true_label"]), "probs": [float(x) for x in r["probs"]]} for r in holdout_non_promoted
    ]
    holdout_promoted_iso = apply_isotonic_to_rows(holdout_promoted, models)
    holdout_non_promoted_iso = apply_isotonic_to_rows(holdout_non_promoted, models)

    report = {
        "num_rows": len(rows),
        "calib_rows": len(calib),
        "holdout_rows": len(holdout),
        "holdout_promoted_rows": len(holdout_promoted),
        "holdout_non_promoted_rows": len(holdout_non_promoted),
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
        "holdout_promoted": {
            "uncalibrated": {
                "nll": round(nll(holdout_promoted_uncal, 1.0), 6),
                "brier": round(brier(holdout_promoted_uncal, 1.0), 6),
                "ece": round(ece(holdout_promoted_uncal, 1.0), 6),
            },
            "isotonic_scaled": {
                "nll": round(nll(holdout_promoted_iso, 1.0), 6),
                "brier": round(brier(holdout_promoted_iso, 1.0), 6),
                "ece": round(ece(holdout_promoted_iso, 1.0), 6),
            },
        },
        "holdout_non_promoted": {
            "uncalibrated": {
                "nll": round(nll(holdout_non_promoted_uncal, 1.0), 6),
                "brier": round(brier(holdout_non_promoted_uncal, 1.0), 6),
                "ece": round(ece(holdout_non_promoted_uncal, 1.0), 6),
            },
            "isotonic_scaled": {
                "nll": round(nll(holdout_non_promoted_iso, 1.0), 6),
                "brier": round(brier(holdout_non_promoted_iso, 1.0), 6),
                "ece": round(ece(holdout_non_promoted_iso, 1.0), 6),
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
    print(
        "Promoted holdout rows: "
        f"{report['holdout_promoted_rows']} | Non-promoted: {report['holdout_non_promoted_rows']}"
    )

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
