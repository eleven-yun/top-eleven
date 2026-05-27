"""Calibration helpers shared by prediction scripts."""

import json
import os


def load_best_temperature(report_path: str) -> float | None:
    """Load best_temperature from a calibration report JSON file."""
    if not report_path or not os.path.exists(report_path):
        return None
    try:
        with open(report_path, encoding="utf-8") as f:
            payload = json.load(f)
        t = payload.get("best_temperature")
        if t is None:
            return None
        t = float(t)
        return t if t > 0 else None
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def resolve_task_temperature(
    root: str,
    task: str,
    explicit_temperature: float | None,
    report_dir: str = "output/backtest",
) -> tuple[float, str]:
    """Resolve task temperature from explicit value or report file.

    Returns `(temperature, source)` where source is one of:
    - "manual": explicit CLI value
    - "report": loaded from calibration report
    - "default": fallback to 1.0
    """
    if explicit_temperature is not None:
        t = float(explicit_temperature)
        if t > 0:
            return t, "manual"
        return 1.0, "default"

    report_path = os.path.join(root, report_dir, f"{task}_test_temperature_report.json")
    t = load_best_temperature(report_path)
    if t is not None:
        return t, "report"

    return 1.0, "default"
