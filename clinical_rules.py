"""Microlab Liver Elastography — Clinical Rules Engine (backend).

Mirrors /app/frontend/src/lib/clinicalRules.js. Used by PDF rendering so the
printed report uses the exact same logic as the live UI.
"""

from typing import Any, Dict, List, Optional

ETIOLOGY_LABELS = {
    "nafld": "NAFLD / NASH",
    "hbv": "Hepatitis B",
    "hcv": "Hepatitis C",
    "alcohol": "Alcohol-related liver disease",
    "cholestatic": "Cholestatic liver disease",
    "mixed": "Mixed / uncertain etiology",
}

FIBROSIS_TABLES = {
    "nafld":       {"f01": (2.0, 7.0), "f2": (7.5, 10.0),  "f3": (10.1, 13.9), "f4": 14.0},
    "hbv":         {"f01": (2.0, 7.0), "f2": (8.0, 9.0),   "f3": (9.1, 11.9),  "f4": 12.0},
    "hcv":         {"f01": (2.0, 7.0), "f2": (8.0, 9.0),   "f3": (9.1, 13.9),  "f4": 14.0},
    "alcohol":     {"f01": (2.0, 7.0), "f2": (7.0, 11.0),  "f3": (11.1, 18.9), "f4": 19.0},
    "cholestatic": {"f01": (2.0, 7.0), "f2": (7.0, 9.0),   "f3": (9.1, 16.9),  "f4": 17.0},
}

MIXED_RANGES = {"low_max": 7.0, "intermediate_max": 13.9, "high": 14.0}

CAP_CUTOFFS = {
    "s0_max_inclusive": 237,
    "s1_max_inclusive": 259,
    "s2_max_inclusive": 290,
    "significant_threshold": 275,
}

CONFOUNDERS = [
    {"key": "acute_hepatitis", "warning": "Acute hepatitis or flare may overestimate liver stiffness."},
    {"key": "hepatic_congestion", "warning": "Heart failure or hepatic congestion may overestimate liver stiffness."},
    {"key": "cholestasis_obstruction", "warning": "Cholestasis may limit interpretation of stiffness values."},
    {"key": "focal_lesion", "warning": "Focal liver lesion may affect interpretation."},
    {"key": "obesity", "warning": "Obesity may reduce technical accuracy; XL probe recommended where available."},
    {"key": "ascites", "warning": "Ascites may reduce examination reliability."},
    {"key": "post_prandial", "warning": "Post-prandial state may affect liver stiffness measurement."},
]


def _num(v: Any) -> Optional[float]:
    if v in (None, ""):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def compute_bmi(weight_kg: Any, height_cm: Any) -> Optional[float]:
    w = _num(weight_kg)
    h = _num(height_cm)
    if not w or not h or h <= 0:
        return None
    m = h / 100
    return round(w / (m * m), 1)


def fibrosis_from_kpa(kpa: Any, etiology: str) -> Dict[str, str]:
    v = _num(kpa)
    if v is None:
        return {"stage": "", "label": "", "scale": ""}
    if not etiology or etiology == "mixed":
        if v <= MIXED_RANGES["low_max"]:
            return {"stage": "Low", "label": "Not suggestive of advanced fibrosis", "scale": "mixed"}
        if v < MIXED_RANGES["high"]:
            return {"stage": "Intermediate", "label": "Indeterminate to suggestive of significant fibrosis", "scale": "mixed"}
        return {"stage": "High", "label": "Suggestive of advanced fibrosis / cirrhosis in appropriate clinical context", "scale": "mixed"}
    t = FIBROSIS_TABLES.get(etiology)
    if not t:
        return {"stage": "", "label": "", "scale": ""}
    if v < t["f2"][0]:
        return {"stage": "F0–F1", "label": "No / mild fibrosis", "scale": "metavir"}
    if v <= t["f2"][1]:
        return {"stage": "F2", "label": "Significant fibrosis", "scale": "metavir"}
    if v <= t["f3"][1]:
        return {"stage": "F3", "label": "Advanced fibrosis", "scale": "metavir"}
    return {"stage": "F4", "label": "Cirrhosis", "scale": "metavir"}


def steatosis_from_cap(cap: Any) -> Dict[str, Any]:
    v = _num(cap)
    if v is None:
        return {"stage": "", "label": "", "significant": False}
    significant = v >= CAP_CUTOFFS["significant_threshold"]
    if v <= CAP_CUTOFFS["s0_max_inclusive"]:
        return {"stage": "S0", "label": "No significant steatosis", "significant": significant}
    if v <= CAP_CUTOFFS["s1_max_inclusive"]:
        return {"stage": "S1", "label": "Mild steatosis", "significant": significant}
    if v <= CAP_CUTOFFS["s2_max_inclusive"]:
        return {"stage": "S2", "label": "Moderate steatosis", "significant": significant}
    return {"stage": "S3", "label": "Severe steatosis", "significant": significant}


def quality_class(measurements: Dict[str, Any]) -> Dict[str, Any]:
    valid = _num(measurements.get("valid_measurements"))
    kpa = _num(measurements.get("kpa_median"))
    ratio = _num(measurements.get("iqr_median_ratio"))
    if ratio is None:
        iqr = _num(measurements.get("iqr"))
        if iqr is not None and kpa and kpa > 0:
            ratio = iqr / kpa
    sr = _num(measurements.get("success_rate"))

    if valid is None or kpa is None or valid < 10:
        return {
            "flag": "Suboptimal",
            "tone": "critical",
            "reason": (
                f"Only {int(valid)} valid measurements (target ≥ 10)."
                if valid is not None and valid < 10
                else "Key acquisition values missing or fewer than 10 valid measurements."
            ),
        }
    ratio_ok = None if ratio is None else ratio <= 0.30
    success_ok = None if sr is None else sr >= 60

    if ratio_ok is not False and success_ok is not False:
        return {
            "flag": "Reliable",
            "tone": "success",
            "reason": "≥ 10 valid measurements and IQR/Median ≤ 0.30.",
        }
    reasons = []
    if ratio_ok is False:
        reasons.append("IQR/Median > 0.30")
    if success_ok is False:
        reasons.append("success rate < 60%")
    return {
        "flag": "Acceptable with caution",
        "tone": "warning",
        "reason": " and ".join(reasons) + "." if reasons else "Caution.",
    }


def portal_hypertension_note(kpa_median: Any, platelet_count: Any) -> str:
    k = _num(kpa_median)
    p = _num(platelet_count)
    if k is None or p is None:
        return ""
    if k <= 15 and p >= 150:
        return "CSPH unlikely in appropriate compensated chronic liver disease setting (LSM ≤ 15 kPa & platelets ≥ 150 ×10⁹/L)."
    if k >= 25:
        return "CSPH likely in appropriate compensated chronic liver disease setting (LSM ≥ 25 kPa)."
    return "Portal hypertension risk indeterminate to increased depending on stiffness and platelet combination — clinical correlation advised."


def confounder_warnings(clinical: Dict[str, Any]) -> List[str]:
    return [c["warning"] for c in CONFOUNDERS if clinical.get(c["key"])]


def build_machine_impression(
    fibrosis: Dict[str, Any],
    steatosis: Dict[str, Any],
    quality: Dict[str, Any],
    etiology_label: str,
    cap: Any,
) -> str:
    lines = []
    if quality.get("flag") == "Suboptimal":
        lines.append(
            "Limited interpretation: examination quality is suboptimal. "
            "Definitive fibrosis staging deferred; consider repeating acquisition."
        )
    elif fibrosis.get("stage"):
        prefix = f" ({etiology_label} reference)" if etiology_label else ""
        lines.append(
            f"Liver stiffness compatible with {fibrosis['stage']} — {fibrosis['label']}{prefix}."
        )
    if steatosis.get("stage"):
        if steatosis.get("significant"):
            lines.append(
                f"CAP {cap if cap not in (None, '') else ''} dB/m — {steatosis['stage']} — "
                f"{steatosis['label']}. Significant steatosis likely."
            )
        else:
            lines.append(f"CAP — {steatosis['stage']} — {steatosis['label']}.")
    if quality.get("flag") == "Acceptable with caution":
        lines.append(
            "Acquisition quality acceptable with caution; clinical correlation advised."
        )
    return "\n".join(lines)
