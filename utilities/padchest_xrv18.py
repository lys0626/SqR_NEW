import ast
import re

import numpy as np
import pandas as pd


PADCHEST_XRV16_LABELS = [
    "Atelectasis",
    "Consolidation",
    "Infiltration",
    "Pneumothorax",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Effusion",
    "Pneumonia",
    "Pleural_Thickening",
    "Cardiomegaly",
    "Nodule",
    "Mass",
    "Hernia",
    "Fracture",
    "Lung Opacity",
]


PADCHEST_XRV16_ALIASES = {
    "Atelectasis": ["atelectasis", "atelectases"],
    "Consolidation": ["consolidation", "air bronchogram"],
    "Infiltration": [
        "infiltration",
        "infiltrates",
        "interstitial pattern",
        "reticular interstitial pattern",
        "reticulonodular interstitial pattern",
    ],
    "Pneumothorax": ["pneumothorax"],
    "Edema": ["edema", "pulmonary edema"],
    "Emphysema": ["emphysema"],
    "Fibrosis": ["fibrosis", "pulmonary fibrosis"],
    "Effusion": ["effusion", "pleural effusion"],
    "Pneumonia": ["pneumonia"],
    "Pleural_Thickening": ["pleural thickening", "pleural_thickening"],
    "Cardiomegaly": ["cardiomegaly"],
    "Nodule": ["nodule", "pulmonary nodule"],
    "Mass": ["mass", "pulmonary mass"],
    "Hernia": ["hernia", "hiatal hernia"],
    "Fracture": ["fracture", "rib fracture"],
    "Lung Opacity": [
        "lung opacity",
        "opacity",
        "airspace opacity",
        "alveolar pattern",
        "ground glass pattern",
    ],
}


def normalize_padchest_term(value):
    value = str(value).strip().lower()
    value = value.replace("_", " ").replace("-", " ")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def parse_padchest_label_terms(value):
    if value is None:
        return []
    try:
        if pd.isna(value):
            return []
    except TypeError:
        pass

    raw_items = value
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"nan", "none", "null"}:
            return []
        try:
            raw_items = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            raw_items = re.split(r"[|;,\n]", text)

    if isinstance(raw_items, dict):
        raw_items = raw_items.keys()
    elif not isinstance(raw_items, (list, tuple, set)):
        raw_items = [raw_items]

    terms = []
    for item in raw_items:
        term = normalize_padchest_term(item)
        if term and term not in {"nan", "none", "null"}:
            terms.append(term)
    return terms


def project_padchest_row_to_xrv16(row):
    terms = []
    for col in ("Labels", "labels", "label", "Findings", "findings"):
        if col in row:
            terms.extend(parse_padchest_label_terms(row.get(col)))

    for target, aliases in PADCHEST_XRV16_ALIASES.items():
        for alias in aliases + [target]:
            for col in {alias, alias.replace(" ", "_"), alias.title(), target}:
                if col in row and _is_positive_value(row.get(col)):
                    terms.append(alias)

    normalized_terms = {normalize_padchest_term(term) for term in terms}
    labels = np.zeros(len(PADCHEST_XRV16_LABELS), dtype=np.float32)
    for idx, target in enumerate(PADCHEST_XRV16_LABELS):
        aliases = {normalize_padchest_term(alias) for alias in PADCHEST_XRV16_ALIASES[target]}
        if normalized_terms & aliases:
            labels[idx] = 1.0
    return labels


def _is_positive_value(value):
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except TypeError:
        pass
    if isinstance(value, str):
        return value.strip().lower() in {"1", "1.0", "true", "yes", "y", "positive", "pos"}
    try:
        return float(value) > 0
    except (TypeError, ValueError):
        return False
