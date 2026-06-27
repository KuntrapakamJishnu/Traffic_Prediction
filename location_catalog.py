from __future__ import annotations

import re
from typing import Iterable

import pandas as pd


AMARAVATI_MAP_CENTER = (16.4785, 80.5600)


AMARAVATI_LOCATION_POI = {
    "VIT-AP Main Gate": {"lat": 16.4785, "lon": 80.5600},
    "VIT-AP Academic Block": {"lat": 16.4794, "lon": 80.5620},
    "VIT-AP Hostel Zone": {"lat": 16.4776, "lon": 80.5635},
    "VIT-AP Library": {"lat": 16.4802, "lon": 80.5646},
    "VIT-AP Admin Block": {"lat": 16.4814, "lon": 80.5592},
    "Amaravati Secretariat": {"lat": 16.4927, "lon": 80.5235},
    "Inavolu Junction": {"lat": 16.4708, "lon": 80.5515},
    "Tadepalli Riverfront": {"lat": 16.5052, "lon": 80.5938},
}


AMARAVATI_HEATMAP_LOCATIONS = [
    {"location": "VIT-AP Main Gate Road 0", "lat": 16.4785, "lon": 80.5600},
    {"location": "VIT-AP Academic Block Road 1", "lat": 16.4794, "lon": 80.5620},
    {"location": "VIT-AP Hostel Zone Road 2", "lat": 16.4776, "lon": 80.5635},
    {"location": "VIT-AP Library Road 3", "lat": 16.4802, "lon": 80.5646},
    {"location": "VIT-AP Admin Block Road 4", "lat": 16.4814, "lon": 80.5592},
    {"location": "Amaravati Secretariat Road 5", "lat": 16.4927, "lon": 80.5235},
    {"location": "Inavolu Junction Road 6", "lat": 16.4708, "lon": 80.5515},
    {"location": "Tadepalli Riverfront Road 7", "lat": 16.5052, "lon": 80.5938},
]


# Approximate Vijayawada / Amaravati node areas used to turn raw SUMO edge IDs
# into readable street/area labels in analytics and live views.
VIJAYAWADA_NODE_AREAS = [
    ["Neerukonda", "VIT-AP Main Gate", "Mangalagiri Highway", "NH16 Service Road", "Amaravati Seed Access Road"],
    ["Undavalli", "Tadepalli", "Mangalagiri Bypass", "Kanuru", "Poranki"],
    ["Labbipet", "Benz Circle", "MG Road", "Eluru Road", "Governorpet"],
    ["Satyanarayanapuram", "One Town", "Vijayawada Railway Station", "Bandar Road", "Bezawada Canal Road"],
    ["Patamata", "Auto Nagar", "Gunadala", "Ibrahimpatnam", "Kanchikacherla Link"],
]


LEGACY_PREFIX_RULES = [
    (re.compile(r"^Dhanbad\s+CBD", re.IGNORECASE), "VIT-AP Main Gate"),
    (re.compile(r"^Baliapur", re.IGNORECASE), "VIT-AP Academic Block"),
    (re.compile(r"^Govindpur", re.IGNORECASE), "VIT-AP Hostel Zone"),
    (re.compile(r"^Hirapur", re.IGNORECASE), "VIT-AP Library"),
    (re.compile(r"^IIT[-\s]?ISM(?:\s+Campus)?", re.IGNORECASE), "VIT-AP Admin Block"),
    (re.compile(r"^Jharia", re.IGNORECASE), "Amaravati Secretariat"),
    (re.compile(r"^Katras", re.IGNORECASE), "Inavolu Junction"),
    (re.compile(r"^Saraidhela", re.IGNORECASE), "Tadepalli Riverfront"),
]


LEGACY_NUMERIC_LABELS = {
    0: "VIT-AP Main Gate",
    1: "VIT-AP Academic Block",
    2: "VIT-AP Hostel Zone",
    3: "VIT-AP Library",
    4: "VIT-AP Admin Block",
    5: "Amaravati Secretariat",
    6: "Inavolu Junction",
    7: "Tadepalli Riverfront",
}


EDGE_ID_PATTERN = re.compile(r"^n(\d)(\d)_n(\d)(\d)$")


def describe_edge_location(value):
    if value is None:
        return value

    text = str(value).strip()
    match = EDGE_ID_PATTERN.match(text)
    if not match:
        return text

    row_a, col_a, row_b, col_b = (int(part) for part in match.groups())
    try:
        area_a = VIJAYAWADA_NODE_AREAS[row_a][col_a]
        area_b = VIJAYAWADA_NODE_AREAS[row_b][col_b]
    except Exception:
        return text

    if area_a == area_b:
        return area_a
    return f"{area_a} - {area_b}"


def normalize_location_label(value):
    if value is None:
        return value
    try:
        if pd.isna(value):
            return value
    except Exception:
        pass

    text = str(value).strip()
    if not text:
        return text

    try:
        numeric_value = int(text)
        if str(numeric_value) == text and numeric_value in LEGACY_NUMERIC_LABELS:
            return LEGACY_NUMERIC_LABELS[numeric_value]
    except Exception:
        pass

    for pattern, replacement in LEGACY_PREFIX_RULES:
        if pattern.search(text):
            return pattern.sub(replacement, text, count=1)

    edge_label = describe_edge_location(text)
    if edge_label != text:
        return edge_label

    return text


def normalize_location_frame(df: pd.DataFrame, columns: Iterable[str] = ("location", "location_name")) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    for column in columns:
        if column in df.columns:
            df[column] = df[column].apply(normalize_location_label)
    return df


def get_amaravati_coords(value):
    label = normalize_location_label(value)
    if label is None:
        return None

    label_text = str(label)
    for prefix, meta in AMARAVATI_LOCATION_POI.items():
        if label_text.startswith(prefix):
            return meta["lat"], meta["lon"]
    return None


def amaravati_heatmap_locations():
    return list(AMARAVATI_HEATMAP_LOCATIONS)
