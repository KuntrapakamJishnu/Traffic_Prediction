"""
Generate a JSON mapping of SUMO edge IDs (e.g. 'n00_n01') to lat/lon coordinates.
This uses a simple grid layout and writes to `cube_layout/location_coords.json`.
Run this from the project root with the venv active:

python scripts/generate_location_coords.py

The generated JSON will be consumed by `pages/Traffic_Map.py` if present.
"""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "cube_layout" / "location_coords.json"

# grid size used in extract_cube_data.py (5x5)
GRID = 5
# base lat/lon and step (tweak as needed)
BASE_LAT = 12.9716
BASE_LON = 77.5946
STEP_LAT = 0.005
STEP_LON = 0.005

# build node coordinates nXY where X=row, Y=col with X,Y in [0,4]
nodes = {}
for x in range(GRID):
    for y in range(GRID):
        name = f"n{x}{y}"
        lat = BASE_LAT + (x * STEP_LAT)
        lon = BASE_LON + (y * STEP_LON)
        nodes[name] = (lat, lon)

# edges (same order as extract_cube_data.py)
EDGES = [
    # horizontal eastbound
    "n00_n01","n01_n02","n02_n03","n03_n04",
    "n10_n11","n11_n12","n12_n13","n13_n14",
    "n20_n21","n21_n22","n22_n23","n23_n24",
    "n30_n31","n31_n32","n32_n33","n33_n34",
    "n40_n41","n41_n42","n42_n43","n43_n44",
    # horizontal westbound
    "n01_n00","n02_n01","n03_n02","n04_n03",
    "n11_n10","n12_n11","n13_n12","n14_n13",
    "n21_n20","n22_n21","n23_n22","n24_n23",
    "n31_n30","n32_n31","n33_n32","n34_n33",
    "n41_n40","n42_n41","n43_n42","n44_n43",
    # vertical southbound
    "n00_n10","n10_n20","n20_n30","n30_n40",
    "n01_n11","n11_n21","n21_n31","n31_n41",
    "n02_n12","n12_n22","n22_n32","n32_n42",
    "n03_n13","n13_n23","n23_n33","n33_n43",
    "n04_n14","n14_n24","n24_n34","n34_n44",
    # vertical northbound
    "n10_n00","n20_n10","n30_n20","n40_n30",
    "n11_n01","n21_n11","n31_n21","n41_n31",
    "n12_n02","n22_n12","n32_n22","n42_n32",
    "n13_n03","n23_n13","n33_n23","n43_n33",
    "n14_n04","n24_n14","n34_n24","n44_n34",
]

mapping = {}
for e in EDGES:
    try:
        a, b = e.split("_")
        # ensure node prefixes: e is like n00_n01; split into n00 and n01
        # convert node names to our node keys (nXY without underscore)
        n1 = a
        n2 = b
        lat1, lon1 = nodes[n1]
        lat2, lon2 = nodes[n2]
        mapping[e] = [(lat1 + lat2) / 2.0, (lon1 + lon2) / 2.0]
    except Exception:
        # skip unknown patterns
        continue

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(mapping, f, indent=2)

print(f"Wrote {len(mapping)} locations to {OUT}")
