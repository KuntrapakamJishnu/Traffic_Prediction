import time
import pandas as pd
import requests
from collections import defaultdict, deque

API_URL = "http://127.0.0.1:8000/predict"
CSV_PATH = "cube_layout/campus_city_traffic.csv"

# how many timesteps the model expects
TIME_STEPS = 12

def stream_latest_per_location(csv_path=CSV_PATH, api_url=API_URL, per_location_delay=0.1, interval=60):
    """Send the most recent TIME_STEPS window for each location once per `interval` seconds.
    This avoids flooding the API with many sliding-window requests.
    """
    df = pd.read_csv(csv_path)
    df = df.sort_values(["location", "timestep"]) 

    locations = sorted(df["location"].unique())
    print(f"Streaming {len(locations)} locations every {interval}s")

    # build per-location list
    groups = {loc: grp[["flow","occupy","speed"]].values.tolist() for loc, grp in df.groupby("location")}

    while True:
        for loc in locations:
            seq = groups.get(loc, [])
            if len(seq) < TIME_STEPS:
                # not enough data yet
                continue

            window = seq[-TIME_STEPS:]
            payload = {"sequence": window, "location": str(loc)}

            try:
                resp = requests.post(api_url, json=payload, timeout=10)
                if resp.status_code == 200:
                    print(f"Sent {loc} → OK")
                else:
                    print(f"Sent {loc} → {resp.status_code}")
            except Exception as exc:
                print(f"Request error for {loc}: {exc}")

            time.sleep(per_location_delay)

        # wait until next cycle
        time.sleep(interval)


if __name__ == "__main__":
    # default: send each location once per minute, small delay between locations
    stream_latest_per_location(per_location_delay=0.1, interval=60)
