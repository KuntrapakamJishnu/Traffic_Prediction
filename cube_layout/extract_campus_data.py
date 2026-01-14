"""
Campus road network is modeled using a region-based abstraction
for traffic flow analysis due to lack of public sensor-level data.
"""

import logging
import sys
import traci
import pandas as pd

SUMO_CMD = ["sumo", "-c", "cube.sumocfg"]

# Full set of logical edge IDs from the 5x5 grid (both directions)
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

SIM_TIME = 3600


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    rows = []

    try:
        logging.info("Starting SUMO via TraCI: %s", " ".join(SUMO_CMD))
        traci.start(SUMO_CMD)

        # determine which of the requested edges actually exist in the net
        available = set(traci.edge.getIDList())
        monitored = [e for e in EDGES if e in available]
        missing = [e for e in EDGES if e not in available]
        if missing:
            logging.warning("%d requested edges not present in network: %s", len(missing), ",".join(missing[:10]) + ("..." if len(missing)>10 else ""))
        logging.info("Monitoring %d edges", len(monitored))

        for t in range(SIM_TIME):
            traci.simulationStep()

            for edge in monitored:
                # last-step vehicle count and mean speed
                try:
                    flow = traci.edge.getLastStepVehicleNumber(edge)
                    speed = traci.edge.getLastStepMeanSpeed(edge)
                except traci.TraCIException:
                    # rare case: edge disappears or internal id mismatch
                    flow = 0
                    speed = 0.0

                # simple occupancy approximation (normalize flow to [0,1])
                occupy = min(flow / 60.0, 1.0)

                rows.append([t, edge, int(flow), float(occupy), float(speed)])

    except Exception as exc:
        logging.exception("SUMO/TraCI error: %s", exc)
    finally:
        try:
            traci.close()
        except Exception:
            pass

    if rows:
        df = pd.DataFrame(rows, columns=["timestep", "location", "flow", "occupy", "speed"])
        out_csv = "cube_city_traffic.csv"
        df.to_csv(out_csv, index=False)
        logging.info(" %s created (%d rows)", out_csv, len(df))
    else:
        logging.warning("No data collected; CSV not created.")


if __name__ == "__main__":
    main()
