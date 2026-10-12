import json
from pathlib import Path

from src.generator import load_or_generate

FMT = "%d/%m/%Y %H:%M:%S"
MAPPING = {
    "columns": {
        "shipment_id": "BookingNo", "origin": "POL", "destination": "POD",
        "dep_sched_ts": "ETD", "dep_actual_ts": "ATD",
        "port_arr_sched_ts": "ETA_Port", "port_arrival_actual_ts": "ATA_Port",
        "customs_sched_ts": "ETA_Customs", "customs_cleared_actual_ts": "ATA_Customs",
        "delivery_sched_ts": "ETA_Final", "delivery_actual_ts": "ATA_Final",
        "slack_transit_h": "TransitBufferHrs", "slack_port_h": "PortBufferHrs", "slack_inland_h": "InlandBufferHrs",
        "dest_congestion_arr": "PortCongestionIdx", "dest_congestion_dep": "PortCongestionIdx_atETD",
    },
    "timestamp_format": FMT, "timezone": "UTC",
    "holidays": ["2025-01-01", "2025-04-18", "2025-04-21", "2025-05-01", "2025-05-29", "2025-12-25"],
}


def to_foreign(df):
    import pandas as pd
    cols = MAPPING["columns"]
    src = {"shipment_id": "shipment_id", "origin": "origin", "destination": "destination",
           "dep_sched_ts": "dep_sched_ts", "dep_actual_ts": "dep_actual_ts", "port_arr_sched_ts": "port_arr_sched_ts",
           "port_arrival_actual_ts": "port_arrival_actual_ts", "customs_sched_ts": "customs_sched_ts",
           "customs_cleared_actual_ts": "customs_cleared_actual_ts", "delivery_sched_ts": "delivery_sched_ts",
           "delivery_actual_ts": "delivery_actual_ts", "slack_transit_h": "slack_transit_h",
           "slack_port_h": "slack_port_h", "slack_inland_h": "slack_inland_h",
           "dest_congestion_arr": "dest_congestion_arr", "dest_congestion_dep": "dest_congestion_dep"}
    out = pd.DataFrame(index=df.index)
    for canon, yours in cols.items():
        s = df[src[canon]]
        out[yours] = s.dt.strftime(FMT) if str(s.dtype).startswith("datetime") else s
    return out          # note: no weather, no inspection, no origin congestion, no ground truth


if __name__ == "__main__":
    here = Path(__file__).parent
    to_foreign(load_or_generate()).to_csv(here / "foreign_shipments.csv", index=False)
    json.dump(MAPPING, open(here / "mapping_example.json", "w", encoding="utf-8"), indent=2)
    print("wrote", here / "foreign_shipments.csv", "and", here / "mapping_example.json")
