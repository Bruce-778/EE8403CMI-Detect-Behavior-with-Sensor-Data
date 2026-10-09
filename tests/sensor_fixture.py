"""Raw sensor fixture shared by dataset/fold integrity checks."""
import numpy as np
import pandas as pd
from cmi_project.dataset_analysis import ALL_SENSOR_COLUMNS


def fixture(length=4, sequence_id="A", subject="S1", label="G1"):
    values = {column: np.ones(length) for column in ALL_SENSOR_COLUMNS}
    values.update({"acc_x": np.zeros(length), "acc_y": np.zeros(length),
                   "acc_z": np.full(length, 9.80665), "rot_w": np.ones(length),
                   "rot_x": np.zeros(length), "rot_y": np.zeros(length), "rot_z": np.zeros(length)})
    values.update({"sequence_id": [sequence_id] * length, "subject": [subject] * length,
                   "sequence_counter": np.arange(length)})
    if label is not None:
        values["gesture"] = [label] * length
    return pd.DataFrame(values)
