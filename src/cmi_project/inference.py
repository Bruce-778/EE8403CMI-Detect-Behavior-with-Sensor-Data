"""Frozen fold ensembles for the official one-sequence-at-a-time CMI API."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .cnn import CMI1DCNN, CNNConfig
from .cnn_data import ARRAY_KEYS, compact_sample
from .evaluation import ALL_GESTURES
from .preprocessing import FoldPreprocessor, PreprocessingConfig, preprocessor_states_equal


def as_pandas(frame) -> pd.DataFrame:
    if isinstance(frame, pd.DataFrame):
        return frame
    if hasattr(frame, "to_dicts"):
        # Polars -> pandas without requiring pyarrow in the runtime.
        return pd.DataFrame(frame.to_dicts())
    raise TypeError("Expected a pandas or Polars DataFrame.")


class RoutedCNNPredictor:
    """Average the five frozen folds, routing A/B independently within each fold.

    Each branch averages one or more explicitly selected CNN probabilities.
    Test inputs never fit normalization, change weights or select checkpoints.
    """

    def __init__(self, bundle_dir: Path, *, device: str = "cpu", cpu_threads: int = 2):
        self.directory = Path(bundle_dir)
        self.device = torch.device(device)
        torch.set_num_threads(cpu_threads)
        manifest = json.loads((self.directory / "bundle.json").read_text(encoding="utf-8"))
        if manifest.get("version") != 1 or manifest.get("label_order") != list(ALL_GESTURES):
            raise ValueError("Incompatible inference bundle/class order.")
        if manifest.get("folds") != list(range(5)):
            raise ValueError("Submission requires all five frozen folds.")
        self.folds_sha256 = manifest["folds_sha256"]
        member_count = manifest.get("members_per_branch", 2)
        if not isinstance(member_count, int) or member_count < 1:
            raise ValueError("Invalid members_per_branch.")
        self.members = defaultdict(lambda: defaultdict(list))
        self.processors, self.inputs = {}, {}
        for member in manifest["members"]:
            path = self.directory / member["path"]
            if self.directory.resolve() not in path.resolve().parents:
                raise ValueError("Checkpoint path escapes the bundle.")
            if hashlib.sha256(path.read_bytes()).hexdigest() != member["sha256"]:
                raise ValueError("Checkpoint hash mismatch.")
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            fold, name = checkpoint["fold"], checkpoint["model_metadata"]["model"]
            if (checkpoint["folds_sha256"] != self.folds_sha256
                    or checkpoint["label_order"] != list(ALL_GESTURES)
                    or fold != member["fold"] or name != member["model"]):
                raise ValueError("Checkpoint fold/model/class order mismatch.")
            meta = checkpoint["model_metadata"]
            model = CMI1DCNN(name, tof_regions=meta["tof_regions"],
                config=CNNConfig.from_dict(meta["config"])).to(self.device)
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            state = checkpoint["preprocessor"]
            inputs = {"tof_regions": meta["tof_regions"], "input_clip": checkpoint["input_clip"]}
            if fold in self.processors:
                if not preprocessor_states_equal(state, self.processors[fold].state) or inputs != self.inputs[fold]:
                    raise ValueError("Ensemble members require identical fold-fitted inputs.")
            else:
                processor = FoldPreprocessor(PreprocessingConfig.from_dict(state["config"]))
                processor.state = state
                self.processors[fold], self.inputs[fold] = processor, inputs
            self.members[fold][name].append(model)
        for fold in range(5):
            if set(self.members[fold]) != {"imu", "multisensor"}:
                raise ValueError("Each fold requires Model A and Model B.")
            if any(len(models) != member_count for models in self.members[fold].values()):
                raise ValueError("Each branch requires the declared number of equal-weight CNNs.")

    @torch.inference_mode()
    def predict_proba(self, sequence, demographics, *, fold: int | None = None) -> np.ndarray:
        sequence = as_pandas(sequence).copy()
        demographics = as_pandas(demographics)
        # Do not consume labels/phase annotations even when doing a local check.
        sequence = sequence.drop(columns=["gesture", "sequence_type", "behavior", "orientation"], errors="ignore")
        folds = list(range(5)) if fold is None else [fold]
        if not set(folds).issubset(self.members):
            raise ValueError("Unknown fold.")
        probabilities = []
        for number in folds:
            sample = compact_sample(self.processors[number].transform(sequence, demographics, "test"),
                                    **self.inputs[number])
            name = "multisensor" if sample["thm_valid"].any() or sample["tof_sensor_present"].any() else "imu"
            batch = {key: torch.from_numpy(np.asarray(sample[key]).copy()).unsqueeze(0).to(self.device)
                     for key in ARRAY_KEYS}
            branch = torch.stack([model(batch).softmax(1) for model in self.members[number][name]]).mean(0)
            probabilities.append(branch[0].cpu().numpy())
        result = np.mean(probabilities, axis=0)
        if not np.isfinite(result).all() or (result < 0).any() or result.sum() <= 0:
            raise ValueError("Invalid inference probabilities.")
        return result / result.sum()

    def predict(self, sequence, demographics) -> str:
        return str(ALL_GESTURES[int(self.predict_proba(sequence, demographics).argmax())])
