"""One model per branch exports use the same frozen-fold routing contract."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cmi_project.cnn import CMI1DCNN, CNNConfig
from cmi_project.evaluation import ALL_GESTURES
from cmi_project.inference import RoutedCNNPredictor
from cmi_project.preprocessing import FoldPreprocessor
from cmi_project.dataset_analysis import SENSOR_COLUMNS
from test_preprocessing import fixture


class InferenceBundleTests(unittest.TestCase):
    def test_five_folds_single_members_route_without_labels_or_test_fit(self):
        torch.set_num_threads(1)
        raw = fixture(5, label=ALL_GESTURES[0])
        demographics = pd.DataFrame({"subject": ["S1"], "handedness": [1]})
        processor = FoldPreprocessor().fit([raw], demographics)
        state = json.loads(json.dumps(processor.state))
        state.pop("train_subjects")
        state.pop("train_sequence_ids")
        config = CNNConfig(imu_channels=(4, 8), auxiliary_channels=(4, 8), hidden_features=8)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            members, expected_a, expected_b = [], [], []
            for fold in range(5):
                for name, probabilities in (("imu", expected_a), ("multisensor", expected_b)):
                    model = CMI1DCNN(name, config=config).eval()
                    with torch.no_grad():
                        for parameter in model.parameters():
                            parameter.zero_()
                        bias = torch.linspace(-1, 1, 18) * (1 + fold / 5)
                        model.classifier[-1].bias.copy_(bias if name == "multisensor" else bias.flip(0))
                    probabilities.append(model.classifier[-1].bias.softmax(0).detach().numpy())
                    path = root / f"{name}_{fold}.pt"
                    member_state = json.loads(json.dumps(state))
                    if name == "multisensor":
                        member_state["config"] = processor.state["config"]
                    torch.save({"fold": fold, "folds_sha256": "fixed", "label_order": list(ALL_GESTURES),
                        "model_metadata": model.metadata(), "state_dict": model.state_dict(),
                        "preprocessor": member_state, "input_clip": 8.0}, path)
                    members.append({"path": path.name, "model": name, "fold": fold,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            manifest = {"version": 1, "label_order": list(ALL_GESTURES), "folds": list(range(5)),
                "folds_sha256": "fixed", "members_per_branch": 1, "members": members}
            (root / "bundle.json").write_text(json.dumps(manifest))
            predictor = RoutedCNNPredictor(root)
            raw["gesture"] = "ignored poison label"
            raw["behavior"] = "ignored phase annotation"
            np.testing.assert_allclose(predictor.predict_proba(raw, demographics), np.mean(expected_b, axis=0), atol=1e-7)
            raw[SENSOR_COLUMNS["THM"] + SENSOR_COLUMNS["ToF"]] = np.nan
            np.testing.assert_allclose(predictor.predict_proba(raw, demographics), np.mean(expected_a, axis=0), atol=1e-7)
            self.assertEqual(state, predictor.processors[0].state)
            changed = torch.load(path, weights_only=True)
            changed["preprocessor"]["statistics"]["imu"]["offset"][0] += 0.001
            torch.save(changed, path)
            manifest["members"][-1]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            (root / "bundle.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "identical fold-fitted inputs"):
                RoutedCNNPredictor(root)


if __name__ == "__main__":
    unittest.main()
