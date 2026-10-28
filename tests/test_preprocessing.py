import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.data.prepare import fit_transformer, overlaps, prepare, split_groups, transform


class PreprocessingTests(unittest.TestCase):
    def test_group_split_is_deterministic_and_disjoint(self):
        groups = np.repeat(np.arange(100), 3)
        fractions = dict(train=.6, validation=.15, calibration=.1, test=.15)
        a = split_groups(groups, fractions, 42)
        np.testing.assert_array_equal(a, split_groups(groups, fractions, 42))
        self.assertFalse(any(overlaps(groups, a).values()))
        self.assertEqual((a == "train").sum(), 180)

    def test_train_statistics_and_unknown_category(self):
        train = pd.DataFrame({"x": [0., 2., np.nan], "proto": ["tcp", "udp", "-"]})
        state = fit_transformer(train, ["x"], ["proto"])
        self.assertEqual(state["median"]["x"], 1.)
        self.assertEqual(state["mean"]["x"], 1.)
        before = json.dumps(state)
        test = transform(pd.DataFrame({"x": [1000., np.nan], "proto": ["new", "-"]}), state)
        self.assertEqual(test.shape, (2, 5))
        self.assertEqual(test[0, 2], 1.)  # unknown, not an all-zero one-hot vector
        self.assertEqual(test[1, 1], 1.)  # missing
        self.assertEqual(test[1, 0], 0.)
        self.assertEqual(before, json.dumps(state))

    def test_invalid_split_fails(self):
        with self.assertRaises(ValueError):
            split_groups(np.arange(2), dict(train=.6, validation=.15, calibration=.1, test=.15), 42)

    def test_end_to_end_smoke_and_repeat(self):
        cfg = json.loads(Path("configs/data.json").read_text())
        rows = []
        for group in range(100):
            for label in [0, 1]:
                row = {col: group + label for col in cfg["numeric_features"]}
                row.update(src_ip=f"10.0.0.{group}", dst_ip="10.1.0.1", src_port=group,
                           dst_port=80, proto="tcp", service="-", conn_state="SF",
                           label=label, type="normal" if label == 0 else "attack")
                rows.append(row)
        rows.append(rows[0].copy())
        with tempfile.TemporaryDirectory(dir=".") as tmp:
            root = Path(tmp)
            source = root / "input.csv"
            pd.DataFrame(rows).to_csv(source, index=False)
            before = source.read_bytes()
            first = prepare(source, "configs/data.json", root / "a")
            second = prepare(source, "configs/data.json", root / "b")
            self.assertEqual(first, second)
            self.assertEqual(first["exact_duplicates_removed"], 1)
            self.assertEqual(first["retained_rows"], 200)
            self.assertEqual(source.read_bytes(), before)
            mapping = pd.read_csv(root / "a/duplicate_mapping.csv")
            self.assertEqual(mapping.iloc[-1].representative_row_index, 0)
            seen_ids = set()
            for split in cfg["split_fractions"]:
                with np.load(root / f"a/{split}.npz", allow_pickle=False) as a, np.load(root / f"b/{split}.npz", allow_pickle=False) as b:
                    for key in a.files:
                        np.testing.assert_array_equal(a[key], b[key])
                    self.assertFalse(seen_ids & set(a["row_id"]))
                    seen_ids.update(a["row_id"])
                    self.assertTrue(np.isfinite(a["X"]).all())
            self.assertEqual(len(seen_ids), 200)

    def test_forbidden_label_feature_and_overwrite_fail(self):
        cfg = json.loads(Path("configs/data.json").read_text())
        with tempfile.TemporaryDirectory(dir=".") as tmp:
            path = Path(tmp) / "config.json"
            cfg["numeric_features"].append("label")
            path.write_text(json.dumps(cfg))
            with self.assertRaises(ValueError):
                prepare("nonexistent.csv", path, Path(tmp) / "new")
            with self.assertRaises(FileExistsError):
                prepare("nonexistent.csv", path, tmp)


if __name__ == "__main__":
    unittest.main()
