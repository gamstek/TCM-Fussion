"""数据集划分回归：group 隔离、确定性、显式覆盖与泄漏检查。
该组用例由项目根 selftest 发现。涉及模型回复的用例使用可控输入，涉及进程的用例以各断言规定的执行边界为准。

主要定义：
- `_write_attributes`: 写出含 image/herb/feature_id/group_key 的 `attributes.csv` 测试夹具。
- DatasetSplitTest.test_groups_are_isolated_and_deterministic: 同 seed 两次划分一致，group 全覆盖且 summary 计数一致。
- test_explicit_override_and_unknown_group: 显式指定划分生效，未知 group 报错。
- test_same_image_in_two_groups_is_rejected、test_empty_split_is_rejected: 跨组泄漏与空划分（无 test 等）被拒。

关联：
- 直接依赖 「lib/offline/dataset_split.py」、「lib/runtime/storage.py」。
- 统一执行见 tests/README.md。
"""
import csv
import shutil
import unittest
import uuid
from pathlib import Path

import paths
from TCM_Fusion.lib.offline import dataset_split as S
from TCM_Fusion.lib.runtime.storage import IntegrationError


def _write_attributes(directory, rows):
    path = Path(directory) / "attributes.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image", "herb", "feature_id", "group_key"])
        writer.writeheader()
        writer.writerows(rows)
    return path


class DatasetSplitTest(unittest.TestCase):
    def setUp(self):
        base = Path(paths.PROJECT_ROOT) / "results" / "ui_out" / "fusion" / "verification"
        base.mkdir(parents=True, exist_ok=True)
        self.dir = base / ("split_" + uuid.uuid4().hex[:8])
        self.dir.mkdir(parents=True, exist_ok=False)
        self.rows = [
            {"image": f"img{i}.jpg", "herb": "黄芩" if i % 2 else "黄芪",
             "feature_id": "fiber", "group_key": f"g{i // 2}"}
            for i in range(20)
        ]

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_groups_are_isolated_and_deterministic(self):
        path = _write_attributes(self.dir, self.rows)
        first = S.build([str(path)], seed=7, ratios=(0.6, 0.2, 0.2))
        second = S.build([str(path)], seed=7, ratios=(0.6, 0.2, 0.2))
        self.assertEqual(first["groups"], second["groups"])
        self.assertEqual(set(first["groups"]), {f"g{i}" for i in range(10)})
        for group, row in first["groups"].items():
            self.assertEqual(S.split_of(first, group), row["split"])
            self.assertTrue(row["images"])
        self.assertEqual(sum(v["groups"] for v in first["summary"].values()), 10)

    def test_explicit_override_and_unknown_group(self):
        path = _write_attributes(self.dir, self.rows)
        value = S.build([str(path)], seed=7, explicit={"g0": "test", "g1": "development"})
        self.assertEqual(value["groups"]["g0"]["split"], "test")
        self.assertEqual(value["groups"]["g1"]["split"], "development")
        with self.assertRaises(IntegrationError):
            S.assign({"g0": {"images": ["a"], "herb": set(), "features": set()}},
                     explicit={"unknown": "test"})

    def test_same_image_in_two_groups_is_rejected(self):
        rows = list(self.rows)
        rows[0] = {**rows[0], "group_key": "g0"}
        rows[1] = {**rows[1], "image": rows[0]["image"], "group_key": "g1"}
        path = _write_attributes(self.dir, rows)
        with self.assertRaises(IntegrationError):
            S.read_groups([str(path)])

    def test_empty_split_is_rejected(self):
        path = _write_attributes(self.dir, self.rows)
        explicit = {f"g{i}": "training" for i in range(10)}
        with self.assertRaises(IntegrationError):
            S.assign(S.read_groups([str(path)]), explicit=explicit)


if __name__ == "__main__":
    unittest.main()
