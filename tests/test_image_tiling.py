"""全视野 tile 均匀采样回归：覆盖范围、数量上限与小图退化行为。

主要定义：
- ImageTilingTests.test_four_tiles_are_spread_over_full_grid: 四个 tile 从左上铺到右下并覆盖整幅图像。
- ImageTilingTests.test_fewer_boxes_than_limit_are_all_kept: 小图产出的 tile 数不少于一张且不超过上限。
- ImageTilingTests.test_limit_is_respected: max_tiles 严格限制返回的 tile 数量。

关联：
- TCM_Fusion.lib.image_tiling.uniform_tiles: 被测试的 tile 采样函数。
- 位于 tests/ 下，随 selftest 递归发现。
"""
import unittest

from TCM_Fusion.lib.image_tiling import uniform_tiles


class ImageTilingTests(unittest.TestCase):
    def test_four_tiles_are_spread_over_full_grid(self):
        boxes = uniform_tiles(2048, 1536, tile=1024, overlap=.15, max_tiles=4)
        self.assertEqual(len(boxes), 4)
        self.assertEqual(boxes[0][:2], [0, 0])
        self.assertEqual(boxes[-1][2:], [2048, 1536])
        self.assertGreater(boxes[-1][0], 0)
        self.assertGreater(boxes[-1][1], 0)

    def test_fewer_boxes_than_limit_are_all_kept(self):
        boxes = uniform_tiles(1024, 768, tile=1024, overlap=.15, max_tiles=4)
        self.assertGreaterEqual(len(boxes), 1)
        self.assertLessEqual(len(boxes), 4)

    def test_limit_is_respected(self):
        boxes = uniform_tiles(2048, 1536, tile=512, overlap=.15, max_tiles=4)
        self.assertEqual(len(boxes), 4)


if __name__ == '__main__':
    unittest.main()
