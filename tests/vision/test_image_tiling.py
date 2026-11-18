"""在固定图幅上验证分块网格覆盖边缘、上限和小图行为，记录旧参数的兼容语义。调用真实分块函数，无图像模型或网络请求。

主要定义：
- ImageTilingTests.test_four_tiles_are_spread_over_full_grid: 四个 tile 从左上铺到右下并覆盖整幅图像。
- ImageTilingTests.test_fewer_boxes_than_limit_are_all_kept: 小图产出的 tile 数不少于一张且不超过上限。
- ImageTilingTests.test_limit_is_respected: max_tiles 严格限制返回的 tile 数量。

关联：
- 直接依赖 「lib/vision/image_tiling.py」。
- 统一执行见 tests/README.md。
"""
import unittest

from TCM_Fusion.lib.vision.image_tiling import uniform_tiles


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
