"""全视野 tile 均匀采样。"""
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
