"""框内容质量评估：只阻断空白，不因成簇/相邻而拒绝。"""
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from TCM_Fusion.lib.box_quality import assess


class BoxQualityTests(unittest.TestCase):
    def test_blank_is_flagged_but_invalid_is_not_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            blank = root / 'blank.png'
            Image.new('RGB', (100, 80), (128, 128, 128)).save(blank)
            result = assess(blank)
            self.assertEqual(result['status'], 'blank')
            self.assertTrue(result['block_segment'])
            invalid = root / 'bad.png'
            invalid.write_bytes(b'not-an-image')
            result = assess(invalid)
            self.assertEqual(result['status'], 'invalid')
            self.assertFalse(result['block_segment'])

    def test_textured_crop_is_usable(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'texture.png'
            image = Image.new('RGB', (200, 120), 'white')
            draw = ImageDraw.Draw(image)
            for x in range(10, 190, 8):
                draw.line((x, 10, x + 30, 110), fill='black', width=2)
            image.save(path)
            result = assess(path)
            self.assertEqual(result['status'], 'usable')
            self.assertFalse(result['block_segment'])


if __name__ == '__main__':
    unittest.main()
