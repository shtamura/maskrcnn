import unittest

from keras import backend as K
import numpy as np
import xrcnn.util.bbox as bbox
from test.generate_random_bbox import generate_random_bbox


class TestBbox(unittest.TestCase):
    def setUp(self):
        self.src_bbox = generate_random_bbox(8, (64, 32), 4, 16)
        self.dst_bbox = self.src_bbox + 1

    def test_restore_bbox(self):
        offset = bbox.get_offset(self.src_bbox, self.dst_bbox)
        out_raw_bbox = bbox.get_bbox(self.src_bbox, offset)

        np.testing.assert_almost_equal(
            K.get_value(out_raw_bbox), K.get_value(self.dst_bbox), decimal=5)

    def test_get_iou(self):
        gtbox = K.variable([[1, 1, 3, 3], [2, 2, 4, 4]])
        anchor = K.variable([
            [1, 1, 3, 3],  # gtbox[0]とは完全に一致。つまりIoU=1。
            # gtbox[1]とは1/4重なる。つまりIoU=1/7。
            [1, 0, 3, 2],  # gtbox[0]とは半分重なる。つまりIoU=1/3。
            [2, 2, 4, 4],  # gtbox[0]とは1/4重なる。つまりIoU=1/7。gtbox[1]とは一致。
            [0, 3, 2, 5],  # gtbox[0]とは隣接。
            [4, 3, 6, 5],  # gtbox[0]とは接点無し。
        ])
        expected = np.array([
            [1, 1 / 7],
            [1 / 3, 0],
            [1 / 7, 1],
            [0, 0],
            [0, 0],
        ])
        iou = K.get_value(bbox.get_iou(anchor, gtbox))
        np.testing.assert_almost_equal(iou, expected, decimal=5)
