import unittest
import math

import numpy as np
import xrcnn.util.anchor as anchor

from xrcnn import config


class TestAnchor(unittest.TestCase):
    def test_generate_anchors(self):
        conf = config.Config()
        a = anchor.Anchor(conf)
        anchors = a._generate_anchors((2, 2))
        centers = (anchors[:, 2:] - anchors[:, :2]) / 2 + anchors[:, :2]
        expected_centers = np.array(
            [[8, 8], [8, 24], [24, 8], [24, 24]]).repeat(9, axis=0)
        np.testing.assert_almost_equal(centers, expected_centers, decimal=5)

    def test_generate_gt_offsets(self):
        conf = config.Config()
        conf.anchor_box_aspect_ratios = [
            (1. / math.sqrt(2), 2. / math.sqrt(2)),
            (1., 1.),
            (2. / math.sqrt(2), 1. / math.sqrt(2))]
        conf.anchor_box_scales = [4, 8, 16]
        conf.backbone_shape = [64, 64]
        conf.stride_per_base_nn_feature = 2
        anc = anchor.Anchor(conf)

        bbox = np.array([[1,  1,  5,  5], [1,  3,  9,  10],
                         [3,  6,  12,  12], [9, 9, 13, 13]])
        bbox2 = np.array([[2,  2,  6,  6], [1,  3,  9,  10],
                          [3,  6,  12,  12], [9, 9, 13, 13]])
        offset, clazz = anc.generate_gt_offsets(
            bbox,  (16, 16), n_max_sample=128)
        offset2, clazz2 = anc.generate_gt_offsets(
            bbox2, (16, 16), n_max_sample=128)
        self.assertEqual(len(np.where(clazz >= 1)[0]), 5)
        self.assertEqual(len(np.where(clazz2 >= 1)[0]), 4)
