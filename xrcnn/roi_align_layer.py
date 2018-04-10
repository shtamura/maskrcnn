from keras import layers as KL
from keras import backend as K
import tensorflow as tf
import logging

from xrcnn.util import log

logger = logging.getLogger(__name__)


class RoiAlignLayer(KL.Layer):
    """RoI Alignを行う。
    フィーチャマップにRoIを適用し、固定サイズのフィーチャマップに変換する。

    Args:
        config: config

    Inputs:
        features: backboneの出力フィーチャマップ。
            （VGG16の畳み込み層(5回目のプーリングの1つ前まで)をの出力）
            入力画像サイズが1024であれば(N, 64, 64, 512)のはず。
        rois: RoI
            (N, n_rois, 4)
            2軸目は0〜1に正規化された座標
            (y1,x1,y2,x2)

    Outputs:
        (N, n_rois, config.roi_align_out_size,
            config.roi_align_out_size, channels)
    """

    def __init__(self, out_shape, config, **kwargs):
        super(RoiAlignLayer, self).__init__(**kwargs)
        self.out_shape = out_shape
        self.batch_size = config.batch_size

    def call(self, inputs):
        features = inputs[0]
        rois = inputs[1]
        n_roi_boxes = K.shape(rois)[1]

        # roisには[0,0,0,0]のRoIも含むが、バッチ毎の要素数を合わせるため、そのまま処理する。

        # crop_and_resizeの準備
        # roisを0軸目を除き（バッチを示す次元を除き）、フラットにする。
        roi_unstack = K.concatenate(tf.unstack(rois), axis=0)
        # roi_unstackの各roiに対応するバッチを指すindex
        batch_pos = K.flatten(
            K.repeat(K.reshape(K.arange(self.batch_size), [-1, 1]),
                     n_roi_boxes))
        # RoiAlignの代わりにcrop_and_resizeを利用。
        # crop_and_resize内部でbilinear interporlationしてようなので、アルゴリズム的には同じっぽい
        crop_boxes = tf.image.crop_and_resize(features,
                                              roi_unstack, batch_pos,
                                              self.out_shape)

        # (N * n_rois, out_size, out_size, channels)
        # から
        # (N, n_rois, out_size, out_size, channels)
        # へ変換
        crop_boxes = K.reshape(crop_boxes,
                               [self.batch_size, n_roi_boxes]
                               + self.out_shape + [-1])
        log.tfprint(crop_boxes, "crop_boxes: ")
        return crop_boxes

    def compute_output_shape(self, input_shape):
        return (input_shape[1][0], input_shape[1][1],
                self.out_shape[0], self.out_shape[1], input_shape[0][-1])
