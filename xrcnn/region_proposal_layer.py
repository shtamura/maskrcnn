from keras import backend as K
from keras import layers as KL
import tensorflow as tf
import xrcnn.util.bbox as bbox
import logging

logger = logging.getLogger(__name__)


class RegionProposalLayer(KL.Layer):
    """RPNの最終レイヤ。
    上位のレイヤから得るオフセット(rpn_offsets), オブジェクトである確率(rpn_objects)から
    バッチ毎にRoIとRoI特徴マップの対応（roi_indices）、さらにアンカーを求める。

    rpn_objectsからスコア（rpn_objects）上位config.n_train_pre_nms(config.n_test_pre_nms)
    を残し、non_maximum_suppressionでIoUがconfig.nms_thresh以上の重複領域を除く。
    （スコアが高い物を優先して残す）
    NMSの結果残った領域から、更にスコアが上位config.n_train_post_nms(config.n_test_post_nms)である領域に絞り込む。

    Inputs:
        feature_map: [N, c, h, w]
        rpn_offsets: [N, R, 4]
            3軸目の形状は以下。
            (dy, dx, dh, dw)
        rpn_objects: [N, R, 1]
            3軸目はオブジェクトである確率。

        要素数N: config.batch_size
        要素数R: config.n_train_post_nms(トレーニング時はconfig.n_test_post_nms)

    Returns:
        領域提案: [N, n_rois, (y1, x1, y2, x2)]
            3軸目の座標は0〜1に正規化されている。
    """

    def __init__(self,
                 anchors,
                 config,
                 **kwargs):
        super(RegionProposalLayer, self).__init__(**kwargs)
        self.input_h = config.image_shape[0]
        self.input_w = config.image_shape[1]
        self.anchors = anchors
        self.nms_thresh = config.nms_thresh
        self.training = config.training
        self.bbox_refinement_std = config.bbox_refinement_std
        if self.training:
            self.n_pre_nms = config.n_train_pre_nms
            self.n_post_nms = config.n_train_post_nms
        else:
            self.n_pre_nms = config.n_test_pre_nms
            self.n_post_nms = config.n_test_post_nms
        self.batch_size = config.batch_size

    def call(self, inputs):
        rpn_offsets = inputs[1]
        # 既存実装に合わせた精度向上
        rpn_offsets *= self.bbox_refinement_std
        rpn_objects = inputs[2]
        fg_scores = rpn_objects[:, :, 1]
        n_anchors = self.anchors.shape[0]

        # スコアが上位の候補のみに絞る
        # r = np.repeat(range(3), 2)
        # >>> i = K.get_value(I)
        # array([[2, 1],
        #        [2, 1],
        #        [0, 1]], dtype=int32)
        #  >>> np.stack((r, i.flatten()), axis=1)
        # array([[0, 2],
        #        [0, 1],
        #        [1, 2],
        #        [1, 1],
        #        [2, 0],
        #        [2, 1]])
        # AI = K.variable(ai)
        # K.eval(tf.reshape(tf.gather_nd(T, AI), (3,2)))
        pre_nms_limit = min(self.n_pre_nms, n_anchors)
        # バッチ毎に上位Nのスコアが存在するIndexを取得する。
        top_k_idx = tf.nn.top_k(fg_scores, pre_nms_limit, sorted=True).indices

        # idxの形状は(R, pre_nms_limit)
        # tf.gather_ndで利用できるよう、バッチ入力のIndexとスコアのIndexの組合せにする。
        #   [[バッチ入力のIndex, スコアのindex], ・・・]
        n_batch = self.batch_size
        # ↑はもともと「K.shape(feature_map)[0]」としたかったが
        # dynamic shapeなのでtensorflowのslice, join系の関数に指定出来ない。。。
        # よって、configから固定値を取得することにした。
        # MaskRCNNの実装も同様だったのでこれでよいはず。

        rn = K.flatten(
            K.repeat(K.reshape(K.arange(n_batch), [-1, 1]),
                     pre_nms_limit))
        pos = K.stack((rn, K.flatten(top_k_idx)), axis=1)
        # スコア上位のindexを元にスコア、roi, アンカーを抽出する。
        fg_scores = K.reshape(tf.gather_nd(fg_scores, pos),
                              [n_batch, pre_nms_limit])
        rpn_offsets = K.reshape(tf.gather_nd(rpn_offsets, pos),
                                [n_batch, pre_nms_limit, 4])
        # バッチ毎に維持するアンカー(pos)が異なるので、バッチ数分アンカーを積み上げる。
        anchors = K.reshape(K.tile(self.anchors, [n_batch, 1]),
                            [n_batch, n_anchors, 4])
        anchors = K.reshape(tf.gather_nd(anchors, pos),
                            [n_batch, pre_nms_limit, 4])

        # アンカーとオフセットからBBoxを得る。
        # バッチ毎にアンカー、オフセットを積み上げ、bbox.get_bboxでまとめて計算する。
        # (N*R, 4)に変形。
        stacked_anchors_per_batch = K.reshape(anchors,
                                              [n_batch * pre_nms_limit, 4])
        stacked_offsets_per_batch = K.reshape(rpn_offsets,
                                              [n_batch * pre_nms_limit, 4])
        bboxes = bbox.get_bbox(stacked_anchors_per_batch,
                               stacked_offsets_per_batch)

        # 画像をはみ出すBBoxは画像領域内に収まるよう座標を調整する。
        bboxes = K.clip(bboxes, [0, 0, 0, 0],
                        [self.input_h, self.input_w,
                         self.input_h, self.input_w])

        # 元の形状(N, R, 4)に戻す
        bboxes = K.reshape(
            bboxes, [K.shape(rpn_offsets)[0], K.shape(rpn_offsets)[1], 4])

        # 小さなオブジェクトの検出も可能としたいため、小さなBBoxは残す。
        # fasterRCNN論文基準のchainercvの実装では、小さなBBOXを削除している。
        # 後発の https://github.com/matterport/Mask_RCNN では残している。

        # NMSでself.n_post_nms以下になるよう絞る。
        # バッチ毎にNMSを呼び出す形にするため、tf.splitを使ってbboxesを[n_batch,R,4]から[R,4]のリストにしてNMSする。
        split_size = tf.tile([1], [n_batch])
        proposal_boxes = K.stack([self._nms(box, score, pre_nms_limit)
                                  for box, score
                                  in zip(tf.split(bboxes, split_size, axis=0),
                                         tf.split(
                                             fg_scores, split_size, axis=0)
                                         )])

        return proposal_boxes

    def _nms(self, bboxes, scores, dim1):
        """nmsの結果を得る。
        結果として得られる座標は0〜1に正規化されたままとする。
        """
        # tf.splitの結果の形状が(1,R,X)のままなので、(R,X)に変換
        bboxes = tf.reshape(bboxes, [dim1, 4])
        scores = tf.reshape(scores, [dim1])
        # tensorflowのnon-max-suppression(NMS)を利用するので、まず入力するboxの座標を0~1に正規化する。
        normalized_boxes = bbox.normalize_bbox(bboxes,
                                               self.input_h, self.input_w)
        indices = tf.image.non_max_suppression(
            normalized_boxes, scores, self.n_post_nms,
            iou_threshold=self.nms_thresh)
        # NMSしたboxに絞り込む
        # 座標は正規化しまま。
        boxes = tf.gather(normalized_boxes, indices)
        # boxが上限self.n_post_nmsを下回る場合は、1軸目の要素数がself.n_post_nmsとなるよう
        # 0で埋めることで、バッチ毎の出力形状を合わせる。（1つのテンソルにまとめられるようにする）
        # dataset.py#data_generatorではself.n_post_nmsの次元数を前提とする。
        padding = tf.maximum(self.n_post_nms - tf.shape(boxes)[0], 0)
        boxes = tf.pad(boxes, [(0, padding), (0, 0)])
        return boxes

    def compute_output_shape(self, input_shape):
        return (self.batch_size, self.n_post_nms, 4)
