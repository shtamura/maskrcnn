import numpy as np
from xrcnn.util import bbox as B


class Anchor:
    def __init__(self, config):
        # def __init__(self, base_size=16,
        #              anchor_ratios=[
        #                  (1. / math.sqrt(2), 2. / math.sqrt(2)),
        #                  (1., 1.),
        #                  (2. / math.sqrt(2), 1. / math.sqrt(2))],
        #              anchor_scales=[128 / 4, 256 / 4, 512 / 4],
        #              backbone_shape=[64 / 4, 64 / 4]):
        """RoI予測の基準となるアンカーを生成する。
        アンカーの基準となる値を指定する。

        Args:
            base_size (number): アンカーを適用する特徴マップ1ピクセルが、入力画像において何ピクセルに値するか。
            anchor_ratios (list of float): アンカーのアスペクト比。
                :math:`[(h, w), ...]`
            anchor_scales (list of numbers): アンカーのサイズ（入力画像におけるサイズ）。
                このサイズの正方形をアンカーの領域とする。
            anchor_ratios (list of numbers): アンカーのアスペクト比
        """
        self.base_size = config.stride_per_base_nn_feature
        self.backbone_shape = config.backbone_shape
        self.anchor_ratios = config.anchor_box_aspect_ratios
        self.anchor_scales = config.anchor_box_scales
        self.bbox_refinement_std = config.bbox_refinement_std
        self.anchor_base = self._anchor_base(
            self.base_size, self.anchor_ratios, self.anchor_scales)
        self.anchors = self._generate_anchors(self.backbone_shape)

    def generate_gt_offsets(self, bbox_gt, img_size,
                            pos_iou_thresh=0.5,
                            neg_iou_thresh=0.3,
                            n_max_sample=256,
                            pos_ratio=0.5):
        """anchorにGroud truthなBBoxを適用し、anchor毎に最もIoUが大きいBBoxを特定し、そのBBoxとのオフセットを得る。
        IoU値により、各アンカーを以下に分類する。
            0.7以上：オブジェクト
                →0.5にする。
                    0.7だとVOCdevkit/VOC2007/Annotations/007325.xmlにあるようなサイズのBboxが
                    GTとして得られなかったため。
            0.3未満：非オブジェクト
            それ以外：評価対象外。つまり、トレーニングには使わないアンカー。

        Args:
            bbox_gt (array): Ground truthなBBox
                Its shape is :math:`(R, 4)`.
            img_size (h，w): 入力画像の高さと幅のタプル.
            pos_iou_thresh: この値以上のIoUをclass=1とする。
            pos_iou_thresh: この値未満のIoUをclass=0とする。
            n_max_sample: 評価対象とする（classが1or0である）オフセットの上限
            pos_ratio: 評価対象サンプル中のPositiveの割合
                n_max_sample, pos_ratioは論文中の以下への対応。
                考慮無しではNegativeサンプルが支配的になる。学習効率も考慮し、このような処理を行うものと思われる。
                Each mini-batch arises from a single image that contains many
                positive and negative example anchors. It is possible to
                optimize for the loss functions of all anchors,
                but this will bias towards negative samples as they are
                dominate. Instead, we randomly sample 256 anchors in an image
                to compute the loss function of a mini-batch, where the sampled
                 positive and negative anchors have a ratio of up to 1:1.
                 If there are fewer than 128 positive samples in an image,
                 we pad the mini-batch with negative ones.

        Returns:
            (offsets, obj_flags):

            offsets (array) : 各アンカーとGround TruthなBBoxとのオフセット。
                Its shape is :math:`(S, 4)`.
                2軸目の内容は以下の通り。
                (x, y ,h, w)
            objects (array): 各アンカーがオブジェクトか否か。
                Its shape is :math:`(S, 1)`.
                2軸目の内容は以下の通り。
                    1：オブジェクト
                    0：非オブジェクト
                    −1：評価対象外
        """

        h, w = img_size
        anchor = self.anchors
        n_anchor_initial = len(anchor)

        # 入力領域をはみ出すアンカーを除外
        index_inside = np.where(
            (anchor[:, 0] >= 0) &
            (anchor[:, 1] >= 0) &
            (anchor[:, 2] <= h) &
            (anchor[:, 3] <= w)
        )[0]
        anchor = anchor[index_inside]

        # 各アンカー毎にGTとのIoUを算出し、最大か0.7以上のIoUを残す。
        # IoU >= 0.7はオブジェクト候補とする（class = 1）
        # IoU < 0.3は非オブジェクト候補とする（class = 0）
        # それ以外のIoUは評価対象外とする（class = -1）
        argmax_ious, objects = self._create_label(anchor, bbox_gt,
                                                  pos_iou_thresh,
                                                  neg_iou_thresh,
                                                  n_max_sample,
                                                  pos_ratio)
        # アンカーとGroud truthのオフセットを得る。
        offsets = B.get_offset(anchor, bbox_gt[argmax_ious])
        # 既存実装に合わせた精度向上
        offsets /= np.array(self.bbox_refinement_std)

        # 元の形状に戻す。
        # index_insideに削減した1次元目の次元数をn_anchor_initialに戻す。
        # 復元した座標は評価対象外なので、ラベルは−1、オフセットは0を設定して無効な状態に。
        objects = self._unmap(objects, n_anchor_initial, index_inside, fill=-1)
        offsets = self._unmap(offsets, n_anchor_initial, index_inside, fill=0)

        return offsets, objects

    def _create_label(self, anchor, bbox, pos_iou_thresh, neg_iou_thresh,
                      n_max_sample, pos_ratio):
        """
        anchorとbboxのIoUを算出し、それぞれオブジェクト候補か否かを得る。
        IoU >= 0.7はオブジェクト候補とする（class = 1）
        IoU < 0.3は非オブジェクト候補とする（class = 0）
        それ以外のIoUは評価対象外とする（class = -1）

        anchor毎に全bboxについてのIoUを算出する。
        つまり、(len(anchor), len(bbox))のマトリクスになる。
        このマトリクスから、anchor毎に最大のIoUを含むbboxのindexを得る。

        Args:
            anchor (tensor): アンカー
                Its shape is :math:`(R, 4)`.
            bbox (tensor): Ground truthなBBox
                Its shape is :math:`(S, 4)`.
            pos_iou_thresh: この値以上のIoUをclass=1とする。
            pos_iou_thresh: この値未満のIoUをclass=0とする。
            n_max_sample: 評価対象とする（classが1or0である）オフセットの上限
            pos_ratio: 評価対象サンプル中のPositiveの割合

        Returns:
            (index_max_iou_per_anchor, label)
            index_max_iou_per_anchor: anchor毎のIoUが最大となるbboxのIndex。
                Its shape is :math:`(R, 1)`.
            label:anchor毎のオブジェクト／非オブジェクト
                Its shape is :math:`(R, 1)`.

        """
        # 評価対象外の−1で初期化
        label = np.full((len(anchor)), -1)

        # アンカー毎にIoUが最大となるbboxの列Indexとその値、最大のIoUを含むアンカーのIndexを得る。
        index_max_iou_per_anchor, max_ious, gt_argmax_ious = self._calc_ious(
            anchor, bbox)

        # 最大のIoUを含むアンカーはPositive
        label[gt_argmax_ious] = 1

        # 閾値以上のIoUはPositive
        label[max_ious >= pos_iou_thresh] = 1

        # 閾値未満のIoUはNegative
        label[max_ious < neg_iou_thresh] = 0

        # Positiveのサンプル数を上限以内に抑える
        n_pos_max = int(pos_ratio * n_max_sample)
        pos_index = np.where(label == 1)[0]
        if len(pos_index) > n_pos_max:
            # n_pos_maxを超える場合は、Positiveをランダムに評価対象外にする
            disable_index = np.random.choice(
                pos_index, size=(len(pos_index) - n_pos_max), replace=False)
            label[disable_index] = -1

        # Negativeサンプルも同様に上限以内に抑える
        n_neg = n_max_sample - np.sum(label == 1)
        neg_index = np.where(label == 0)[0]
        if len(neg_index) > n_neg:
            disable_index = np.random.choice(
                neg_index, size=(len(neg_index) - n_neg), replace=False)
            label[disable_index] = -1

        return index_max_iou_per_anchor, label

    def _calc_ious(self, anchor, bbox):
        # anchor毎に全bboxとのIoUを得る。
        ious = B.get_iou(anchor, bbox)
        # anchor毎に最大のIoUが格納されている列Indexを得る。
        argmax_ious = ious.argmax(axis=1)
        # argmax_iousが示すIndexの実数、つまりアンカー毎の最大のIoUを得る。
        max_ious = ious[np.arange(ious.shape[0]), argmax_ious]

        # IoUが最大となるアンカーのIndexを特定する
        # 以下はchainercvに於ける実装だが、これだと全てのBBoxとのIoUが0の
        # アンカーについてもgt_argmax_iousに含まれそう。。。つまり全てPositive扱いになる。
        # 論文に従い、最大IoUのアンカーのみを特定する。
        # gt_argmax_ious = ious.argmax(axis=0)
        # gt_max_ious = ious[gt_argmax_ious, np.arange(ious.shape[1])]
        # gt_argmax_ious = np.where(ious == gt_max_ious)[0]
        gt_argmax_ious = np.where(ious == ious.max())[0]

        return argmax_ious, max_ious, gt_argmax_ious

    def _unmap(self, data, count, index, fill=0):
        # 元の形状に戻す。

        if len(data.shape) == 1:
            ret = np.empty((count,), dtype=data.dtype)
            ret.fill(fill)
            ret[index] = data
        else:
            ret = np.empty((count,) + data.shape[1:], dtype=data.dtype)
            ret.fill(fill)
            ret[index, :] = data
        return ret

    def _generate_anchors(self, feature_shape):
        """特徴マップの各ピクセル毎のアンカーを生成する。

        Args:
            feature_shape: 特徴マップの高さと幅のタプル
                (h, w)

        Returns:
            ndarray
            形状は以下の通り。
            (len(feature_height) * len(feature_width)
                * len(self.anchor_ratios) * len(self.anchor_scales), 4)
            1軸目は「特徴マップの行」→「特徴マップの列」→「アスペクト比の順」→「アンカーサイズ」で並ぶ。
            例：
            2軸目に格納される座標の形状は以下の通り。
            : math: `(y_{min}, x_{min}, y_{max}, x_{max})`

        """

        feature_height, feature_width = feature_shape
        # フィーチャマップの全ピクセルを示す交点座標
        shift_y = np.arange(0, feature_height * self.base_size, self.base_size)
        shift_x = np.arange(0, feature_width * self.base_size, self.base_size)
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        # 交点毎にself._anchor_baseを加算することで交点毎のアンカーを算出したい。
        # 各交点のアンカーのベースとなる座標を求める
        shift = np.stack((shift_y.flatten(), shift_x.flatten(),
                          shift_y.flatten(), shift_x.flatten()), axis=1)
        # np.arange(0, 5, 1)で以下のようになる。
        # >>> shift_y
        # array([[0, 0, 0, 0, 0],
        #        [1, 1, 1, 1, 1],
        #        [2, 2, 2, 2, 2],
        #        [3, 3, 3, 3, 3],
        #        [4, 4, 4, 4, 4]])
        # >>> shift_x
        # array([[0, 1, 2, 3, 4],
        #        [0, 1, 2, 3, 4],
        #        [0, 1, 2, 3, 4],
        #        [0, 1, 2, 3, 4],
        #        [0, 1, 2, 3, 4]])
        # >>> shift
        # array([[0, 0, 0, 0],
        #        [0, 1, 0, 1],
        #        [0, 2, 0, 2],
        #        [0, 3, 0, 3],
        #        [0, 4, 0, 4],
        #        [1, 0, 1, 0],
        #        [1, 1, 1, 1],
        #        [1, 2, 1, 2],
        #        [1, 3, 1, 3],
        #        [1, 4, 1, 4],
        #        [2, 0, 2, 0],
        #        [2, 1, 2, 1],
        #        [2, 2, 2, 2],
        #        [2, 3, 2, 3],
        #        [2, 4, 2, 4],
        #        [3, 0, 3, 0],
        #        [3, 1, 3, 1],
        #        [3, 2, 3, 2],
        #        [3, 3, 3, 3],
        #        [3, 4, 3, 4],
        #        [4, 0, 4, 0],
        #        [4, 1, 4, 1],
        #        [4, 2, 4, 2],
        #        [4, 3, 4, 3],
        #        [4, 4, 4, 4]])

        n_a = self.anchor_base.shape[0]
        n_s = shift.shape[0]
        # 各交点毎にアンカーの座標を求める。
        # まずはそのために次元を調整。
        # (len(feature_height) * len(feature_width), 1, 4)にする。
        # 上記5*5の例であれば、(25,1,4)
        shift = np.transpose(np.reshape(shift, (1, n_s, 4)), (1, 0, 2))
        # (1, len(self.anchor_ratios) * len(self.anchor_scales), 4)にする。
        # 上記5*5の例であれば、(1,9,4)
        anchor = np.reshape(self.anchor_base, (1, n_a, 4))
        # shift + anchorにより、shift[n, :, :]とanchor[:, k, :]の組合せが得られる。
        # つまり、各交点毎にanchor_baseを加算した結果が得られる。
        # 結果として得られるテンソルの形状は以下の通り。
        # (len(feature_height) * len(feature_width),
        #   len(self.anchor_ratios) * len(self.anchor_scales), 4)
        # 上記5*5の例であれば、(25,9,4)
        anchor = shift.astype(float) + anchor

        # 上記を以下の形状に変換する。
        # (len(feature_height) * len(feature_width)
        #   * len(self.anchor_ratios) * len(self.anchor_scales), 4)
        anchor = np.reshape(anchor, (n_s * n_a, 4))
        return anchor.astype('float32')

    def _anchor_base(self, base_size, anchor_ratios, anchor_scales):
        """基準となるアンカーを生成する。
        ratiosとanchor_scales毎にアンカーを示す座標（矩形の左上と右下の座標）を返す。
        矩形の中心は(base_size / 2, base_size / 2)とする。（論文に合わせ、受容野の中心とする）

        Args:
            base_size(number): アンカーを適用する特徴マップ1ピクセルが、入力画像において何ピクセルに値するか。
            anchor_ratios(list of float): アンカーのアスペクト比。
                : math: `[(h, w), ...]`
            anchor_scales(list of numbers): アンカーのサイズ（入力画像におけるサイズ）。
                このサイズの正方形をアンカーの領域とする。

        Returns:
            numpy配列
            形状は以下の通り。
            (len(anchor_ratios) * len(anchor_scales), 4)
            2軸目に格納される座標の形状は以下の通り。
            : math: `(y_{min}, x_{min}, y_{max}, x_{max})`

        """
        # 受容野の中心を指定
        py = base_size / 2.
        px = base_size / 2.

        anchor_base = np.zeros((len(anchor_ratios) * len(anchor_scales), 4),
                               dtype=np.float32)
        for i in range(len(anchor_ratios)):
            for j in range(len(anchor_scales)):
                h = anchor_scales[j] * anchor_ratios[i][0]
                w = anchor_scales[j] * anchor_ratios[i][1]

                index = i * len(anchor_scales) + j
                # 矩形右上の座標
                anchor_base[index, 0] = py - h / 2.
                anchor_base[index, 1] = px - w / 2.
                # 矩形左上の座標
                anchor_base[index, 2] = py + h / 2.
                anchor_base[index, 3] = px + w / 2.
        return anchor_base.astype('float32')
