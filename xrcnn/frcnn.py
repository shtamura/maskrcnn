from logging import getLogger
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.layers import Input, TimeDistributed, Lambda, Activation, Dense, \
    Flatten, Reshape, Layer
from keras.layers.convolutional import Conv2D
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
from keras.utils import multi_gpu_model

import tensorflow as tf
from xrcnn.batchnorm import BatchNorm
import xrcnn.loss as loss
from xrcnn.util import bbox
from xrcnn.util import log
from xrcnn.region_proposal_layer import RegionProposalLayer
from xrcnn.roi_align_layer import RoiAlignLayer

logger = getLogger(__name__)


class Frcnn:
    def __init__(self, anchors, config):
        self.anchors = anchors
        self.config = config

    def _model_backbone_plane(self):
        if self.config.backbone_nn_type == 'vgg':
            model = VGG16(weights='imagenet')
        else:
            model = ResNet50(weights='imagenet')
        return model

    def _model_backbone_headless(self):
        if self.config.backbone_nn_type == 'vgg':
            model = VGG16(weights='imagenet', include_top=False)
            # 畳み込み層の後のプーリング層を除く
            # https://github.com/keras-team/keras/issues/2371
            # https://github.com/keras-team/keras/issues/6229
            # http://forums.fast.ai/t/how-to-finetune-with-new-keras-api/2328/9
            model.layers.pop()
        else:
            model = ResNet50(weights='imagenet', include_top=False)
        # VGGの重みは学習対象外
        for layer in model.layers:
            layer.trainable = False
        output = model.layers[-1].output
        _input = model.input
        return _input, output

    def _nn_rpn(self, backbone, trainable):
        """Region Proporsal Network
        領域提案とオブジェクト推測が得られるNN。

        Args:
            backbone : 起点となるNNのレイヤ
            config: Config

        Returns:
            [rois, offsets, objects]
            rois: 領域提案
                形状は以下の通り。
                    (N, n_anchor, 4)
                    3軸目は領域の左上と右下の座標。
                        (y1, x1, y2, x2)
            offsets: 領域提案とアンカーのオフセット
                形状は以下の通り。
                    (N, n_anchor, 4)
                    3軸目は領域提案とアンカーのオフセット（中心、幅、高さ）。
                        (tx, ty, th, tw)
                        つまりアンカーがn個とすると、(tx0,ty0,th0,tw0,tx1,ty1, ... ,thn, twn)
                        それぞれの値は論文に記載の通り以下とする。
                            tx =(x−xa)/wa, ty =(y−ya)/ha,
                            tw = log(w/wa), th = log(h/ha)
                        ※それぞれ、アンカーからのオフセット
                        ※「x」は予測された領域の中心x、「xa」はアンカーの中心x。
            objects: オブエジェクト、非オブジェクトである確率
                形状は以下の通り。
                    (N, n_anchor, 2)
                    [:, :, 0]をオブジェクトではない確率、[:, :, 1]をオブジェクトである確率とみなす。

        """
        # 中間層（ZFの場合は256-d、VGGの場合は512-d）
        shared = Conv2D(512, 3, padding='same', activation='relu',
                        kernel_initializer='he_uniform',
                        name='rpn_conv1', trainable=trainable)(backbone)
        # 領域座標提案
        offsets = Conv2D(self.config.n_anchor * 4, 1, padding='valid',
                         activation='linear',
                         kernel_initializer='he_uniform',
                         name='rpn_offsets', trainable=trainable)(shared)
        # (N, n_anchor, 4)の形状に変換
        offsets = Reshape([-1, 4], name='rpn_offsets_reshape',
                          trainable=trainable)(offsets)

        # オブジェクト判別
        obj = Conv2D(self.config.n_anchor * 2, 1, padding='valid',
                     activation='linear',
                     kernel_initializer='glorot_uniform',
                     name='rpn_objects_val', trainable=trainable)(shared)
        # (N, n_anchor, 2)の形状に変換
        obj_logit = Reshape([-1, 2], name='rpn_objects_reshape',
                            trainable=trainable)(obj)
        # オブジェクト／非オブジェクトを示す数値を確率に変換する
        obj_prob = Activation('softmax', name='rpn_objects_prob',
                              trainable=trainable)(obj_logit)

        # 領域提案
        # 座標が0~1に正規化されている
        normalized_rois = RegionProposalLayer(self.anchors, self.config,
                                              name='region_proporsal_layer',
                                              trainable=trainable)(
            [backbone, offsets, obj_prob])

        return normalized_rois, offsets, obj_prob, obj_logit

    def _nn_head(self, backbone, region_proposal):
        """Head Network
        region_proposal*クラスラベル毎のオフセット予測、
        オブジェクト毎の存在確率が得られるNN。

        Args:
            backbone : 起点となるNNのレイヤ
            region_proposal: RegionProposalLayerで得られた領域提案
                形状は以下の通り。
                    (N, n_rois, 4)
                    3軸目は領域の左上と右下の座標が0〜1に正規化されている。
                        (y1, x1, y2, x2)
        Returns:
            [offsets, labels]
            offsets: 領域提案からのオフセット
                形状は以下の通り。
                    (N, n_rois, n_label, 4)
                    3軸目は領域の中心、幅、高さが0〜1に正規化された値。
                        (tx, ty, th, tw)
            labels: クラスラベル毎の存在確率
                形状は以下の通り。
                    (N, n_rois, n_label)

        """
        #
        n_label = self.config.n_dataset_labels
        # RoI Align
        # 論文ではRoiPoolingだが、より精度の高いRoiAlignにする。
        out = RoiAlignLayer(self.config.roi_align_pool_shape, self.config,
                            name='head_roi_align')(
            [backbone, region_proposal])

        # FasterRCNN論文では4096だが、MaskRCNN論文では1024に削減している。
        # 4096だとGPU(tesra K80)でOutOfMemoryになるので2048に減らしてみた。
        # Resnetだど2048でもOutOfMemoryになるので1024にする。
        if self.config.backbone_nn_type == 'vgg':
            unit_size = 2048
        else:
            unit_size = 1024

        out = TimeDistributed(Flatten(name='head_flatten'))(out)
        out = TimeDistributed(Dense(unit_size, kernel_initializer='he_uniform',
                                    name='head_fc1'))(out)
        out = TimeDistributed(BatchNorm(axis=1), name='head_fc1_bn')(out)
        out = Activation('relu')(out)

        out = TimeDistributed(Dense(unit_size, kernel_initializer='he_uniform',
                                    name='head_fc2'))(out)
        out = TimeDistributed(BatchNorm(axis=1), name='head_fc2_bn')(out)
        out = Activation('relu')(out)

        # 畳込みに置き換え
        # out = TimeDistributed(Conv2D(1024, (self.config.roi_align_out_size,
        #                                     self.config.roi_align_out_size),
        #                              kernel_initializer='he_uniform',
        #                              padding="valid"),
        #                       name="head_conv1")(out)
        # out = TimeDistributed(BatchNorm(axis=3), name='head_conv1_bn')(out)
        # out = Activation('relu')(out)
        # out = TimeDistributed(Conv2D(1024, (1, 1)),
        #                       kernel_initializer='he_uniform',
        #                       name="head_conv2")(out)
        # out = TimeDistributed(BatchNorm(axis=3),
        #                       name='head_conv2_bn')(out)
        # out = Activation('relu')(out)
        # out = Lambda(lambda x: K.squeeze(K.squeeze(x, 3), 2),
        #              name="head_")(out)

        # region_proposal毎にそれぞれのクラスラベルの存在確率を得る。
        # 形状は[N, n_region, n_label]
        labels_logit = TimeDistributed(
            Dense(n_label,
                  kernel_initializer='glorot_uniform'),
            name='head_label_val')(out)
        labels_prob = Activation('softmax', name='head_label')(labels_logit)

        # region_proposal毎、クラスラベル毎にregion_proposalのBBoxからのオフセットを得る。
        offsets = TimeDistributed(Dense(4 * n_label, activation='linear',
                                        kernel_initializer='zero'),
                                  name='head_offsets')(out)
        # [N, n_rois, n_label, 4]に変形
        offsets = Reshape([-1, n_label, 4],
                          name='head_offsets_reshape')(offsets)

        return offsets, labels_prob, labels_logit

    def _build_model(self):
        rpn_trainable = self.config.training_mode in ['rpn_only', 'all']
        head_trainable = self.config.training_mode in ['head_only', 'all']

        # backbone network
        backbone_in, backbone_out = self._model_backbone_headless()

        # rpn
        normalized_rois, rpn_offsets, objects, objects_logit \
            = self._nn_rpn(backbone_out, rpn_trainable)

        # 学習時のみ損失を計算
        if self.config.training:
            # 学習時
            # 入力
            input_gt_rois = Input(
                shape=[None, 4], name="input_gt_rois", dtype='float32')
            input_gt_objects = Input(
                shape=[None], name="input_gt_objects", dtype='int32')
            inputs = [backbone_in, input_gt_rois, input_gt_objects]

            losses = []
            if rpn_trainable:
                # 損失計算
                # RPNの損失
                rpn_offsets_loss = Lambda(lambda x: loss.rpn_offsets_loss(*x),
                                          name="rpn_offsets_loss")(
                    [input_gt_rois, input_gt_objects, rpn_offsets])
                rpn_objects_loss = Lambda(lambda x: loss.rpn_objects_loss(*x),
                                          name="rpn_objects_loss")(
                    [input_gt_objects, objects])

                losses += [rpn_offsets_loss, rpn_objects_loss]

            if head_trainable:
                input_gt_boxes = Input(
                    shape=[None, 4], name="input_gt_boxes", dtype='float32')
                input_gt_label_ids = Input(
                    shape=[None], name="input_gt_label_ids", dtype='int32')
                inputs += [input_gt_boxes, input_gt_label_ids]

                # 正解データとRoIから評価対象のRoIを絞り込み、それに対応する正解データを得る。
                normalized_sample_rois, normalized_sample_gt_offsets, \
                    sample_gt_labels \
                    = SubsamplingRoiLayer(self.config,
                                          name='subsampling_roi_and_gt')(
                        [normalized_rois, input_gt_boxes, input_gt_label_ids])
                # 以下のようにoutput_shapeを直接指定するとIndexErrorが発生したので、
                # ↑のようにカスタムレイヤー化する
                # batch_size = K.shape(normalized_rois)[0]
                # sample_rois, sample_gt_offsets, sample_labels = \
                #     Lambda(lambda x: self._subsampling_roi_and_gt(*x),
                #            output_shape=[(batch_size, None, 4),
                #                          (batch_size, None, 4),
                #                          (batch_size, None)],
                #            name="subsampling_roi_and_gt")(
                #         [normalized_rois, input_gt_boxes,
                #         input_gt_label_ids])

                # head
                head_offsets, labels, labels_logit\
                    = self._nn_head(backbone_out, normalized_sample_rois)

                # 損失計算
                # ヘッドの損失はModel#compileで損失関数を指定する方法では対応出来ないため、
                # Layerとして定義してModel#add_lossで加算する。
                head_offsets_loss = Lambda(lambda x:
                                           loss.head_offsets_loss(*x),
                                           name="head_offsets_loss")(
                    [normalized_sample_gt_offsets, sample_gt_labels,
                        head_offsets])
                head_labels_loss = Lambda(lambda x:
                                          loss.head_labels_loss(*x),
                                          name="head_labels_loss")(
                    [sample_gt_labels, labels])

                # 損失
                losses += [head_offsets_loss, head_labels_loss]

            # 出力＝損失
            outputs = losses

        else:
            # 予測時
            # head
            # head_offsetsは0〜1で正規化された値
            head_offsets, labels, _ = self._nn_head(
                backbone_out, normalized_rois)

            # 予測時は損失不要
            # ダミーの損失関数
            dummy_loss = Lambda(lambda x: K.constant(0), name="dummy_loss")(
                [backbone_in])
            losses = [dummy_loss, dummy_loss, dummy_loss]
            inputs = [backbone_in]
            # normalized_roisの正規化を戻した座標にhead_offsetを適用することでBBoxを得る。
            outputs = [normalized_rois, head_offsets, labels,
                       rpn_offsets, objects]

        model = Model(inputs=inputs, outputs=outputs, name='faser_r_cnn')
        # Kerasは複数指定した損失の合計をモデル全体の損失として評価してくれる。
        # 損失を追加
        for output in losses:
            model.add_loss(tf.reduce_mean(output, keep_dims=True))
        return model, len(outputs)

    def compiled_model(self):
        if self.config.gpu_count > 1:
            # 複数GPUで並列処理
            with tf.device('/cpu:0'):
                model, n_outputs = self._build_model()
            model = multi_gpu_model(model, self.config.gpu_count)
        else:
            model, n_outputs = self._build_model()

        # compile()ではlossを指定しないが、空ではエラーになるためNoneのリストを指定する。
        model.compile(optimizer=Adam(lr=self.config.learning_rate),
                      loss=[None] * n_outputs)
        return model


class SubsamplingRoiLayer(Layer):
    def __init__(self, config, **kwargs):
        super(SubsamplingRoiLayer, self).__init__(**kwargs)
        self.config = config
        self.n_samples_per_batch = 64

    def call(self, inputs):
        return self._subsampling(*inputs)

    def compute_output_shape(self, input_shape):
        return [(self.config.batch_size, self.n_samples_per_batch, 4),
                (self.config.batch_size, self.n_samples_per_batch, 4),
                (self.config.batch_size, self.n_samples_per_batch)]

    def _subsampling(self, normalized_rois, gt_bboxes, gt_labels,
                     pos_iou_thresh=0.5,
                     exclusive_iou_tresh=0.1,
                     pos_ratio=0.25):
        """正解データとのIoUを基にRoIをサンプリングする。
        IoUがpos_iou_thresh以上であるRoIをオブジェクトとみなす。
            オブジェクトはサンプルの25%以内とする。（n_samples_per_batch * pos_ratio 以内）
        pos_iou_thresh未満、exclusive_iou_thresh以上は非オブジェクトとみなす。
        exclusive_iou_thresh未満は偶然の一致であり意味なし（難解）なので無視。
        ※論文ではheuristic for hard example mining.と記載されている点。
        バッチ毎のサンプル数はn_samples_per_batch以内とする。
        （n_samples_per_batch未満の場合は、n_samples_per_batchになるよう0パディングする。）

        上記のサンプリングに対応する正解データのラベル、また、BBoxとのオフセットも得る。

        Args:
            normalized_rois (tensor) : RegionProposalLayerで得られたRoI。
                (N, n_rois, 4)
                3軸目は領域の左上と右下の座標が0〜1に正規化された値。
                入力画像サイズの高さ、幅で除算することで正規化された値。
                    (y1, x1, y2, x2)
            gt_bboxes (ndarray) : 正解BBox。
                (N, config.n_max_gt_objects_per_image, 4)
                座標は正規化されていない。
            gt_labels (ndarray) : 正解ラベル。
                (N, config.n_max_gt_objects_per_image)
                ==0:背景データ
                >=1:オブジェクト
        Returns:
            sample_rois (tensor): サンプリングしたRoI。
                (N, n_samples_per_batch, 4)
                3軸目の座標は0〜1に正規化された値。
            sample_gt_offset (tensor): サンプリングしたRoIに対応するBBoxとのオフセット。
                (N, n_samples_per_batch, 4)
                3軸目の座標は0〜1に正規化された値をself.config.bbox_refinement_stdで割ることで標準化した値。
            sample_gt_labels (tensor): サンプリングしたRoIに対応するBBoxのラベル。
                (N, n_samples_per_batch)
        """
        pos_roi_per_batch = round(self.n_samples_per_batch * pos_ratio)

        # gt_bboxesをnormalized_roisに合わせて正規化する。
        # これでIoUが評価出来るようになる。
        input_h = self.config.image_shape[0]
        input_w = self.config.image_shape[1]
        normalized_gt_bboxes = bbox.normalize_bbox(gt_bboxes, input_h, input_w)

        # 入力をバッチ毎に分割
        normalized_rois = tf.split(normalized_rois, self.config.batch_size)
        normalized_gt_bboxes = tf.split(normalized_gt_bboxes,
                                        self.config.batch_size)
        gt_labels = tf.split(gt_labels, self.config.batch_size)

        sample_rois = []
        sample_gt_offsets = []
        sample_gt_labels = []

        for roi, gt_bbox, gt_label in zip(normalized_rois,
                                          normalized_gt_bboxes, gt_labels):
            # 0次元目(バッチサイズ)は不要なので削除
            roi = log.tfprint(roi, "roi: ")
            gt_bbox = log.tfprint(gt_bbox, "gt_bbox: ")
            gt_label = log.tfprint(gt_label, "gt_label: ")

            roi = K.squeeze(roi, 0)
            gt_bbox = K.squeeze(gt_bbox, 0)
            gt_label = K.squeeze(gt_label, 0)

            roi = log.tfprint(roi, "roi_squeezed: ")
            gt_bbox = log.tfprint(gt_bbox, "gt_bbox_squeezed: ")
            gt_label = log.tfprint(gt_label, "gt_label_squeezed: ")

            # ゼロパディング行を除外
            # K.gather(zero, K.squeeze(tf.where(K.any(zero, axis=1)), -1) )
            idx_roi_row = K.flatten(tf.where(K.any(roi, axis=1)))
            idx_gt_bbox = K.flatten(tf.where(K.any(gt_bbox, axis=1)))
            roi = K.gather(roi, idx_roi_row)
            # gt_bboxとgt_labelは行数と行の並びが同じなので同じidxを利用できる
            gt_bbox = K.gather(gt_bbox, idx_gt_bbox)
            gt_label = K.gather(gt_label, idx_gt_bbox)

            gt_bbox = log.tfprint(gt_bbox, "gt_bbox_gathered: ")
            gt_label = log.tfprint(gt_label, "gt_label_gathered: ")

            # IoUを求める。
            # (n_rois, )
            ious = bbox.get_iou_K(roi, gt_bbox)
            ious = log.tfprint(ious, "ious: ")

            # 各RoI毎にIoU最大のBBoxの位置を得る
            idx_max_gt = K.argmax(ious, axis=1)
            idx_max_gt = log.tfprint(idx_max_gt, "idx_max_gt: ")

            max_iou = K.max(ious, axis=1)  # max_iouの行数はroiと同じになる
            max_iou = log.tfprint(max_iou, "max_iou: ")
            idx_pos = K.flatten(tf.where(max_iou >= pos_iou_thresh))
            # positiveサンプル数をpos_roi_per_batch以内に制限
            limit_pos = K.minimum(pos_roi_per_batch, K.shape(idx_pos)[0])
            idx_pos = K.switch(K.shape(idx_pos)[0] > 0,
                               tf.random_shuffle(idx_pos)[:limit_pos],
                               idx_pos)
            limit_pos = log.tfprint(limit_pos, "limit_pos: ")
            idx_pos = log.tfprint(idx_pos,  "idx_pos: ")

            # negativeサンプル数を
            #   n_samples_per_batch - pos_roi_per_batch
            # に制限
            idx_neg = K.flatten(tf.where((max_iou < pos_iou_thresh)
                                         & (max_iou >= exclusive_iou_tresh)))
            # negativeサンプル数は pos_roi_per_batch - limit_pos(つまり残り) 以内に制限
            limit_neg = self.n_samples_per_batch - limit_pos
            limit_neg = K.minimum(limit_neg, K.shape(idx_neg)[0])
            idx_neg = K.switch(K.shape(idx_neg)[0] > 0,
                               tf.random_shuffle(idx_neg)[:limit_neg],
                               idx_neg)
            limit_neg = log.tfprint(limit_neg, "limit_neg: ")
            idx_neg = log.tfprint(idx_neg,  "idx_neg: ")

            # 返却するサンプルを抽出
            # GTのoffsets, labelsは各roisに対応させる。つまり、同じ位置に格納する。
            idx_keep = K.concatenate((idx_pos, idx_neg))
            idx_keep = log.tfprint(idx_keep, "idx_keep: ")

            # 各RoIの最大IoUを示すIndexについても、上記返却するサンプルのみを残す。
            idx_gt_keep = K.gather(idx_max_gt, idx_keep)
            # IoUが閾値以上のPositiveとみなされるサンプルのみを残すためのIndex。
            idx_gt_keep_pos = K.gather(idx_max_gt, idx_pos)
            idx_gt_keep = log.tfprint(idx_gt_keep, "idx_gt_keep: ")

            sample_roi = K.gather(roi, idx_keep)
            sample_gt_offset = bbox.get_offset_K(
                sample_roi, K.gather(gt_bbox, idx_gt_keep))
            # negativeな要素には0を設定
            sample_gt_label = K.concatenate((K.cast(K.gather(
                gt_label, idx_gt_keep_pos),
                dtype='int32'),
                K.zeros([limit_neg],  # K.zerosは0階テンソルを受け付けないので配列化。。。
                        dtype='int32')))

            # 行数がn_samples_per_batch未満の場合は0パディング
            remain = tf.maximum(self.n_samples_per_batch
                                - tf.shape(sample_roi)[0], 0)
            sample_roi = tf.pad(sample_roi, [(0, remain), (0, 0)],
                                name='subsample_sample_roi')
            sample_gt_offset = tf.pad(sample_gt_offset, [(0, remain), (0, 0)],
                                      name='subsample_sample_gt_offset')
            sample_gt_offset /= self.config.bbox_refinement_std
            sample_gt_label = tf.pad(sample_gt_label, [(0, remain)],
                                     name='subsample_sample_gt_label')

            sample_roi = log.tfprint(sample_roi, "sample_roi: ")
            sample_gt_offset = log.tfprint(
                sample_gt_offset, "sample_gt_offset: ")
            sample_gt_label = log.tfprint(sample_gt_label, "sample_gt_label: ")

            sample_rois.append(sample_roi)
            sample_gt_offsets.append(sample_gt_offset)
            sample_gt_labels.append(sample_gt_label)

        return [K.stack(sample_rois), K.stack(sample_gt_offsets),
                K.stack(sample_gt_labels)]
