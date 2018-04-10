from logging import getLogger
from keras.layers import Input, TimeDistributed, Lambda, Activation, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import Model
from keras import backend as K

import tensorflow as tf
from xrcnn.batchnorm import BatchNorm
import xrcnn.loss as loss
from xrcnn.frcnn import Frcnn
from xrcnn.frcnn import SubsamplingRoiLayer
from xrcnn.util import bbox
from xrcnn.roi_align_layer import RoiAlignLayer

logger = getLogger(__name__)


class MaskRCNN(Frcnn):
    def _nn_squeeze_roi(self, batch_rois, batch_offsets, batch_labels):
        input_batch_size = batch_rois.shape[0]
        # batch_rois = log.tfprint(batch_rois, "batch_rois:before_split:debug")
        batch_rois = tf.split(batch_rois, input_batch_size)
        batch_offsets = tf.split(batch_offsets, input_batch_size)
        batch_labels = tf.split(batch_labels, input_batch_size)

        ret_bboxes, ret_normalized_rois, ret_labels, ret_scores = \
            [], [], [], []

        for normalized_rois, head_offsets, labels \
                in zip(batch_rois, batch_offsets, batch_labels):
            # バッチを示す0次元目を削除
            normalized_rois, head_offsets, labels = \
                K.squeeze(normalized_rois, axis=0), \
                K.squeeze(head_offsets, axis=0), \
                K.squeeze(labels, axis=0)

            # 各RoI毎に最も確率の高いラベルに対応するOffsets, masksを抽出
            # [n, n_labels] -> [n, 1]
            labels_id = K.cast(K.argmax(labels, axis=-1), tf.int32)
            # labels_id = log.tfprint(labels_id, "labels_id:inloop:debug")
            idx_labels = K.cast(K.stack([K.arange(K.shape(labels)[0]),
                                         labels_id],
                                        axis=1), tf.int32)
            # idx_labels = log.tfprint(idx_labels, "idx_labels:inloop:debug")
            # head_offsets = log.tfprint(
            #     head_offsets, "head_offsets:pre:inloop:debug")
            # [n, n_labels, 4] -> [n, 1, 4]
            head_offsets = tf.gather_nd(head_offsets, idx_labels)
            # head_offsets = log.tfprint(
            #     head_offsets, "head_offsets:gather:inloop:debug")
            head_offsets *= K.variable(self.config.bbox_refinement_std)
            # head_offsets = log.tfprint(
            #     head_offsets, "head_offsets:std:inloop:debug")
            # [n, n_labels] -> [n, 1]
            labels_prob = tf.gather_nd(labels, idx_labels)

            # オブジェクトである確率が閾値以上の領域のみを残す
            # 1次元目の列番号のみ抽出
            idx_labels, _ = tf.unique(tf.where(
                labels_prob >= self.config.detect_label_prob)[:, 0])
            # idx_labels = log.tfprint(idx_labels, "idx_labels:inloop:debug")
            normalized_rois = tf.gather(normalized_rois, idx_labels)
            head_offsets = tf.gather(head_offsets, idx_labels)
            labels_prob = tf.gather(labels_prob, idx_labels)
            labels_id = tf.gather(labels_id, idx_labels)
            # labels_prob = log.tfprint(labels_prob,
            #   "labels_prob:inloop:debug")

            # 背景は除く
            idx_labels, _ = tf.unique(tf.where(labels_id > 0)[:, 0])
            # idx_labels = log.tfprint(idx_labels, "idx_labels:inloop:debug")
            normalized_rois = tf.gather(normalized_rois, idx_labels)
            head_offsets = tf.gather(head_offsets, idx_labels)
            labels_prob = tf.gather(labels_prob, idx_labels)
            labels_id = tf.gather(labels_id, idx_labels)
            # labels_prob = log.tfprint(labels_prob,
            #   "labels_prob:inloop:debug")
            # labels_id = log.tfprint(labels_id, "labels_id:inloop:debug")

            # ラベルの確率の高い順に並べる
            _, idx_labels_order = tf.nn.top_k(
                labels_prob, k=K.shape(labels_prob)[0], sorted=True)
            normalized_rois = tf.gather(normalized_rois, idx_labels_order)
            head_offsets = tf.gather(head_offsets, idx_labels_order)
            labels_prob = tf.gather(labels_prob, idx_labels_order)
            labels_id = tf.gather(labels_id, idx_labels_order)

            # bbox復元
            h, w = self.config.image_shape[0], self.config.image_shape[1]
            bboxes = K.cast(bbox.restore_bbox(normalized_rois, head_offsets,
                                              h, w),
                            tf.float32)

            # バッチ毎にNMSする
            # TODO オブジェクトの種類が複数であれば、オブジェクトの種類ごとにNMSしたほうがよさそう。
            idx_keep = tf.image.non_max_suppression(
                bboxes, labels_prob,
                max_output_size=self.config.detect_max_instances,
                iou_threshold=self.config.detect_nms_thresh)
            bboxes = tf.gather(bboxes, idx_keep)
            normalized_rois = tf.gather(normalized_rois, idx_keep)
            labels_prob = tf.gather(labels_prob, idx_keep)
            labels_id = tf.gather(labels_id, idx_keep)

            diff = K.cast(K.maximum(
                self.config.detect_max_instances - K.shape(bboxes)[0], 0),
                tf.int32)
            bboxes = tf.pad(bboxes, [[0, diff], [0, 0]],
                            mode='CONSTANT', constant_values=0,
                            name="pad_bboxes")
            normalized_rois = tf.pad(normalized_rois, [[0, diff], [0, 0]],
                                     mode='CONSTANT', constant_values=0,
                                     name="pad_normalized_rois")
            labels_prob = tf.pad(labels_prob, [[0, diff]],
                                 mode='CONSTANT', constant_values=0,
                                 name="pad_labels_prob")
            labels_id = tf.pad(labels_id, [[0, diff]],
                               mode='CONSTANT', constant_values=0,
                               name="pad_labels_id")

            ret_bboxes.append(K.cast(bboxes, tf.int32))
            ret_normalized_rois.append(normalized_rois)
            ret_labels.append(K.cast(labels_id, tf.int32))
            ret_scores.append(labels_prob)

        return [K.stack(ret_bboxes), K.stack(ret_normalized_rois),
                K.stack(ret_labels), K.stack(ret_scores)]

    def _nn_mask(self, backbone_out, normalized_rois):
        """Builds the computation graph of the mask head of Feature Pyramid Network.

            Args:
                backbone_out: backboneネットワークの出力
                rois: 正規化されたRoI
                    [N, n_rois, (y1, x1, y2, x2)]
            Returns:
                masks:
                    [N, n_rois, num_classes, pool_size*2, pool_size*2]
        """
        # [N, R, pool_size, pool_size, channels]
        out = RoiAlignLayer(self.config.mask_roi_align_pool_shape, self.config,
                            name='head_mask_roi_align')([backbone_out,
                                                         normalized_rois])

        # 畳み込み層は論文通り4層。フィーチャマップの解像度は維持
        out = TimeDistributed(Conv2D(256, (3, 3), padding='same'),
                              name='head_mask_conv1')(out)
        out = TimeDistributed(BatchNorm(axis=3),
                              name='head_mask_conv1_bn')(out)
        out = Activation('relu')(out)

        out = TimeDistributed(Conv2D(256, (3, 3), padding='same'),
                              name='head_mask_conv2')(out)
        out = TimeDistributed(BatchNorm(axis=3),
                              name='head_mask_conv2_bn')(out)
        out = Activation('relu')(out)

        out = TimeDistributed(Conv2D(256, (3, 3), padding='same'),
                              name='head_mask_conv3')(out)
        out = TimeDistributed(BatchNorm(axis=3),
                              name='head_mask_conv3_bn')(out)
        out = Activation('relu')(out)

        out = TimeDistributed(Conv2D(256, (3, 3), padding='same'),
                              name='head_mask_conv4')(out)
        out = TimeDistributed(BatchNorm(axis=3),
                              name='head_mask_conv4_bn')(out)
        out = Activation('relu')(out)

        # 解像度を28*28（倍）に上げるための逆畳み込み
        out = TimeDistributed(Conv2DTranspose(256, (2, 2), strides=2),
                              name='head_mask_deconv')(out)
        out = Activation('relu')(out)
        out = TimeDistributed(Conv2D(self.config.n_dataset_labels,
                                     (1, 1), strides=1, activation='sigmoid'),
                              name='head_mask_binary')(out)
        # [N, n_rois, n_label, h ,w]に変形
        out = Reshape([-1, self.config.n_dataset_labels,
                       self.config.mask_out_shape[0],
                       self.config.mask_out_shape[1]],
                      name='head_mask_reshape')(out)
        return out

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
                h, w = self.config.image_shape[0], self.config.image_shape[1]
                input_gt_masks = Input(
                    shape=[None, h, w], name="input_gt_masks", dtype='float32')
                inputs += [input_gt_boxes, input_gt_label_ids, input_gt_masks]

                # 正解データとRoIから評価対象のRoIを絞り込み、それに対応する正解データを得る。
                normalized_sample_rois, normalized_sample_gt_offsets, \
                    sample_gt_labels, sample_gt_masks = \
                    MaskSubsamplingRoiLayer(self.config,
                                            name='mask_subsampling')(
                        [normalized_rois, input_gt_boxes,
                            input_gt_label_ids, input_gt_masks])

                # head
                head_offsets, labels, labels_logit \
                    = self._nn_head(backbone_out, normalized_sample_rois)
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

                # mask
                masks = self._nn_mask(backbone_out, normalized_sample_rois)
                head_mask_loss = Lambda(lambda x:
                                        loss.head_mask_loss(*x),
                                        name="head_mask_loss")(
                    [sample_gt_masks, sample_gt_labels, masks])

                # 損失
                losses += [head_offsets_loss, head_labels_loss, head_mask_loss]

            # 出力＝損失
            outputs = losses

        else:
            # 予測時
            # head
            # head_offsetsは0〜1で正規化された値
            head_offsets, labels, _ = self._nn_head(
                backbone_out, normalized_rois)

            # 候補を絞り込む
            bboxes, rois, labels, scores = Lambda(lambda x:
                                                  self._nn_squeeze_roi(*x),
                                                  name="squeeze_roi")(
                [normalized_rois, head_offsets, labels])

            # mask
            masks = self._nn_mask(backbone_out, rois)

            def _squeeze_masks(masks, idx_labels):
                dim1 = K.flatten(K.repeat(K.expand_dims(
                    K.arange(K.shape(masks)[0])), K.shape(masks)[1]))
                dim2 = K.tile(K.arange(K.shape(masks)[1]),
                              [K.shape(masks)[0]])
                idx = K.stack([dim1, dim2,
                               K.cast(K.flatten(labels), tf.int32)], axis=1)
                # idx = log.tfprint(idx, "idx:inloop:debug")
                squeezed_masks = tf.gather_nd(masks, idx)
                squeezed_masks = K.reshape(squeezed_masks,
                                           [K.shape(masks)[0],
                                            K.shape(masks)[1],
                                            K.shape(masks)[3],
                                            K.shape(masks)[4]])
                return squeezed_masks

            # ラベルに対応するマスクを残す
            masks = Lambda(lambda x: _squeeze_masks(x[0],
                                                    K.cast(x[1], tf.int32)),
                           name="squeeze_mask")([masks, labels])

            # 予測時は損失不要
            # ダミーの損失関数
            dummy_loss = Lambda(lambda x: K.constant(0), name="dummy_loss")(
                [backbone_in])
            losses = [dummy_loss, dummy_loss, dummy_loss,
                      dummy_loss, dummy_loss, dummy_loss]
            inputs = [backbone_in]

            outputs = [bboxes, labels, scores, masks, rois,
                       rpn_offsets, objects]

        model = Model(inputs=inputs, outputs=outputs, name='mask_r_cnn')
        # Kerasは複数指定した損失の合計をモデル全体の損失として評価してくれる。
        # 損失を追加
        for output in losses:
            model.add_loss(tf.reduce_mean(output, keep_dims=True))
        return model, len(outputs)


class MaskSubsamplingRoiLayer(SubsamplingRoiLayer):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

    def compute_output_shape(self, input_shape):
        return [(self.config.batch_size, self.n_samples_per_batch, 4),
                (self.config.batch_size, self.n_samples_per_batch, 4),
                (self.config.batch_size, self.n_samples_per_batch),
                (self.config.batch_size, self.n_samples_per_batch,
                 self.config.mask_out_shape[0], self.config.mask_out_shape[1])]

    def _subsampling(self, normalized_rois, gt_bboxes, gt_labels, gt_masks,
                     pos_iou_thresh=0.5,
                     exclusive_iou_tresh=0.1,
                     pos_ratio=0.25):
        """
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
            gt_masks (ndarray) : 正解mask。
                (N, config.n_max_gt_objects_per_image,
                    config.image_shape[0], config.image_shape[1])
                座標は正規化されていない。
        Returns:
            sample_rois (tensor): サンプリングしたRoI。
                (N, n_samples_per_batch, 4)
                3軸目の座標は0〜1に正規化された値。
            sample_gt_offset (tensor): サンプリングしたRoIに対応するBBoxとのオフセット。
                (N, n_samples_per_batch, 4)
                3軸目の座標は0〜1に正規化された値をself.config.bbox_refinement_stdで割ることで標準化した値。
            sample_gt_labels (tensor): サンプリングしたRoIに対応するBBoxのラベル。
                (N, n_samples_per_batch)
            sample_gt_masks (tensor): サンプリングしたRoIに対応するmask。
                _nn_mask()で得られる特徴マップのサイズにリサイズされている。
                (N, n_samples_per_batch, config.mask_out_shape[0],
                    config.mask_out_shape[1])
        """
        # TODO maskに関する冗長なコードの排除
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
        gt_masks = tf.split(gt_masks, self.config.batch_size)

        sample_rois = []
        sample_gt_offsets = []
        sample_gt_labels = []
        sample_gt_masks = []

        for roi, gt_bbox, gt_label, gt_mask in zip(normalized_rois,
                                                   normalized_gt_bboxes,
                                                   gt_labels,
                                                   gt_masks):
            # 0次元目(バッチサイズ)は不要なので削除
            roi = K.squeeze(roi, 0)
            gt_bbox = K.squeeze(gt_bbox, 0)
            gt_label = K.squeeze(gt_label, 0)
            gt_mask = K.squeeze(gt_mask, 0)  # mask

            # ゼロパディング行を除外
            idx_roi_row = K.flatten(tf.where(K.any(roi, axis=1)))
            idx_gt_bbox = K.flatten(tf.where(K.any(gt_bbox, axis=1)))
            roi = K.gather(roi, idx_roi_row)
            # gt_bbox, gt_label, gt_masksは行数と行の並びが同じなので同じidxを利用できる
            gt_bbox = K.gather(gt_bbox, idx_gt_bbox)
            gt_label = K.gather(gt_label, idx_gt_bbox)
            gt_mask = K.gather(gt_mask, idx_gt_bbox)  # mask

            # IoUを求める。
            # (n_rois, )
            ious = bbox.get_iou_K(roi, gt_bbox)

            # 各RoI毎にIoU最大のBBoxの位置を得る
            idx_max_gt = K.argmax(ious, axis=1)

            max_iou = K.max(ious, axis=1)  # max_iouの行数はroiと同じになる
            idx_pos = K.flatten(tf.where(max_iou >= pos_iou_thresh))
            # positiveサンプル数をpos_roi_per_batch以内に制限
            limit_pos = K.minimum(pos_roi_per_batch, K.shape(idx_pos)[0])
            idx_pos = K.switch(K.shape(idx_pos)[0] > 0,
                               tf.random_shuffle(idx_pos)[:limit_pos],
                               idx_pos)

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

            # 返却するサンプルを抽出
            # GTのoffsets, labelsは各roisに対応させる。つまり、同じ位置に格納する。
            idx_keep = K.cast(K.concatenate((idx_pos, idx_neg)), tf.int32)

            # 各RoIの最大IoUを示すIndexについても、上記返却するサンプルのみを残す。
            idx_gt_keep = K.cast(K.gather(idx_max_gt, idx_keep), tf.int32)
            # IoUが閾値以上のPositiveとみなされるサンプルのみを残すためのIndex。
            idx_gt_keep_pos = K.cast(K.gather(idx_max_gt, idx_pos), tf.int32)

            # 残すべきRoIを残す
            sample_roi_pre_pad = K.gather(roi, idx_keep)
            # sample_roiに対応するgt_bboxとのオフセットを求める
            sample_gt_offset = bbox.get_offset_K(
                sample_roi_pre_pad, K.gather(gt_bbox, idx_gt_keep))
            # sample_gt_offsetの算出に用いたbboxに対応するラベルを残す
            # negativeな要素には0を設定
            sample_gt_label = K.concatenate((K.cast(K.gather(
                gt_label, idx_gt_keep_pos),
                dtype='int32'),
                K.zeros([limit_neg],  # K.zerosは0階テンソルを受け付けないので配列化。。。
                        dtype='int32')))
            sample_gt_mask = K.gather(gt_mask, idx_gt_keep)  # mask

            # 行数がn_samples_per_batch未満の場合は0パディング
            remain = tf.maximum(self.n_samples_per_batch
                                - tf.shape(sample_roi_pre_pad)[0], 0)
            sample_roi = tf.pad(sample_roi_pre_pad, [(0, remain), (0, 0)],
                                name='subsample_sample_roi')
            sample_gt_offset = tf.pad(sample_gt_offset, [(0, remain), (0, 0)],
                                      name='subsample_sample_gt_offset')
            sample_gt_offset /= self.config.bbox_refinement_std
            sample_gt_label = tf.pad(sample_gt_label, [(0, remain)],
                                     name='subsample_sample_gt_label')

            # maskのバイナリマップをProposalLayerで得られるRoIでclipし、_nn_mask()で得られる特徴マップのサイズにリサイズ。
            # _nn_mask()で得られるマスクは、RoIを基準とするため。
            # RoIがズレていると評価にならない。。。RoIの精度が上がらないとネットワーク最終結果の精度は上がらない。。。
            # tf.image.crop_and_resize でリサイズするため、4次元のテンソルに変換
            sample_gt_mask = tf.expand_dims(sample_gt_mask, -1)
            idx_box = K.arange(0, K.shape(sample_roi_pre_pad)[0])
            sample_gt_mask = tf.image.crop_and_resize(
                sample_gt_mask, sample_roi_pre_pad, idx_box,
                self.config.mask_out_shape)
            # crop_and_resizeのために追加した4次元目を削除
            sample_gt_mask = tf.squeeze(sample_gt_mask, axis=3)
            # リサイズの結果発生する少数を四捨五入して0 or 1に戻す。
            sample_gt_mask = tf.round(sample_gt_mask)
            # bbox, offset等と同様にpadding
            sample_gt_mask = tf.pad(sample_gt_mask, [(0, remain),  # mask
                                                     (0, 0), (0, 0)],
                                    name='subsample_sample_gt_mask')

            sample_rois.append(sample_roi)
            sample_gt_offsets.append(sample_gt_offset)
            sample_gt_labels.append(sample_gt_label)
            sample_gt_masks.append(sample_gt_mask)  # mask

        return [K.stack(sample_rois), K.stack(sample_gt_offsets),
                K.stack(sample_gt_labels), K.stack(sample_gt_masks)]  # mask
