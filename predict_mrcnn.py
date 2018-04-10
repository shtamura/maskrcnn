import argparse
import logging
import re
import tensorflow as tf
from keras import backend as K
import numpy as np
import scipy.misc
from xrcnn.config import Config
from xrcnn.mrcnn import MaskRCNN
from xrcnn.util.anchor import Anchor
from xrcnn.util import bbox
from xrcnn.util import image
# from xrcnn.util import log

import cv2

FORMAT = '%(asctime)-15s %(levelname)s #[%(thread)d] %(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO)

logger = logging.getLogger(__name__)
logger.info("---start---")


def add_rect(dest_img, box, color, thickness):
    cv2.rectangle(dest_img, (box[1], box[0]),
                  (box[3], box[2]),
                  color, thickness=thickness)


def add_mask(dest_img, mask, bbox, color, image_shape):
    threshold = 0.5
    y1, x1, y2, x2 = bbox
    h, w = y2 - y1, x2 - x1
    logger.debug("y1, x1, y2, x2: %s, h, w: %s", (y1, x1, y2, x2), (h, w))
    logger.debug("mask.shape: %s", mask.shape)
    mask = scipy.misc.imresize(mask, (h, w),
                               interp='bilinear').astype(np.float32)
    # scipy.misc.imresizeの結果は0~255にスケールされるので、0〜1に戻す。
    mask /= 255.0
    # 0 or 1に変換。
    mask = np.where(mask >= threshold, 1, 0).astype(np.uint8)

    # 0~image_shapeの枠外のマスクは除外する
    _y1, _x1, _y2, _x2 = max(0, y1), max(0, x1), min(image_shape[0], y2), \
        min(image_shape[1], x2)
    d_y1, d_x1, d_y2, d_x2 = _y1 - y1, _x1 - x1, _y2 - y2, _x2 - x2
    mask = mask[d_y1:h + d_y2, d_x1:w + d_x2]

    # マスクを画像に配置。image_shapeは入力画像の[h, w]
    fullsize_mask = np.zeros(image_shape, dtype=np.uint8)
    fullsize_mask[_y1:_y2, _x1:_x2] = mask

    logger.debug("mask.shape: %s, image_shape: %s, bbox: %s (%s) ",
                 mask.shape, image_shape, bbox, (y2 - y1, x2 - x1))
    logger.debug("d_y1, d_x1, d_y2, d_x2: %s, mask.shape: %s ",
                 (d_y1, d_x1, d_y2, d_x2), mask.shape)

    # # mask
    mask_image = np.zeros(image_shape + [3], dtype=np.uint8)
    mask_image[:, :] = color
    mask_image = cv2.bitwise_and(mask_image, mask_image, mask=fullsize_mask)
    # mask = np.dstack([mask, mask, mask])
    # mask[:, :, 0][mask[:, :, 0] == 1] = color[0]
    # mask[:, :, 1][mask[:, :, 1] == 1] = color[1]
    # mask[:, :, 2][mask[:, :, 2] == 1] = color[2]
    cv2.addWeighted(mask_image, 1.5, dest_img, 1, 0, dest_img)


config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
K.set_session(session)
# set_debugger_session()

config = Config()
anchor = Anchor(config)
config.training = False
config.batch_size = 1
# 学習時に利用したデータセットに含まれるラベル数を指定する。
config.n_dataset_labels = 1 + 1  # 背景 + people
logger.warn("指定されたラベル数: %s. 学習時のラベル数と異なる場合、エラーになります。",
            config.n_dataset_labels)

# dump tensor
# log.out_name_pattern = ".+debug$"

argparser = argparse.ArgumentParser(description="FasterRCNNで物体検出")
argparser.add_argument('--image_path', type=str,
                       required=True, help="処理対象の画像ファイルパス")
argparser.add_argument('--weights_path', type=str,
                       required=True, help="モデルの重みファイルのパス")
argparser.add_argument('--rpn', type=bool,
                       required=False, help="RPNの予測結果を表示")
args = argparser.parse_args()

mrcnn = MaskRCNN(anchor.anchors, config)


def pred(image_path):
    # 画像をnumpy配列として読み込む
    img = image.load_image_as_ndarray(image_path)
    img = img.astype(np.uint8)
    logger.debug("img.shape: %s", img.shape)
    # 学習時と同様にリサイズ
    img, _, _ = image.resize_with_padding(
        img,
        config.image_min_size,
        config.image_max_size)

    # バッチサイズの次元を追加
    input_img = np.array([img])
    logger.debug("input_img.shape.resized: %s", input_img.shape)
    # logger.info("window: %s", window)
    # logger.info("scale: %s", scale)

    # 表示用画像はopencvに合わせてRGBからBGRへ変換
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    bboxes, labels, scores, masks, rois, rpn_offsets, rpn_objects = \
        model.predict([input_img], verbose=1,
                      batch_size=config.batch_size)

    # バッチサイズを示す1次元目を削除
    bboxes, labels, scores, masks, rois, rpn_offsets, rpn_objects = \
        np.squeeze(bboxes, axis=0), \
        np.squeeze(labels, axis=0), \
        np.squeeze(scores, axis=0), \
        np.squeeze(masks, axis=0), \
        np.squeeze(rois, axis=0), \
        np.squeeze(rpn_offsets, axis=0), \
        np.squeeze(rpn_objects, axis=0)

    save_path_suffix = re.split('/|\.', image_path)
    save_path_suffix = save_path_suffix[-2] + '.png'
    if args.rpn:
        # 前景のみ
        rpn_obj_pos = rpn_objects[:, 0]
        rpn_anchor = anchor.anchors
        # スコア降順
        idx_pos = rpn_obj_pos.argsort()[::-1]
        rpn_obj_pos = rpn_obj_pos[idx_pos]
        rpn_offsets = rpn_offsets[idx_pos]
        rpn_anchor = rpn_anchor[idx_pos]
        # 上位50件で
        top = 10
        rpn_obj_pos = rpn_obj_pos[:top]
        rpn_offsets = rpn_offsets[:top]
        rpn_anchor = rpn_anchor[:top]

        rpn_offsets *= np.array(config.bbox_refinement_std)

        print(rpn_obj_pos, rpn_offsets, rpn_anchor)
        boxes = bbox.get_bbox(rpn_anchor,
                              rpn_offsets)
        boxes = boxes.clip(0,
                           config.image_max_size).astype('int32')
        for i, box in enumerate(boxes):
            box = box.astype('int32')
            add_rect(img, box, (0, 0, 255), 1)

        save_path = './pred_rpn_' + save_path_suffix
    else:
        # # ラベルに対応するマクスを残す
        # masks = masks[np.arange(masks.shape[0]), ]

        # 背景,paddingは除く
        idx_labels = np.where(labels > 0)
        bboxes = bboxes[idx_labels]
        labels = labels[idx_labels]
        scores = scores[idx_labels]
        masks = masks[idx_labels]
        rois = rois[idx_labels]

        h, w = config.image_shape[0], config.image_shape[1]
        rois *= [h, w, h, w]
        logger.debug("rois.shape: %s", rois.shape)
        logger.debug("rois: %s", rois)
        logger.debug("bboxes.shape: %s", bboxes.shape)
        logger.debug("bboxes: %s", bboxes)
        logger.debug("labels.shape: %s", labels.shape)
        # logger.debug("labels: %s", labels)
        logger.debug("scores.shape: %s", scores.shape)
        logger.debug("scores: %s", scores)
        logger.debug("masks.shape: %s", masks.shape)
        # logger.debug("masks: %s", masks)

        blue = [i for i in range(255)[::(255 // (bboxes.shape[0] + 1))]]
        green = blue[::-1]

        # Proposal表示
        for roi in rois:
            add_rect(img, roi, (0, 0, 255), 1)

        # bbox表示
        for box, mask, b, g in zip(bboxes, masks, blue, green):

            # bbox, mask表示
            add_rect(img, box, (b, g, 0), 2)
            add_mask(img, mask, box, (b, g, 0), config.image_shape[:2])

        save_path = './pred_' + save_path_suffix

    cv2.imwrite(save_path, img)


with tf.device('/cpu:0'):
    model = mrcnn.compiled_model()
    logger.debug("compile model.")

# with tf.device('/gpu:1'):
    model.load_weights(args.weights_path, by_name=True)
    logger.debug("load_weights.")

    paths = args.image_path.split(',')
    for path in paths:
        pred(path)
