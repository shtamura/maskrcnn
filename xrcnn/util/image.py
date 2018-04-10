import skimage.io
import scipy.misc
import numpy as np
import random

from logging import getLogger

logger = getLogger(__name__)


def load_image_as_ndarray(image_path):
    """画像を読み込んで、[h,w,channel(RGB)]形式のnumpy配列として取得する。
        Args:
            image_path: 画像ファイルのパス
        Returns:
            [h,w,channel(RGB)]形式
    """
    image = skimage.io.imread(image_path)
    return image


def resize_with_padding(image_array, min_size, max_size):
    """アスペクト比を維持したままリサイズする。
    高さ、または幅の小さい方がmin_sizeとなるようリサイズする。
    リサイズの結果、高さ、または幅の大きい方がmax_sizeを超える場合は、高さ、または幅の大きい方をmax_sizeとする。
    リサイズ後画像を max_size*max_size の枠の中央に配置し、周辺を0でPaddingする。

    Args:
        image_array: [h,w,3]の配列
        min_size:
        max_size:

    Returns:
        resized_image: リサイズ後の画像
        window: (y1, x1, y2, x2). リサイズ後の画像が画像全体のどの位置にあるかを示す座標
        scale: 元画像に対してのスケール
    """
    h, w = image_array.shape[:2]
    window = (0, 0, h, w)
    scale = 1

    scale = max(1, min_size / min(h, w))

    # max_sizeを超えるないよう調整
    image_max = max(h, w)
    if round(image_max * scale) > max_size:
        scale = max_size / image_max

    if scale != 1:
        image_array = scipy.misc.imresize(image_array,
                                          (round(h * scale), round(w * scale)))
    # Padding
    h, w = image_array.shape[:2]
    top_pad = (max_size - h) // 2
    bottom_pad = max_size - h - top_pad
    left_pad = (max_size - w) // 2
    right_pad = max_size - w - left_pad
    padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
    image_array = np.pad(image_array, padding,
                         mode='constant', constant_values=0)
    window = (top_pad, left_pad, h + top_pad, w + left_pad)

    return image_array, window, scale


def resize_mask(mask, padding_top_left, scale):
    """
        Args:
            mask: バイナリマスク
                [height, width]
    """
    # [height, width] -> [height, width, 3]
    mask = np.dstack([mask, mask, mask])
    mask, _, _ = resize_with_padding(mask, padding_top_left, scale)
    # [height, width, 3] -> [height, width]
    mask = np.reshape(mask[:, :, 0], mask.shape[:2])
    return mask


def resize_bbox(bbox, padding_top_left, scale):
    logger.debug("resize_bbox:in: %s %s %s",
                 bbox, padding_top_left, scale)
    # top_left(y, x)　をscaleだけ大きくした矩形に足すことでPadding分ずらす
    bbox = bbox * scale + np.tile(padding_top_left, 2)
    logger.debug("resize_bbox:out: %s", bbox)
    return bbox


def random_flip(image_array, force_flip=False):
    x_flip = random.choice([True, False]) | force_flip
    # 上下逆転は結果の精度を落とすっぽい。
    y_flip = False  # random.choice([True, False]) | force_flip

    img = image_array.copy()
    if y_flip:
        img = np.flip(img, axis=0)
    if x_flip:
        img = np.flip(img, axis=1)
    return img, x_flip, y_flip


def flip_mask(mask, x_flip, y_flip):
    """
        Args:
            mask: バイナリマスク
                [height, width]
    """
    mask = mask.copy()
    if y_flip:
        mask = np.flip(mask, axis=0)
    if x_flip:
        mask = np.flip(mask, axis=1)
    return mask


def flip_bbox(bbox, image_size, x_flip, y_flip):
    logger.debug("flip_bbox:in: %s %s %s %s",
                 bbox, image_size, x_flip, y_flip)
    h, w = image_size
    flipped_bbox = bbox.copy()
    if y_flip:
        flipped_bbox[0] = h - bbox[2]  # top
        flipped_bbox[2] = h - bbox[0]  # bottom
    if x_flip:
        flipped_bbox[1] = w - bbox[3]  # left
        flipped_bbox[3] = w - bbox[1]  # right
    # print(image_size, ":", (bbox[2] - bbox[0], bbox[3] - bbox[1]),
    #       (flipped_bbox[2] - flipped_bbox[0],
    #       flipped_bbox[3] - flipped_bbox[1]))
    logger.debug("flip_bbox:out: %s", flipped_bbox)
    return flipped_bbox
