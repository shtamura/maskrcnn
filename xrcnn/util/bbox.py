from keras import backend as K
import numpy as np
import tensorflow as tf


def normalize_bbox(input_bboxes, input_h, input_w):
    """入力データのbboxを0~1に正規化する。
    入力画像の縦横で割る。
    NNから得られる予測値として利用する。
    従って損失評価時にはGTをこの関数を通した値を利用する。
    """
    return input_bboxes / K.variable([input_h, input_w, input_h, input_w])


def get_bbox(src_bbox, offset):
    """src_bboxにoffsetを適用し、元の領域を復元する。
    RPNから得たoffset予測値をアンカーボックスに適用して提案領域を得る。といったケースで利用する。

    Args:
        src_bbox (tensor / ndarray): オフセットを適用するBoudingBox。
            Its shape is :math:`(R, 4)`.
            2軸目に以下の順でBBoxの座標を保持する。
            :math:`p_{ymin}, p_{xmin}, p_{ymax}, p_{xmax}`.
        offset (tensor / ndarray): オフセット。
            形状はsrc_bboxに同じ。
            2軸目にオフセットの形状を保持数r。 :math:`t_y, t_x, t_h, t_w`.
                tx =(x−xa)/wa, ty =(y−ya)/ha, tw = log(w/wa), th = log(h/ha)
                ※それぞれ、アンカーからのオフセット
                ※「x」は予測された領域の中心x、「xa」はアンカーの中心x。

    Returns:
        tensor:
        オフセットを適用したBoudingBox。
        形状はsrc_bboxに同じ。
        1軸目はsrc_bboxと同じ情報を示す。
        2軸目にはオフセットを適用した座標を保持する。
        :math:`\\hat{g}_{ymin}, \\hat{g}_{xmin},
        \\hat{g}_{ymax}, \\hat{g}_{xmax}`.

    """
    if type(src_bbox) == np.ndarray and type(offset) == np.ndarray:
        xp = np
    else:
        xp = K

    if src_bbox.shape[0] == 0:
        return xp.zeros((0, 4), dtype=offset[:, 0].dtype)

    # src_bbox（anchorなど）の左上と右下の座標から、中心座標＋高さ＋幅の形式に変換する
    src_height = src_bbox[:, 2] - src_bbox[:, 0]
    src_width = src_bbox[:, 3] - src_bbox[:, 1]
    src_ctr_y = src_bbox[:, 0] + 0.5 * src_height
    src_ctr_x = src_bbox[:, 1] + 0.5 * src_width

    # オフセットを中心座標、高さ、幅毎にまとめる
    dy = offset[:, 0]
    dx = offset[:, 1]
    dh = offset[:, 2]
    dw = offset[:, 3]

    # 論文にあるオフセット算出式（以下）から逆算
    # tx =(x−xa)/wa, ty =(y−ya)/ha, tw = log(w/wa), th = log(h/ha)
    # ※それぞれ、アンカーからのオフセット
    # ※「x」は予測された領域の中心x、「xa」はアンカーの中心x。
    ctr_y = dy * src_height + src_ctr_y
    ctr_x = dx * src_width + src_ctr_x
    h = xp.exp(dh) * src_height
    w = xp.exp(dw) * src_width

    # 矩形の左上と右下の座標に変換
    ymin = ctr_y - 0.5 * h
    xmin = ctr_x - 0.5 * w
    ymax = ctr_y + 0.5 * h
    xmax = ctr_x + 0.5 * w
    bbox = xp.transpose(xp.stack((ymin, xmin, ymax, xmax), axis=0))
    return bbox


def restore_bbox(normalized_rois, normalized_offsets, input_h, input_w):
    """
    正規化されたRoIとオフセットから、入力画像にスケールアップしたBBOXを取得する。
    Args:
        normalized_rois (ndarray):
        normalized_offsets (ndarray):
    """
    is_numpy = type(normalized_rois) == np.ndarray \
        and type(normalized_offsets) == np.ndarray
    if is_numpy:
        xp = np
        box = np.array([input_h, input_w, input_h, input_w])
    else:
        xp = K
        box = K.variable([input_h, input_w, input_h, input_w])

    normalized_bboxes = get_bbox(normalized_rois, normalized_offsets)
    bboxes = normalized_bboxes * box
    bboxes = xp.round(bboxes)
    if is_numpy:
        bboxes = bboxes.astype(xp.int32)
    else:
        bboxes = K.cast(bboxes, tf.int32)

    return bboxes


def get_offset(src_bbox, dst_bbox):
    """src_bboxからdst_bboxを得るために必要なオフセットを取得する。
    get_bbox(src_bbox, offset) => dst_bbox となる。

    Args:
        src_bbox (ndarray): 基準となるBoudingBox。
            Its shape is :math:`(R, 4)`.
            2軸目に以下の順でBBoxの座標を保持する。
            :math:`p_{ymin}, p_{xmin}, p_{ymax}, p_{xmax}`.
        dst_bbox (ndarray): 基準となるBoudingBox。
            Its shape is :math:`(R, 4)`.
            2軸目に以下の順でBBoxの座標を保持する。
            :math:`p_{ymin}, p_{xmin}, p_{ymax}, p_{xmax}`.

    Returns:
        ndarray:
        オフセット
        This has shape :math:`(R, 4)`.
        The second axis contains four values :math:`t_y, t_x, t_h, t_w`.

    """
    epsilon = 1e-07

    # src_bboxを中心座標＋高さ＋幅の形式に変換する
    height = src_bbox[:, 2] - src_bbox[:, 0]
    width = src_bbox[:, 3] - src_bbox[:, 1]
    ctr_y = src_bbox[:, 0] + 0.5 * height
    ctr_x = src_bbox[:, 1] + 0.5 * width

    # dst_bboxも同じく変換する
    base_height = dst_bbox[:, 2] - dst_bbox[:, 0]
    base_width = dst_bbox[:, 3] - dst_bbox[:, 1]
    base_ctr_y = dst_bbox[:, 0] + 0.5 * base_height
    base_ctr_x = dst_bbox[:, 1] + 0.5 * base_width

    # 0除算にならないよう調整
    height = np.maximum(height, epsilon)
    width = np.maximum(width, epsilon)

    # 論文にあるオフセット算出式より
    # tx =(x−xa)/wa, ty =(y−ya)/ha, tw = log(w/wa), th = log(h/ha)
    dy = (base_ctr_y - ctr_y) / height
    dx = (base_ctr_x - ctr_x) / width
    # print(height, width, base_height, base_width)
    dh = np.log(base_height / height)
    dw = np.log(base_width / width)

    # (R, 4)の形状に変換
    offset = np.transpose(np.stack((dy, dx, dh, dw), axis=0))
    return offset


def get_offset_K(src_bbox, dst_bbox):
    """src_bboxからdst_bboxを得るために必要なオフセットを取得する。
    get_bbox(src_bbox, offset) => dst_bbox となる。

    Args:
        src_bbox (tensor): 基準となるBoudingBox。
            Its shape is :math:`(R, 4)`.
            2軸目に以下の順でBBoxの座標を保持する。
            :math:`p_{ymin}, p_{xmin}, p_{ymax}, p_{xmax}`.
        dst_bbox (tensor): 基準となるBoudingBox。
            Its shape is :math:`(R, 4)`.
            2軸目に以下の順でBBoxの座標を保持する。
            :math:`p_{ymin}, p_{xmin}, p_{ymax}, p_{xmax}`.

    Returns:
        tensor:
        オフセット
        This has shape :math:`(R, 4)`.
        The second axis contains four values :math:`t_y, t_x, t_h, t_w`.

    """
    epsilon = K.epsilon()

    # src_bboxを中心座標＋高さ＋幅の形式に変換する
    height = src_bbox[:, 2] - src_bbox[:, 0]
    width = src_bbox[:, 3] - src_bbox[:, 1]
    ctr_y = src_bbox[:, 0] + 0.5 * height
    ctr_x = src_bbox[:, 1] + 0.5 * width

    # dst_bboxも同じく変換する
    base_height = dst_bbox[:, 2] - dst_bbox[:, 0]
    base_width = dst_bbox[:, 3] - dst_bbox[:, 1]
    base_ctr_y = dst_bbox[:, 0] + 0.5 * base_height
    base_ctr_x = dst_bbox[:, 1] + 0.5 * base_width

    # 0除算にならないよう調整
    height = K.maximum(height, epsilon)
    width = K.maximum(width, epsilon)

    # 論文にあるオフセット算出式より
    # tx =(x−xa)/wa, ty =(y−ya)/ha, tw = log(w/wa), th = log(h/ha)
    dy = (base_ctr_y - ctr_y) / height
    dx = (base_ctr_x - ctr_x) / width
    dh = K.log(base_height / height)
    dw = K.log(base_width / width)

    # (R, 4)の形状に変換
    offset = K.transpose(K.stack((dy, dx, dh, dw), axis=0))
    return offset


def get_iou(bbox_base, bbox_target):
    """2つのBoundingBoxのIoU（Intersection Over Union）を取得する。
        https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
    Args:
        bbox_base (ndarray): 基準になるBoudingBox。
            Its shape is :math:`(N, 4)`.
            2軸目に以下の順でBBoxの座標を保持する。
            :math:`p_{ymin}, p_{xmin}, p_{ymax}, p_{xmax}`.
        bbox_target (ndarray): BoudingBox。
            Its shape is :math:`(K, 4)`.
            2軸目に以下の順でBBoxの座標を保持する。
            :math:`p_{ymin}, p_{xmin}, p_{ymax}, p_{xmax}`.

        bbox_baseの各Box毎にbbox_targetを適用し、IoUを求める。

    Returns:
        ndarray:
        IoU(0 <= IoU <= 1)
        形状は以下の通り。
        :math:`(N, K)`.

    """
    if bbox_base.shape[1] != 4 or bbox_target.shape[1] != 4:
        raise IndexError

    # 交差領域の左上の座標
    # bbox_base[:, None, :]のより次元を増やすことで、
    # bbox_baseとbbox_targetを総当りで評価出来る。
    # (N, K, 2)の座標が得られる
    tl = np.maximum(bbox_base[:, None, :2], bbox_target[:, :2])
    # 交差領域の右下の座標
    # (N, K, 2)の座標が得られる
    br = np.minimum(bbox_base[:, None, 2:], bbox_target[:, 2:])

    # 右下-左下＝交差領域の(h, w)が得られる。
    # h*wで交差領域の面積。ただし、交差領域がない（右下 <= 左上）ものは除くため0とする。
    area_i = np.prod(br - tl, axis=2) * \
        np.all(br > tl, axis=2).astype('float32')
    area_base = np.prod(bbox_base[:, 2:] - bbox_base[:, :2], axis=1)
    area_target = np.prod(bbox_target[:, 2:] - bbox_target[:, :2], axis=1)
    return area_i / (area_base[:, None] + area_target - area_i)


def get_iou_K(bbox_base, bbox_target):
    """2つのBoundingBoxのIoU（Intersection Over Union）を取得する。
        https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
    Args:
        bbox_base (tensor): 基準になるBoudingBox。
            Its shape is :math:`(N, 4)`.
            2軸目に以下の順でBBoxの座標を保持する。
            :math:`p_{ymin}, p_{xmin}, p_{ymax}, p_{xmax}`.
        bbox_target (tensor): BoudingBox。
            Its shape is :math:`(K, 4)`.
            2軸目に以下の順でBBoxの座標を保持する。
            :math:`p_{ymin}, p_{xmin}, p_{ymax}, p_{xmax}`.

        bbox_baseの各Box毎にbbox_targetを適用し、IoUを求める。

    Returns:
        tensor:
        IoU(0 <= IoU <= 1)
        形状は以下の通り。
        :math:`(N, K)`.

    """
    if bbox_base.shape[1] != 4 or bbox_target.shape[1] != 4:
        raise IndexError

    # 交差領域の左上の座標
    # bbox_base[:, None, :]のより次元を増やすことで、
    # bbox_baseとbbox_targetを総当りで評価出来る。
    # (N, K, 2)の座標が得られる
    tl = K.maximum(bbox_base[:, None, :2], bbox_target[:, :2])
    # 交差領域の右下の座標
    # (N, K, 2)の座標が得られる
    br = K.minimum(bbox_base[:, None, 2:], bbox_target[:, 2:])

    # 右下-左下＝交差領域の(h, w)が得られる。
    # h*wで交差領域の面積。ただし、交差領域がない（右下 <= 左上）ものは除くため0とする。
    area_i = K.prod(br - tl, axis=2) * \
        K.cast(K.all(br > tl, axis=2), 'float32')
    area_base = K.prod(bbox_base[:, 2:] - bbox_base[:, :2], axis=1)
    area_target = K.prod(bbox_target[:, 2:] - bbox_target[:, :2], axis=1)
    return area_i / (area_base[:, None] + area_target - area_i)
