from keras import backend as K
import tensorflow as tf
import logging

from xrcnn.util import log

logger = logging.getLogger(__name__)

"""
    RPNの損失関数
        J(p→,u,v→,t→)=Jcls(p→,u)+λ[u>=1]Jloc(v→,t→)

        Jcls(p→,u)=−log⁡pu

        Jloc=∑i∈{x,y,w,h}smoothL1(ti−vi)
            smoothL1(x)={
                0.5 * x^2   if(|x|<1)
                |x|−0.5 otherwise
                }
"""


def sparse_categorical_crossentropy(gt_ids, pred_one_hot_post_softmax):
    """
    K.sparse_categorical_crossentropyだと結果がNaNになる。。。
    0割り算が発生しているかも。
    https://qiita.com/4Ui_iUrz1/items/35a8089ab0ebc98061c1
    対策として、微少値を用いてlog(0)にならないよう調整した本関数を作成。
    """
    gt_ids = log.tfprint(gt_ids, "cross:gt_ids:")
    pred_one_hot_post_softmax = log.tfprint(pred_one_hot_post_softmax,
                                            "cross:pred_one_hot_post_softmax:")

    gt_one_hot = K.one_hot(gt_ids, K.shape(pred_one_hot_post_softmax)[-1])
    gt_one_hot = log.tfprint(gt_one_hot, "cross:gt_one_hot:")

    epsilon = K.epsilon()  # 1e-07
    loss = -K.sum(
        gt_one_hot * K.log(
            tf.clip_by_value(pred_one_hot_post_softmax, epsilon, 1 - epsilon)),
        axis=-1)
    loss = log.tfprint(loss, "cross:loss:")
    return loss


def smooth_l1(gt, pred):
    # https://qiita.com/GushiSnow/items/8c946208de0d6a4e31e7
    diff = K.abs(gt - pred)
    less_than_one = K.cast(K.less(diff, 1.0), "float32")
    # difffが1より小さい場合、less_than_one==1
    loss = (less_than_one * 0.5 * diff**2) + (1 - less_than_one) * (diff - 0.5)
    return loss


def offsets_loss(gt_offsets, pred_offsets, dump=False):
    """オフセット回帰の損失関数
    positive（gt_fg > 0）データのみ評価対象とする

    Args:
        gt_offsets: 正解オフセット
            [R, 4]
            3軸目は領域提案とアンカーのオフセット（中心、幅、高さ）。
                (tx, ty, th, tw)
        pred_offsets: 予測値
            [R, 4].

    Note:
        この関数の呼び出し元はrpn_offsets_lossとhead_offsets_loss。
        RPNでのRoI予測が外れると全てNegativeなBBoxとなり、結果的にhead_offsets_lossへ渡される正解データのラベルが全てNegativeとなる。
        その場合、head_offsets_lossで得られる損失は0となるが、rpn_offsets_lossで得られる損失は大きくなるはずなので、
        損失全体(rpn_offsets_loss + head_offsets_loss)で評価すれば適切な損失になるはず。
    """
    loss = K.switch(tf.size(gt_offsets) > 0,
                    smooth_l1(gt_offsets, pred_offsets), tf.constant(0.0))
    loss = K.mean(loss)
    return loss


def rpn_offsets_loss(gt_offsets, gt_fg, pred_offsets):
    """RPNのオフセット回帰の損失関数
    positive（gt_fg > 0）データのみ評価対象とする

    gt_offsets: 正解オフセット
        [N, R, 4]
        3軸目は領域提案とアンカーのオフセット（中心、幅、高さ）。
            (tx, ty, th, tw)
    gt_fg: 正解データの前景／背景
        [N, R]
    pred_offsets: 予測値
        [N, R, 4].
    """
    pos_idx = tf.where(gt_fg > 0)
    gt_offsets = tf.gather_nd(gt_offsets, pos_idx)
    pred_offsets = tf.gather_nd(pred_offsets, pos_idx)
    # FasterRCNNの論文上は、RPNのオフセット回帰には係数10を乗ずることでオブジェクト分類損失とのバランスを取ることになっている。
    # が、rpnの損失の全損失に占める割合が高すぎるようなら係数調整
    p = 1.
    loss = p * offsets_loss(gt_offsets, pred_offsets)
    loss = log.tfprint(loss, "rpn_offsets_loss")
    return loss


def head_offsets_loss(gt_offsets, gt_labels, pred_offsets):
    """ヘッドのオフセット回帰の損失関数
    positive（gt_fg > 0）データのみ評価対象とする

    gt_offsets: 正解オフセット
        [N, R, 4]
    gt_labels: 正解データのラベルID
        [N, R]
    pred_offsets: ラベル毎の予測値
        [N, R, n_labels, 4].
    """

    # 正解データのラベルIDに対応するオフセットのみを損失評価対象とする。
    # 論文には以下のようにあるので、正解ラベルのBBoxのみで良さそう。
    # The second task loss, Lloc, is defined over a tuple of true bounding-box
    # regression targets for class u, v = (vx, vy, vw, vh), and a predicted
    # tuple tu = (tux , tuy , tuw, tuh ), again for class u.
    pos_idx = tf.where(gt_labels > 0)
    i = K.cast(pos_idx[:, 0], tf.int32)
    j = K.cast(pos_idx[:, 1], tf.int32)
    k = K.cast(tf.gather_nd(gt_labels, pos_idx), tf.int32)
    pos_pred_idx = K.stack((i, j, k), axis=1)
    pred_offsets = tf.gather_nd(pred_offsets, pos_pred_idx)
    gt_offsets = tf.gather_nd(gt_offsets, pos_idx)

    loss = offsets_loss(gt_offsets, pred_offsets)
    loss = log.tfprint(loss, "head_offsets_loss")
    return loss


def labels_loss(gt, pred):
    """ラベル分類の損失関数

    gt: 正解
        [N, R]
        2軸目はラベルを示すID
    pred: 予測値(softmax済み)
        [N, R, labels].
    """

    # 交差エントロピー誤差
    # バッチ毎の計算ではなく、全体の平均値でOK。
    # 論文に以下の記載がある。
    #    In our current implementation (as in the released code),
    #    the cls term in Eqn.(1) is normalized by the mini-batch size
    #    (i.e., Ncls = 256) and the reg term is normalized by the number of
    #    anchor locations (i.e., Nreg ∼ 2, 400).
    gt = K.cast(gt, 'int32')
    loss = K.switch(tf.size(gt) > 0,
                    sparse_categorical_crossentropy(gt, pred), K.constant(0.0))
    loss = K.mean(loss)
    return loss


def rpn_objects_loss(gt, pred):
    """RPNのオブジェクト／非オブジェクト分類の損失関数

    gt: 正解
        [N, anchors]
        2軸目の値は以下の通り。
        positive=1, negative=0, neutral(exclude from eval)=-1
    pred: 予測値(softmax済み)
        [batch, anchors, 2].
        3軸目はオブジェクトor非オブジェクトを示す数値。
    """
    # 評価対象外の−1に該当する要素を除く
    indices = tf.where(gt > -1)
    # print("indicies", indices)
    # print("pred", pred)
    gt = tf.gather_nd(gt, indices)
    pred = tf.gather_nd(pred, indices)

    # 交差エントロピー誤差
    # バッチ毎の計算ではなく、全体の平均値でOK。
    # 論文に以下の記載がある。
    #    In our current implementation (as in the released code),
    #    the cls term in Eqn.(1) is normalized by the mini-batch size
    #    (i.e., Ncls = 256) and the reg term is normalized by the number of
    #    anchor locations (i.e., Nreg ∼ 2, 400).
    loss = labels_loss(gt, pred)
    loss = log.tfprint(loss, "rpn_objects_loss")
    return loss


def head_labels_loss(gt, pred):
    """ヘッドのラベル分類の損失関数

    gt: 正解
        [N, R]
        2軸目はラベルを示すID
    pred: 予測値(softmax済み)
        [N, R, labels].
    """
    gt = log.tfprint(gt, "head_labels_loss_val:gt", summarize=1024)
    pred = log.tfprint(pred, "head_labels_loss_val:pred", summarize=1024)
    loss = labels_loss(gt, pred)
    loss = log.tfprint(loss, "head_labels_loss")
    return loss


def head_mask_loss(gt_masks, gt_labels, pred_masks):
    """マスクの損失関数

    gt_masks: 正解データ。
        マスクデータをbboxの領域のみ切り抜いてconfig.mask_out_shapeにリサイズしたデータ。
        [N, R, h, w]
        バイナリマスク
    gt_labels: 正解データのラベルID
        [N, R]
    pred_masks: 予測値
        バイナリマスク
        [N, R, n_labels h, w]
    ※h, w は config.mask_out_shape になる。
    """
    # Positiveなラベルが付与されているRoIのみ評価対象とする
    pos_idx = tf.where(gt_labels > 0)
    i = K.cast(pos_idx[:, 0], tf.int32)
    j = K.cast(pos_idx[:, 1], tf.int32)
    k = K.cast(tf.gather_nd(gt_labels, pos_idx), tf.int32)
    # i = log.tfprint(i, "i:head_mask_loss")
    # j = log.tfprint(j, "j:head_mask_loss")
    # k = log.tfprint(k, "k:head_mask_loss")
    pos_pred_idx = K.stack((i, j, k), axis=1)
    # pos_pred_idx = log.tfprint(pos_pred_idx, "pos_pred_idx:head_mask_loss")
    pred_masks = tf.gather_nd(pred_masks, pos_pred_idx)
    gt_masks = tf.gather_nd(gt_masks, pos_idx)

    loss = K.switch(tf.size(gt_masks) > 0,
                    K.binary_crossentropy(gt_masks, pred_masks),
                    tf.constant(0.0))
    loss = K.mean(loss)
    loss = log.tfprint(loss, "head_mask_loss")
    return loss
