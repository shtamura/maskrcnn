import argparse
import logging
import numpy as np
import xrcnn.util.anchor as anchor
import xrcnn.util.bbox as bbox
import xrcnn.config as config
import xrcnn.util.dataset as dataset
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

FORMAT = '%(asctime)-15s %(levelname)s #[%(thread)d] %(message)s'
logging.basicConfig(format=FORMAT, level=logging.DEBUG)

logger = logging.getLogger(__name__)
logger.info("---start---")

argparser = argparse.ArgumentParser()
argparser.add_argument('--path', type=str,
                       required=True, help="VOCデータセットが配置してあるディレクトリ")
argparser.add_argument('--prefix', type=str,
                       required=True)
args = argparser.parse_args()

conf = config.Config()
anc = anchor.Anchor(conf)

di = dataset.pascal_voc_data_generator(args.path, anc, conf, train_val='train',
                                       n_max=3, prefix=args.prefix)
data, _ = next(di)
# print(data)

img = data[0]
rpn_offset = data[1]
rpn_offset *= np.array(conf.bbox_refinement_std)
rpn_fbs = data[2]
pos_idx = np.where(rpn_fbs == 1)
pos_anchor = anc.anchors[pos_idx[1]]
pos_offset = rpn_offset[pos_idx[0], pos_idx[1]]
box = bbox.get_bbox(pos_anchor, pos_offset)
box = box.astype('int32')
print(box)
img = np.squeeze(img, axis=0)
print(img.shape)

fig, ax = plt.subplots(1)
ax.imshow(img)


def add_rect(dest_ax, bbox):
    rect = patches.Rectangle((bbox[1], bbox[0]),
                             bbox[3] - bbox[1], bbox[2] - bbox[0],
                             linewidth=1, edgecolor='r', facecolor='none',)
    dest_ax.add_patch(rect)


for b in box:
    add_rect(ax, b)
plt.show()
plt.close()


def add_rect_cv(dest_img, box, color):
    cv2.rectangle(dest_img, (box[1], box[0]),
                  (box[3], box[2]),
                  color)


for i, b in enumerate(box):
    add_rect_cv(img, b, (255, 0, 0))

cv2.imwrite('./check_gt.png', img)


#  確認
# get_bboxする値の尺度がGTと予測結果でズレてそう。。。
#   →これは大丈夫だった。。。子要素しか取得しないのでOK
# XMLファイルにある矩形以上の情報が得られている。。。
#
#  GTの値に全てのBBOXが含まれていない？2つあるはずのGTBBOXが1つになっている。。。
#  data_generator
#   矩形がネストしているケースあり。。。
#       人->顔、手、脚、的な。。。
#   座標がおかしい。。。x1=x2、y1=y2になっている。。。
# [np.array(images), np.array(rpn_offsets),
#        np.array(rpn_fbs), np.array(bboxes),
#        np.array(labels)], []
# ↑確認済み
#
