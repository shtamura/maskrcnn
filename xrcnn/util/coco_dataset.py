import os
import logging
import random
import cv2
import numpy as np
from pycocotools.coco import COCO

import xrcnn.util.image as image

logger = logging.getLogger(__name__)


class Generator:
    def __init__(self, config, data_root_dir, data_type='val2017',
                 target_category_names=['person']):
        # TODO ['person']以外が指定された場合にカテゴリIDからone_hot化しているところが破綻する。要改善。
        """
            Args:
                category_names
                ['person']のみの指定、もしくは指定なし（全カテゴリ）のみサポート。
        カテゴリIDとカテゴリ名の対応は以下の通り
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
        22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
        43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
        62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
        85, 86, 87, 88, 89, 90]
        ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
         'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
         'sports ball', 'kite', 'baseball bat', 'baseball glove',
         'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
            'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
            'book', 'clock', 'vase', 'scissors', 'teddy bear',
            'hair drier', 'toothbrush']
        """
        self.config = config
        self.data_root_dir = data_root_dir
        self.annotation_path = '{}/annotations/instances_{}.json'.format(
            data_root_dir, data_type)
        self.image_dir_path = '{}/{}/'.format(data_root_dir, data_type)
        self.coco = COCO(self.annotation_path)
        self._target_category_names = target_category_names
        # 背景＝0を追加
        self._category_ids = self.coco.getCatIds(
            catNms=self._target_category_names)

    def get_labels(self):
        cats = self.coco.loadCats(self._category_ids)
        # 背景を追加
        return ['__background__'] + [cat['name'] for cat in cats]

    def get_label_ids(self):
        return [0] + self._category_ids

    def _get_all_image_ids(self):
        # annotationが存在するイメージのみに限定。
        image_ids = set()
        for id in self.get_label_ids():
            image_ids |= set(self.coco.getImgIds(catIds=id))
        return image_ids

    def _get_metas(self, image_ids):
        """ load COCO dataset.

            Returns:
                annotation
                  |- images
                        |- filepath
                        |- size : (width, height)
                        |- objects
                            |- label_id（COCOは1オリジン。0は背景を示す。）
                            |- iscrowd
                            |- bbox : (ymin, xmin, ymax, xmax)
                            |- mask : [width, height, 1] バイナリマスク
        """
        img_metas = self.coco.loadImgs(ids=image_ids)
        metas = []
        for image_meta in img_metas:
            # {'license': 1,
            # 'file_name': '000000002685.jpg',
            # 'coco_url':
            #   'http://images.cocodataset.org/val2017/000000002685.jpg',
            # 'height': 555,
            # 'width': 640,
            # 'date_captured': '2013-11-25 19:10:39',
            # 'flickr_url':
            #   'http://farm9.staticflickr.com/8535/8710326856_2aac3d36fb_z.jpg',
            # 'id': 2685}
            # iscrowd=Falseで群衆を含むデータを除く
            image_id = image_meta['id']
            a_ids = self.coco.getAnnIds(imgIds=image_id, iscrowd=False)
            annotations = self.coco.loadAnns(a_ids)

            if len(annotations) > 0:
                image_path = os.path.join(
                    self.image_dir_path, image_meta['file_name'])
                width = int(image_meta['width'])
                height = int(image_meta['height'])
                meta = {'image_path': image_path,
                        'size': (width, height),
                        'objects': []}

                for annotation in annotations:
                    # {'segmentation': [[574.96, ..., 440.26]],
                    # 'area': 36959.305749999985,
                    # 'iscrowd': 0,
                    # 'image_id': 2685,
                    # 'bbox': [315.54, 56.12, 323.02, 384.14],
                    # 'category_id': 1,
                    # 'id': 1226144}
                    label_id = annotation['category_id']
                    # 対象外のカテゴリIDは除外
                    if label_id not in self.get_label_ids():
                        continue

                    # バイナリマスク
                    mask = self.coco.annToMask(annotation)

                    # bboxはmask要素から抽出
                    # annotationのbboxがmaskよりも小さい事があるみたい
                    idx = np.where(mask == 1)
                    bbox = np.array([np.min(idx[0]), np.min(idx[1]),
                                     np.max(idx[0]), np.max(idx[1])])

                    meta['objects'].append({
                        'label_id': label_id,
                        'bbox': bbox,
                        'mask': mask
                    })

                # 対象とするオブジェクトを含む情報のみ追加する
                if len(meta['objects']) > 0:
                    metas.append(meta)
                else:
                    logger.warn("image_id[%s] has no object." % image_id)

            else:
                logger.warn("image_id[%s] has no object." % image_id)

        logger.debug("load_image_meta: %s", metas)
        return metas

    def generate(self, anchor, n_max=None, image_ids=None,
                 include_mask=True):
        """
        VOCイメージセットをkeras.model.fit_generatorで使用するgeneratorの形式で取得する。
            Args:
            Returns:

        """
        if image_ids is None:
            image_ids = list(self._get_all_image_ids())

        random.shuffle(image_ids)

        # 複数GPU利用の場合は入力データが各GPUに均等に配分される。
        # モデル内でconfig.batch_size＝バッチサイズとして実装しているところがあるので、
        # GPU数毎に入力データがconfig.batch_sizeとなるよう掛けておく。
        batch_size = self.config.batch_size * self.config.gpu_count
        batch_count = 0
        head_trainable = self.config.training_mode in ['head_only', 'all']
        while True:
            for image_id in image_ids:
                try:
                    if batch_count == 0:
                        images = []
                        rpn_offsets = []
                        rpn_fbs = []
                        bboxes = []
                        labels = []
                        masks = []

                    metas = self._get_metas([image_id])
                    if len(metas) == 0:
                        continue
                    meta = metas[0]

                    logger.info("loaded image: %s", meta['image_path'])
                    # 画像を規定のサイズにリサイズ。
                    img = image.load_image_as_ndarray(meta['image_path'])
                    # モノクロ（2次元データ）は除く
                    # val2017/000000061418.jpg など
                    if(len(img.shape) < 3):
                        logger.warn("skip 2 dim image: %s", meta['image_path'])
                        continue

                    img, window, scale = image.resize_with_padding(
                        img,
                        self.config.image_min_size,
                        self.config.image_max_size)
                    logger.debug("window, scale: %s, %s", window, scale)
                    # ランダムにflip
                    img, flip_x, flip_y = image.random_flip(img)

                    # 画像毎のオブジェクト数は固定にする。複数画像を1つのテンソルにおさめるため。
                    bb = np.zeros([self.config.n_max_gt_objects_per_image, 4])
                    bb_raw = []
                    lb = np.zeros([self.config.n_max_gt_objects_per_image])
                    mk = np.zeros([self.config.n_max_gt_objects_per_image,
                                   self.config.image_max_size,
                                   self.config.image_max_size])
                    mk_raw = []
                    # bbox, maskもリサイズ＆flip
                    for i, obj in enumerate(meta['objects']):
                        b = image.flip_bbox(
                            image.resize_bbox(obj['bbox'], window[:2], scale),
                            img.shape[:2], flip_x, flip_y)
                        # boxサイズが小さすぎるオブジェクトは除外
                        h, w = b[2] - b[0], b[3] - b[1]
                        if h <= self.config.ignore_box_size \
                                or w <= self.config.ignore_box_size:
                            continue
                        bb_raw.append(b)

                        m = image.flip_mask(
                            image.resize_mask(obj['mask'],
                                              self.config.image_min_size,
                                              self.config.image_max_size),
                            flip_x, flip_y)
                        mk_raw.append(m)

                        lb[i] = obj['label_id']

                    # 有効なラベルが1つもないデータは無効なので返却しない
                    if not np.any(lb > 0):
                        continue

                    # RPN向けのGTをまとめる
                    of, fb = anchor.generate_gt_offsets(
                        np.array(bb_raw), self.config.image_shape[:2])
                    logger.debug("shapes: offset: %s, fb: %s", of.shape,
                                 fb.shape)

                    images.append(img)
                    rpn_offsets.append(of)
                    rpn_fbs.append(fb)

                    bb[:len(bb_raw), :] = bb_raw
                    bboxes.append(bb)

                    mk[:len(mk_raw), :] = mk_raw
                    masks.append(mk)

                    labels.append(lb)

                    if np.any(np.argwhere(np.isnan(of))):
                        logger.error("nanを含むオフセットを検出！スキップします。")
                        continue

                    batch_count += 1
                    if batch_count >= batch_size:
                        batch_count = 0
                        inputs = [np.array(images), np.array(rpn_offsets),
                                  np.array(rpn_fbs)]
                        if head_trainable:
                            inputs += [np.array(bboxes), np.array(labels)]
                            if include_mask:
                                inputs += [np.array(masks)]
                        yield inputs, []

                except ValueError as e:
                    logger.error("想定外エラー")
                    logger.error(e)
                    continue

    def show_images(self, anchor, n_max=None, image_ids=None):
        iter = self.generate(anchor, n_max, image_ids)
        for data, _ in iter:
            img = data[0][0]
            # rgb -> bgr
            img = np.flip(img, axis=2).astype(np.uint8)
            boxes = data[3][0]
            masks = data[5][0]
            # 0パディングした行は除く
            idx_pos = np.where(np.any(boxes, axis=1))[0]
            boxes = boxes[idx_pos]
            masks = masks[idx_pos]
            c = [i for i in range(255)[::(255 // boxes.shape[0] - 1)]]
            i = 0
            for bbox, mask in zip(boxes, masks):
                bbox = bbox.astype(np.uint8)
                mask = mask.astype(np.uint8)
                color = (c[i], c[::-1][i], 0)
                # bbox
                cv2.rectangle(img, (bbox[1], bbox[0]),
                              (bbox[3], bbox[2]), color)
                # # mask
                # mask_img = np.zeros(img.shape, img.dtype)
                # mask_img[:, :] = color
                mask = np.dstack([mask, mask, mask])
                mask[:, :, 0][mask[:, :, 0] == 1] = color[0]
                mask[:, :, 1][mask[:, :, 1] == 1] = color[1]
                mask[:, :, 2][mask[:, :, 2] == 1] = color[2]
                # mask_img = cv2.bitwise_and(mask_img, mask_img, mask=mask)
                cv2.addWeighted(mask, 1, img, 1, 0, img)
                i += 1
            cv2.imshow('img', img)
            cv2.waitKey(0)
