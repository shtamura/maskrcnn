import os
import logging
import random
import xml.etree.ElementTree as ET
import cv2
import numpy as np

import xrcnn.util.image as image

logger = logging.getLogger(__name__)

_dir_voc_imageset = 'ImageSets'
_dir_voc_annotation = 'Annotations'
_dir_voc_image = 'JPEGImages'

# VOCデータセットで提供されるラベルに背景（非オブジェクト）を示すラベルを加える。
# 背景=0とするため、リスト先頭に加える。
_voc_labels = ['__background__', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse', 'motorbike',
               'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor']


def get_pascal_voc_labels():
    return _voc_labels


def pascal_voc_data_generator(path, anchor, config, train_val='train',
                              n_max=None, prefix=None):
    """
    VOCイメージセットをkeras.model.fit_generatorで使用するgeneratorの形式で取得する。
        Args:
        Returns:

    """
    if prefix is not None:
        logger.info("load specific meta: %s", prefix)
        image_meta = load_image_meta(path, prefix)
        image_metas = [image_meta]
    else:
        if train_val == 'train':
            logger.info("load voc train data.")
            image_metas, _ = load_pascal_voc_traindata(path, n_max)
        else:
            logger.info("load voc validation data.")
            image_metas, _ = load_pascal_voc_validationdata(path, n_max)

    random.shuffle(image_metas)

    # 複数GPU利用の場合は入力データが各GPUに均等に配分される。
    # モデル内でconfig.batch_size＝バッチサイズとして実装しているところがあるので、
    # GPU数毎に入力データがconfig.batch_sizeとなるよう掛けておく。
    batch_size = config.batch_size * config.gpu_count
    batch_count = 0
    head_trainable = config.training_mode in ['head_only', 'all']
    while True:
        for meta in image_metas:
            if batch_count == 0:
                images = []
                rpn_offsets = []
                rpn_fbs = []
                bboxes = []
                labels = []

            # 画像を規定のサイズにリサイズ。
            img = image.load_image_as_ndarray(meta['image_path'])
            img, window, scale = image.resize_with_padding(
                img,
                config.image_min_size,
                config.image_max_size)
            logger.debug("window, scale: %s, %s", window, scale)
            # ランダムにflip
            img, flip_x, flip_y = image.random_flip(img)
            images.append(img)

            # 画像毎のオブジェクト数は固定にする。複数画像を1つのテンソルにおさめるため。
            bb = np.zeros([config.n_max_gt_objects_per_image, 4])
            bb_raw = []
            lb = np.zeros([config.n_max_gt_objects_per_image])
            # bboxもリサイズ＆flip
            for i, obj in enumerate(meta['objects']):
                b = image.flip_bbox(
                    image.resize_bbox(obj['bbox'], window[:2], scale),
                    img.shape[:2], flip_x, flip_y)
                bb_raw.append(b)
                lb[i] = obj['label_id']
            logger.debug("bb_raw: %s", bb_raw)
            # RPN向けのGTをまとめる
            of, fb = anchor.generate_gt_offsets(np.array(bb_raw),
                                                config.image_shape[:2])
            logger.debug("shapes: offset: %s, fb: %s", of.shape, fb.shape)

            # 有効なラベルが1つもないデータは無効なので返却しない
            if not np.any(lb > 0):
                continue

            rpn_offsets.append(of)
            rpn_fbs.append(fb)

            bb[:len(bb_raw), :] = bb_raw
            bboxes.append(bb)
            labels.append(lb)

            logger.info("loaded image: %s", meta['image_path'])

            if np.any(np.argwhere(np.isnan(of))):
                raise ValueError("nanを含むオフセットを検出！")

            batch_count += 1
            if batch_count >= batch_size:
                batch_count = 0
                inputs = [np.array(images), np.array(rpn_offsets),
                          np.array(rpn_fbs)]
                if head_trainable:
                    inputs += [np.array(bboxes), np.array(labels)]
                yield inputs, []


def load_image_meta(path, prefix):
    annotation_path = os.path.join(
        path, _dir_voc_annotation, prefix + '.xml')

    try:
        # parse annotation file
        xml = ET.parse(annotation_path)
        root = xml.getroot()
        objects = root.findall('object')

        if len(objects) > 0:
            image_path = os.path.join(
                path, _dir_voc_image, root.find('filename').text)
            width = int(root.find('size').find('width').text)
            height = int(root.find('size').find('height').text)
            segmented = bool(int(root.find('segmented').text))
            data = {'image_path': image_path,
                    'size': (width, height),
                    'segmented': segmented,
                    'objects': []}

            for obj in objects:
                name = obj.find('name').text
                try:
                    # 0は背景（非オブジェクト）を示すため、0オリジンとする。
                    name_id = _voc_labels.index(name) + 1
                except ValueError as e:
                    # 想定外のラベルなので処理しない
                    logger.warn(e)
                    continue
                truncated = bool(int(obj.find('truncated').text))
                difficult = bool(int(obj.find('difficult').text))
                bbox = obj.find('bndbox')
                xmin = int(round(float(bbox.find('xmin').text)))
                ymin = int(round(float(bbox.find('ymin').text)))
                xmax = int(round(float(bbox.find('xmax').text)))
                ymax = int(round(float(bbox.find('ymax').text)))

                data['objects'].append({
                    'label': name,
                    'label_id': name_id,
                    'truncated': truncated,
                    'difficult': difficult,
                    'bbox': np.array([ymin, xmin, ymax, xmax])
                })

            logger.debug("load_image_meta: %s", data)
            return data

        else:
            logger.warn("%s has no object." % annotation_path)
            return None

    except Exception as e:
        logger.warn(e)
        raise e


def _load_pascal_voc(path, imagelist, n_max=None):
    """ load PASCAL VOC dataset.
        http://host.robots.ox.ac.uk/pascal/VOC/

        Args:
            path parent directory of VOC dataset.

        Returns:
            VOC Image annotation as dictionary.
            annotation
              |- images
                    |- filepath
                    |- size : (width, height)
                    |- segmented
                    |- objects
                        |- label
                        |- label_id（1オリジンとする。0は背景を示す。）
                        |- truncated
                        |- difficult
                        |- bbox : [ymin, xmin, ymax, xmax]
    """
    images = []
    labels = set()
    target_data_list = os.path.join(path, _dir_voc_imageset, 'Main', imagelist)
    with open(target_data_list, 'r') as f:
        for i, line in enumerate(f):
            if n_max is not None and i >= n_max:
                break
            line = line.strip()
            meta = load_image_meta(path, line)
            images.append(meta)
            for obj in meta['objects']:
                for label in obj['label']:
                    labels.add(label)

    return images, labels


def load_pascal_voc_traindata(path, n_max=None):
    return _load_pascal_voc(path, 'train.txt', n_max)


def load_pascal_voc_validationdata(path, n_max=None):
    return _load_pascal_voc(path, 'val.txt', n_max)


def show_voc_image(image):
    img = cv2.imread(image['image_path'])
    for obj in image['objects']:
        bbox = obj['bbox']
        cv2.rectangle(img, (bbox[1], bbox[0]), (bbox[3], bbox[2]), (0, 0, 255))
    cv2.imshow('img', img)
    cv2.waitKey(0)


def show(path, prefix):
    meta = load_image_meta(path, prefix)
    show_voc_image(meta)

# def resize(image, min_size, max_size):
#     size = image['size']
#     w_org = size[0]
#     w, h = _get_resiezed_imagesize(size[0], size[1], min_size, max_size)
#     image['size'] = (w, h)
#
#     scale = float(w) / float(w_org)
#     for obj in image['objects']:
#         bbox = obj['bbox']
#         bbox[0][0] = int(bbox[0][0] * scale)
#         bbox[0][1] = int(bbox[0][1] * scale)
#         bbox[1][0] = int(bbox[1][0] * scale)
#         bbox[1][1] = int(bbox[1][1] * scale)
