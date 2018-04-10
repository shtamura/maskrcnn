if __name__ == '__main__':
    import argparse
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import xrcnn.util.dataset as dataset
    import xrcnn.util.image as FI
    from xrcnn.config import Config

    config = Config()

    def add_rect(dest_ax, bbox):
        rect = patches.Rectangle((bbox[1], bbox[0]),
                                 bbox[3] - bbox[1], bbox[2] - bbox[0],
                                 linewidth=1, edgecolor='r', facecolor='none',)
        dest_ax.add_patch(rect)

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--path', type=str,
                           required=True)
    args = argparser.parse_args()

    images, _ = dataset.load_pascal_voc_traindata(args.path, 4)
    for image in images:
        # original
        img = FI.load_image_as_ndarray(image['image_path'])
        fig, ax = plt.subplots(1)
        ax.imshow(img)
        for obj in image['objects']:
            bbox = obj['bbox']
            add_rect(ax, bbox)
        plt.show()
        plt.close()

        # resize
        resized_image, window, scale = FI.resize_with_padding(
            img, config.image_min_size, config.image_max_size)
        fig, ax = plt.subplots(1)
        ax.imshow(resized_image)
        resized_bbox = []
        for obj in image['objects']:
            bbox = obj['bbox']
            resized = FI.resize_bbox(bbox, window[:2], scale)
            resized_bbox.append(resized)
            add_rect(ax, resized)
        plt.show()
        plt.close()

        # flip
        flipped_image, x_flip, y_flip = FI.random_flip(resized_image)
        fig, ax = plt.subplots(1)
        ax.imshow(flipped_image)
        for bbox in resized_bbox:
            flipped = FI.flip_bbox(bbox,
                                   flipped_image.shape[:2], x_flip, y_flip)
            add_rect(ax, flipped)
        plt.show()
        plt.close()
