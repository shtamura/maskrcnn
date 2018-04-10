import math


class Config:

    train_data_path = '../dataset/VOCdevkit/VOC2007'

    # data augumentation setting
    # 教師データ加工(data augmentation)の有無
    use_horizontal_flips = False
    use_vertical_flips = False
    rot_90 = False

    # anchor box ratios
    # アンカーボックスのアスペクト比
    # [(h, w), ...]
    # 面積は固定
    anchor_box_aspect_ratios = [
        (1. / math.sqrt(2), 2. / math.sqrt(2)),
        (1., 1.),
        (2. / math.sqrt(2), 1. / math.sqrt(2))
    ]

    # bboxに適用する精度向上の為のパラメータ
    # TODO 値の根拠について調査。ひとまずは参考とする実装にあるパラメータを指定する。
    # bbox_refinement_std = [1.0, 1.0, 1.0, 1.0]
    bbox_refinement_std = [0.1, 0.1, 0.2, 0.2]

    # 学習時に利用するオブジェクトの最大数（画像1つ当たり）
    n_max_gt_objects_per_image = 100

    # non-non_maximum_suppression(NMS)の閾値。
    nms_thresh = 0.7

    # NMS前にこの閾値まで領域数を削減する。
    # トレーニングモードで利用されるパラメータ。
    n_train_pre_nms = 12000

    # NMS後にこの閾値まで領域数を削減する。
    # トレーニングモードで利用されるパラメータ。
    n_train_post_nms = 2000

    # NMS前にこの閾値まで領域数を削減する。
    # テストモードで利用されるパラメータ。
    n_test_pre_nms = 6000

    # NMS後にこの閾値まで領域数を削減する。
    # テストモードで利用されるパラメータ。
    n_test_post_nms = 300

    # バッチサイズ
    # 予測時は1にすること。
    batch_size = 2

    # RoIAlignの出力サイズ
    roi_align_pool_shape = [7, 7]
    mask_roi_align_pool_shape = [14, 14]

    # データセットに含まれるラベルの種類（背景を示すラベルも含む）
    n_dataset_labels = 21

    # Trueであればトレーニングモード。
    training = True
    # rpn_only でRPNのみのトレーニング。
    # 全体でトレーニングするとRPNのOffsetの損失がNaNになるため、
    # まずはRPNからトレーニング。
    # training_mode = 'all' | 'rpn_only' | 'head_only'
    training_mode = 'all'

    # バックボーンネットワークの種類
    # vgg, resnet
    backbone_nn_type = 'vgg'

    # GPU利用の場合は0以上（利用可能なGPU数を指定）
    gpu_count = 0

    # NNトレーニング時の学習率
    learning_rate = 0.001

    # 検出時に行うnon-non_maximum_suppression(NMS)の閾値。
    # 同一のオブジェクトに対するbbox予測の重複を排除する。
    detect_nms_thresh = 0.3
    # 予測結果として採用するラベル確率
    detect_label_prob = 0.7
    # 予測結果として得られる件数
    detect_max_instances = 100

    def __init__(self):
        # Number of pixels per pixel on base network feature map
        # ベースネットワークの特徴マップ1ピクセル当たりの入力画像におけるピクセル数
        if self.backbone_nn_type == 'vgg':
            # VGG16をベースにするため16ピクセルになる（stride=2*2の畳み込みが4回行われるため、サイズが元の1/16になる。）
            self.stride_per_base_nn_feature = 16
            self.image_max_size = 224  # Kerasの学習済みモデルのInputに合わせて224
        else:
            # ResNet50をベースにするため場合は32ピクセルになる（stride=2*2の畳み込みが5回行われるため、サイズが元の1/32になる。）
            self.stride_per_base_nn_feature = 32
            self.image_max_size = 224  # Kerasの学習済みモデルのInputに合わせて224

        # # リサイズ後の最小サイズ
        # image_min_size = 600
        # # リサイズ後の最大サイズ
        # image_max_size = 1024
        # imagenet画像でpretrainしたバックボーンネットワークを使う場合のサイズ
        self.image_min_size = 150

        # 入力画像のサイズを基準としたアンカーボックスのピクセル数
        # 縦横同一サイズ
        # anchor_box_scales = [128, 256, 512]
        # Kerasの学習済みモデル（入力が224*224）を使う場合のサイズ
        self.anchor_box_scales = [32, 64, self.image_max_size // 2]

        self.n_anchor = len(self.anchor_box_scales) * \
            len(self.anchor_box_aspect_ratios)

        # Input image size
        # (h, w, 3)
        self.image_shape = [self.image_max_size, self.image_max_size, 3]
        # # backboneネットワークの出力サイズ
        self.backbone_shape = [
            int(math.ceil(self.image_shape[0] /
                          self.stride_per_base_nn_feature)),
            int(math.ceil(self.image_shape[1] /
                          self.stride_per_base_nn_feature))
        ]

        # 評価対象外にするBBOXのサイズ
        # 特徴抽出が困難と思われる小さなBoxを除外する
        self.ignore_box_size = self.image_max_size // 20

        # maskネットワークの出力サイズは入力時のプーリングサイズの2倍
        self.mask_out_shape = [self.mask_roi_align_pool_shape[0] * 2,
                               self.mask_roi_align_pool_shape[1] * 2]
