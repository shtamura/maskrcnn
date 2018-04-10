import argparse
import logging
import tensorflow as tf
from keras import backend as K
import keras.callbacks
from keras import utils
from xrcnn.config import Config
from xrcnn.mrcnn import MaskRCNN
from xrcnn.util.anchor import Anchor
import xrcnn.util.coco_dataset as coco_dataset


from tensorflow.python import debug as tf_debug
from xrcnn.util import log


def name_filter(datum, tensor):
    print(datum.tensor_name)
    return "sample_gt_mask" in datum.tensor_name


def set_debugger_session():
    sess = K.get_session()
    sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    sess.add_tensor_filter('name_filter', name_filter)
    K.set_session(sess)


FORMAT = '%(asctime)-15s %(levelname)s #[%(thread)d] %(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO)

logger = logging.getLogger(__name__)
logger.info("---start---")

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
K.set_session(session)
# set_debugger_session()

config = Config()
anchor = Anchor(config)

argparser = argparse.ArgumentParser(description="FasterRCNNのトレーニング")
argparser.add_argument('--data_path', type=str,
                       required=True, help="COCOデータセットが配置してあるディレクトリ")
argparser.add_argument('--max_sample', type=int,
                       required=False, help="利用するVOCデータの上限")
argparser.add_argument('--weights_path', type=str,
                       required=False, help="モデルの重みファイルのパス")
args = argparser.parse_args()

n_max = args.max_sample
if n_max and n_max <= 0:
    n_max = None

print(args)

logger.info("use coco dataset.")
# カテゴリはpersonに限る
gen = coco_dataset.Generator(config, args.data_path,
                             data_type='train2017',
                             target_category_names=['person'])
train_data_generator = gen.generate(anchor, n_max=n_max)
labels = gen.get_labels()
gen = coco_dataset.Generator(config, args.data_path,
                             data_type='val2017',
                             target_category_names=['person'])
val_data_generator = gen.generate(anchor, n_max=n_max)
labels = gen.get_labels()

config.n_dataset_labels = len(labels)
config.training = True
config.gpu_count = 1
config.batch_size = 2
config.learning_rate = 0.001
log.out_name_pattern = ".+_loss$"
# print(images[0])

mrcnn = MaskRCNN(anchor.anchors, config)
model = mrcnn.compiled_model()
print(model.summary())

if args.weights_path:
    model.load_weights(args.weights_path, by_name=True)

utils.plot_model(model, './model.png', True, True)

for i, layer in enumerate(model.layers):
    if layer.__class__.__name__ == 'TimeDistributed':
        name = layer.layer.name
        trainable = layer.layer.trainable
    else:
        name = layer.name
        trainable = layer.trainable
    print('layer:', i, ':', name, trainable)

callbacks = [keras.callbacks.TerminateOnNaN(),
             keras.callbacks.TensorBoard(log_dir='./tb_log',
                                         histogram_freq=0,
                                         write_graph=True,
                                         write_images=False),
             keras.callbacks.ModelCheckpoint(filepath='./model/maskrcnn.h5',
                                             verbose=1,
                                             save_weights_only=True,
                                             save_best_only=True),
             keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                               verbose=1,
                                               factor=0.7,
                                               patience=10,
                                               min_lr=config.learning_rate
                                               / 30)]
model.fit_generator(train_data_generator, steps_per_epoch=100, epochs=400,
                    verbose=1,
                    workers=4,
                    max_queue_size=10,
                    use_multiprocessing=True,
                    callbacks=callbacks,
                    validation_data=val_data_generator,
                    validation_steps=20)
model.save_weights('./model/maskrcnn-latest.h5')
