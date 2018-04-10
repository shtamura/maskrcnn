import keras.layers as KL


class BatchNorm(KL.BatchNormalization):
    # https://github.com/matterport/Mask_RCNN/
    # より。
    """Batch Normalization class. Subclasses the Keras BN class and
    hardcodes training=False so the BN layer doesn't update
    during training.

    Batch normalization has a negative effect on training if batches are small
    so we disable it here.
    """

    def call(self, inputs, training=None):
        return super(self.__class__, self).call(inputs, training=False)
