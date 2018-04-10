import re
import tensorflow as tf

stop = False
out_name_pattern = ".*"


def tfprint(tensor, prefix=None, summarize=256):
    """tf.Printのショートカットメソッド
        Return:
            tf.Print(tensor, [tensor],・・・・)
            の復帰値をそのまま返す。
    """
    if stop:
        # stopならprint無し
        return tensor

    if prefix is None:
        prefix = tensor.name

    if not re.match(out_name_pattern, prefix):
        return tensor

    return tf.Print(tensor, [tensor], prefix + ": ",
                    summarize=summarize)
