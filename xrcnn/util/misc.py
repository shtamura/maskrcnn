import tensorflow as tf


def stack_each_number(max_num, stack_count):
    """
        0からmax_num-1までの整数を数字毎にstack_count回積み上げたリストを得る。
        例:
            max_num=3
            stack_count=2
            return=[0,0,1,1,2,2]
    """
    v = tf.range(max_num)
    v = tf.reshape(v, [-1, 1])
    v = tf.tile(v, [1, stack_count])
    v = tf.reshape(v, [-1])
    return v
