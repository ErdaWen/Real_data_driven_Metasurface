import numpy as np
import tensorflow as tf

def masked_mse(y_true,y_pred):
    ptrn_masked = tf.cast(y_true,tf.float32)
    ptrn_pred = tf.cast(y_pred,tf.float32)
    mask = tf.math.logical_not(tf.math.is_nan(ptrn_masked))
    mask_float = tf.cast(mask,tf.float32)
    ptrn_masked = tf.where(tf.math.is_nan(ptrn_masked), -10*tf.ones_like(ptrn_masked), ptrn_masked)
    sq_errors = tf.math.square(ptrn_masked - ptrn_pred)
    # sq_errors = c
    sq_errors_masked = tf.math.multiply_no_nan(sq_errors,mask_float)
    mse_masked = tf.math.reduce_sum(sq_errors_masked, -1)/tf.math.reduce_sum(mask_float,-1)
    # mse_masked = tf.where(tf.math.is_nan(mse_masked), 10*tf.ones_like(mse_masked), mse_masked)
    return mse_masked   