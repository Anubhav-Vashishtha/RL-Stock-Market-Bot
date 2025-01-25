import tensorflow as tf

def huber_loss(self, y_true, y_pred, delta=1.0):
        error = y_true - y_pred
        is_small_error = tf.abs(error) <= delta
        squared_loss = 0.5 * tf.square(error)
        linear_loss = delta * (tf.abs(error) - 0.5 * delta)
        return tf.where(is_small_error, squared_loss, linear_loss)