import tensorflow as tf

def huber_loss(states, targets, delta=1.0):
        error = states - targets
        is_small_error = tf.abs(error) <= delta
        squared_loss = 0.5 * tf.square(error)
        linear_loss = delta * (tf.abs(error) - 0.5 * delta)
        return tf.where(is_small_error, squared_loss, linear_loss)