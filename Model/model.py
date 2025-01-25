from keras.layers import Dense
from keras.models import Sequential
import tensorflow as tf
from Model.loss import huber_loss

def _build_model(self):
        model = Sequential()
        model.add(Dense(units=128, activation="relu", input_dim=self.state_size))
        model.add(Dense(units=256, activation="relu"))
        model.add(Dense(units=256, activation="relu"))
        model.add(Dense(units=128, activation="relu"))
        model.add(Dense(units=self.action_size))

        model.compile(loss=huber_loss, optimizer =\
                       tf.keras.optimizers.Adam(learning_rate=self.learning_rate))

        
        return model