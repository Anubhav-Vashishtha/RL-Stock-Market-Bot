from keras.layers import Dense
from keras.models import Sequential
import tensorflow as tf
from Model.loss import huber_loss

def _build_model(self , model_path):
    model = Sequential()
    model.add(Dense(units=128, activation="relu", input_dim=self.state_size))
    model.add(Dense(units=256, activation="relu"))
    model.add(Dense(units=256, activation="relu"))
    model.add(Dense(units=128, activation="relu"))
    model.add(Dense(units=self.action_size))

    model.compile(loss=huber_loss, optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
    if model_path: 
        model.load_weights(model_path)
        print('Pretrained Weight for this epoche is ' , model_path.split('/')[-1])
     
    return model

