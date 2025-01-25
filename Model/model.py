from keras.layers import Dense
from keras.models import Sequential

def _build_model(self):
        model = Sequential()
        model.add(Dense(units=128, activation="relu", input_dim=self.state_size))
        model.add(Dense(units=256, activation="relu"))
        model.add(Dense(units=256, activation="relu"))
        model.add(Dense(units=128, activation="relu"))
        model.add(Dense(units=self.action_size))

        model.compile(loss=self.huber_loss, optimizer =\
                       tf.keras.optimizers.Adam(learning_rate=self.learning_rate))

        
        return model