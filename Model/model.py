import os
from keras.layers import Dense
from keras.models import Sequential
import tensorflow as tf
from Model.loss import huber_loss

def _build_model(self):
    model = Sequential()
    model.add(Dense(units=128, activation="relu", input_dim=self.state_size))
    model.add(Dense(units=64, activation="relu"))
    model.add(Dense(units=32, activation="relu"))
    model.add(Dense(units=self.action_size))

    model.compile(loss=huber_loss, optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        
    weights_dir = os.path.join(os.getcwd(), "Weights")  
    model_files = [f for f in os.listdir(weights_dir) if f.startswith("model_") and f.endswith(".h5")]

    if model_files:
        model_numbers = [int(f.split("_")[1].split(".")[0]) for f in model_files]
        best_model_file = f"model_{max(model_numbers)}.weights.h5"
        best_model_path = os.path.join(weights_dir, best_model_file)

        try:
            model.load_weights(best_model_path)
            print(f"Loaded weights from {best_model_file}")
        except Exception as e:
            print(f"Error loading weights from {best_model_file}: {e}")
    else:
        pass

    return model

