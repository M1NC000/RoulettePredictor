import os
from tensorflow.keras.models import load_model

def save_model(model, model_name="roulette_model.h5"):
    model.save(model_name)
    print(f"Model saved as {model_name}")

def load_trained_model(model_name="roulette_model.h5"):
    if os.path.exists(model_name):
        model = load_model(model_name)
        print(f"Model {model_name} loaded.")
        return model
    else:
        print(f"Model {model_name} does not exist.")
        return None
