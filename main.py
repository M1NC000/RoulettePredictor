import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Input
from sklearn.model_selection import train_test_split
from data_processing import normalize_data, denormalize_data, create_sequences
from model_utils import save_model, load_trained_model
from logging_utils import setup_logging, log_prediction
import logging

class RoulettePredictor:
    def __init__(self, look_back=5):
        self.look_back = look_back
        self.model = self.load_or_build_model()
        self.data = []
        self.data_min = None
        self.data_max = None

    def load_or_build_model(self):
        model = load_trained_model()
        if model is None:
            model = self.build_model()
        return model

    def build_model(self):
        model = Sequential()
        model.add(Input(shape=(self.look_back, 1)))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model

    def update_model(self, new_data):
        self.data.extend(new_data)
        if len(self.data) > self.look_back:
            normalized_data, self.data_min, self.data_max = normalize_data(np.array(self.data))
            X, y = create_sequences(normalized_data, self.look_back)
            X = np.reshape(X, (X.shape[0], X.shape[1], 1))
            self.model.fit(X, y, epochs=10, batch_size=1, verbose=0)
            save_model(self.model)

    def predict_next_numbers(self, num_predictions=6):
        if len(self.data) < self.look_back:
            raise ValueError("Nedostatočný počet údajov na predikciu.")
        normalized_data, _, _ = normalize_data(np.array(self.data), (self.data_min, self.data_max))
        last_sequence = np.array(normalized_data[-self.look_back:]).reshape(1, self.look_back, 1)
        predictions = []
        current_sequence = last_sequence
        while len(predictions) < num_predictions:
            prediction = self.model.predict(current_sequence)
            next_number_normalized = prediction[0][0]
            next_number = int(denormalize_data(next_number_normalized, self.data_min, self.data_max))
            if next_number not in predictions and 0 <= next_number <= 36:
                predictions.append(next_number)
            current_sequence = np.append(current_sequence[0][1:], [[next_number_normalized]], axis=0)
            current_sequence = np.reshape(current_sequence, (1, self.look_back, 1))
        return predictions

def main():
    setup_logging()
    predictor = RoulettePredictor()
    print("Začiatok programu. Zadajte posledných 20 čísiel, ktoré padli.")
    logging.info("Začiatok programu. Zadajte posledných 20 čísiel, ktoré padli.")

    while len(predictor.data) < 20:
        new_number = int(input(f"Zadajte číslo {len(predictor.data) + 1} z posledných 20 čísiel, ktoré padli: "))
        predictor.data.append(new_number)

    while True:
        predictor.update_model(predictor.data)
        predicted_numbers = predictor.predict_next_numbers(num_predictions=6)
        print(f"Predicted next 6 numbers: {predicted_numbers}")
        logging.info(f"Predicted next 6 numbers: {predicted_numbers}")

        new_number = int(input("Zadajte posledné číslo, ktoré padlo: "))
        predictor.update_model([new_number])
        log_prediction(predicted_numbers, new_number)

if __name__ == "__main__":
    main()
