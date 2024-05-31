import logging

def setup_logging(log_file="roulette_predictor.log"):
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='%(asctime)s:%(levelname)s:%(message)s')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    logging.info("Logging setup complete.")

def log_prediction(predictions, actual_number):
    logging.info(f"Predicted numbers: {predictions}")
    logging.info(f"Actual number: {actual_number}")
    if actual_number not in predictions:
        logging.warning(f"Prediction error. Actual number {actual_number} was not in the predicted numbers.")
    else:
        logging.info(f"Successful prediction. Actual number {actual_number} was in the predicted numbers.")