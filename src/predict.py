import pickle
import numpy as np

def load_model(filename):
    """
    Loads a saved model from a pickle file.
    """
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    return model

def make_predictions(model, input_data):
    """
    Uses the model to predict classes for the given input data.
    
    input_data: A 2D numpy array (e.g., [[5.1, 3.5, 1.4, 0.2]])
    """
    return model.predict(input_data)
