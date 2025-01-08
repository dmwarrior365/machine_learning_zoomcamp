import joblib
import numpy as np

def preprocess_input(data):
    """
    Preprocess input data before making predictions
    Args:
        data: Input data to preprocess
    Returns:
        Preprocessed data ready for model prediction
    """
    
    return np.array(data)

def postprocess_output(prediction):
    """
    Postprocess model predictions before returning to client
    Args:
        prediction: Raw model prediction
    Returns:
        Processed prediction ready to return to client
    """
    # Add your postprocessing steps here
    return prediction.tolist()