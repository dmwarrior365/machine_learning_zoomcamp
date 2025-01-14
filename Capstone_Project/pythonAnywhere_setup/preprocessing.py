import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class DataPreprocessor:
    def __init__(self):
        # Load saved label encoders
        try:
            self.label_encoders = joblib.load('label_encoders.pkl')
        except:
            print("Warning: Label encoders not found. Make sure to save them during training.")
            self.label_encoders = {}

        self.categorical_columns = [
            'make', 'model', 'vehicle_class',
            'transmission', 'fuel_type'
        ]

        self.numeric_columns = [
            'engine_size(l)', 'cylinders',
            'fuel_efficiency'
        ]

    def preprocess_input(self, data):
        """
        Preprocess input data using saved label encoders
        Args:
            data: Dictionary or list containing input data
        Returns:
            Preprocessed numpy array ready for prediction
        """
        try:
            # Convert input to DataFrame if it's a dictionary
            if isinstance(data, dict):
                df = pd.DataFrame([data])
            else:
                df = pd.DataFrame([data], columns=self.categorical_columns + self.numeric_columns)

            # Filter out records with fuel_type "N" if present
            df = df[df['fuel_type'] != "N"]

            # Encode categorical variables
            for column in self.categorical_columns:
                if column in self.label_encoders:
                    le = self.label_encoders[column]
                    try:
                        df[column] = le.transform(df[column])
                    except ValueError as e:
                        print(f"Warning: Unknown category in {column}. Using default value.")
                        # Handle unknown categories by using a default value
                        df[column] = 0

            # Ensure numeric columns are float
            for column in self.numeric_columns:
                df[column] = pd.to_numeric(df[column], errors='coerce')

            return df.values

        except Exception as e:
            raise ValueError(f"Error preprocessing input data: {str(e)}")