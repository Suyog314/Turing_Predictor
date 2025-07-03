
# 01webapp/model_runner.py

import sys
import os

# Make sure we can import from predict/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from predict.predict_params import predict_parameters

def run_prediction(image_path):
    return predict_parameters(image_path)
