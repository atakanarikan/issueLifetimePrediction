from keras.models import load_model
import numpy as np
import os


class IntervalPicker:

    def __init__(self):
        self.step_size = 10
        self.models = [
            load_model(f'{os.getcwd()}/out/models/riivo/beforeClass/riivobefore1.h5'),
            load_model(f'{os.getcwd()}/out/models/riivo/beforeClass/riivobefore7.h5'),
            load_model(f'{os.getcwd()}/out/models/riivo/beforeClass/riivobefore14.h5'),
            load_model(f'{os.getcwd()}/out/models/riivo/beforeClass/riivobefore30.h5'),
            load_model(f'{os.getcwd()}/out/models/riivo/beforeClass/riivobefore90.h5')
        ]

    def predict_classes(self, given_issue_values):
        return map(
            lambda model: model.predict_classes([[given_issue_values]])[0],
            self.models
        )

    def get_random_valid_starting_point(self):
        candidate = np.random.randint(0, 10000, 7)
        while not self.is_prediction_valid(candidate):
            candidate = np.random.randint(0, 10000, 7)
        return candidate

    def is_prediction_valid(self, predictions):
        for i, prediction in enumerate(predictions):
            if prediction == 0:
                return 1 not in predictions[i:]
        return True

    def generate_intervals(self):
        starting_point = self.get_random_valid_starting_point()
        current = starting_point
        while self.is_prediction_valid(current):