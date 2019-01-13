from keras.models import load_model
import numpy as np
import os
import threading


class IntervalPicker:

    def __init__(self):
        self.limit = 50
        self.models = [
            load_model(f'{os.getcwd()}/out/models/riivo/beforeClass/riivobefore1.h5'),
            load_model(f'{os.getcwd()}/out/models/riivo/beforeClass/riivobefore7.h5'),
            load_model(f'{os.getcwd()}/out/models/riivo/beforeClass/riivobefore14.h5'),
            load_model(f'{os.getcwd()}/out/models/riivo/beforeClass/riivobefore30.h5'),
            load_model(f'{os.getcwd()}/out/models/riivo/beforeClass/riivobefore90.h5')
        ]

    def predict_classes(self, given_issue_values):
        return list(
            map(
                lambda model: model.predict_classes([[given_issue_values]])[0],
                self.models
            )
        )

    def get_random_valid_starting_point(self):
        candidate = np.random.randint(0, 1000, 7)
        while not self.is_prediction_valid(self.predict_classes(candidate)):
            candidate = np.random.randint(0, 1000, 7)
        return [int(nummer) for nummer in candidate]

    def is_prediction_valid(self, predictions):
        for i, prediction in enumerate(predictions):
            if prediction == 0:
                return 1 not in predictions[i:]
        return True

    def generate_intervals(self):
        starting_point = self.get_random_valid_starting_point()
        intervals = [(p, p) for p in starting_point]
        i = 0
        while i < len(intervals):
            is_valid_interval = True
            upper = 0
            lower = 0
            while (upper - lower) < self.limit and is_valid_interval:
                lower, upper = intervals[i]
                intervals[i] = (lower - 5, upper + 5)
                candidates = list(map(lambda tupl: tupl[0] if tupl[0] == tupl[1] else tupl, intervals))
                is_valid_interval = self.are_intervals_valid(candidates)
            i += 1
        print(intervals)
        return intervals

    def are_intervals_valid(self, intervals):
        if all([isinstance(i, int) for i in intervals]):
            return self.is_prediction_valid(intervals)
        for i, elem in enumerate(intervals):
            if isinstance(elem, tuple):
                return all(
                    self.are_intervals_valid(intervals[:i] + [value] + intervals[i + 1:])
                    for value in range(elem[0], elem[1] + 1, 5)
                )


ip = IntervalPicker()

for i in range(10):
    thread = threading.Thread(target=ip.generate_intervals)
    thread.run()
