import glob
import os
import re
import keras
from keras import layers
import numpy as np
from scipy.io.arff import loadarff
import pandas as pd
from sklearn.model_selection import KFold


# noinspection PyMethodMayBeStatic,PyUnresolvedReferences
class IssueCloseTimeClassifier:

    def __init__(self, in_file, time_class):
        self.precisions = []
        self.recalls = []
        self.pfs = []
        self.time_class = int(time_class)
        self.dataframe, self.dataset = self.get_dataframe_and_dataset(in_file)
        self.model = self.get_complied_neural_network_model()

    def get_dataframe_and_dataset(self, filepath):
        data = loadarff(filepath)
        dataframe = pd.DataFrame(data[0]).astype(int)
        dataframe['timeopen'] = dataframe['timeopen'].apply(lambda cls: 0 if cls == 0 else 1)
        return dataframe, data[1].name

    def get_complied_neural_network_model(self):
        model = keras.Sequential()
        model.add(keras.layers.Dense(7, input_dim=7, activation='relu'))
        model.add(keras.layers.Dense(16, activation='relu'))
        model.add(keras.layers.Dense(32, activation='relu'))
        model.add(keras.layers.Dense(16, activation='relu'))
        model.add(keras.layers.Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def get_compiled_cnn_model(self):
        # not using CNNs because:
        # https://www.quora.com/How-can-convolutional-neural-networks-be-used-for-non-image-data
        kernelSize = (1, 7)
        cnn = keras.Sequential()
        cnn.add(layers.Conv2D(filters=7, kernel_size=kernelSize, input_shape=(7, 1, 0), activation='relu'))
        cnn.add(layers.Conv2D(filters=64, kernel_size=kernelSize, activation='relu'))
        cnn.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
        cnn.add(layers.Conv2D(filters=64, kernel_size=kernelSize, activation='relu'))
        cnn.add(layers.Conv2D(filters=64, kernel_size=kernelSize, activation='relu'))
        cnn.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
        cnn.add(layers.Dropout(0.2))
        cnn.add(layers.Flatten())
        cnn.add(layers.Dense(units=128, activation='relu', kernel_initializer='uniform'))
        cnn.add(layers.Dense(units=64, activation='relu', kernel_initializer='uniform'))
        cnn.add(layers.Dense(units=2, activation='softmax', kernel_initializer='uniform'))
        cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return cnn

    def train(self, x, y):
        return self.model.fit(x=x, y=y, epochs=5, batch_size=10, verbose=0)

    def save_to_file(self, filepath):
        self.model.save(filepath)

    def evaluate_and_get_results(self, x, y):
        scores = self.model.evaluate(x, y)
        return "%s: %.2f%%" % (self.model.metrics_names[1], scores[1] * 100)

    def calculate_results(self, x, y):
        y_pred = list(classifier.model.predict_classes(np.array(x.values.tolist())))
        tn, fp, fn, tp = self.get_confusion_matrix(y, y_pred)
        precision = 0 if tp + fp == 0 else tp / (tp + fp)
        recall = 0 if tp + fn == 0 else tp / (tp + fn)
        pf = 0 if tn + fp == 0 else fp / (tn + fp)
        self.precisions.append(precision)
        self.recalls.append(recall)
        self.pfs.append(pf)

    def get_confusion_matrix(self, y_true, y_pred):
        tn, fp, fn, tp = 0, 0, 0, 0
        for real_value, predicted in zip(y_true, y_pred):
            if predicted == 1:
                if real_value == 1:
                    tp += 1
                if real_value == 0:
                    fp += 1
            if predicted == 0:
                if real_value == 1:
                    fn += 1
                if real_value == 0:
                    tn += 1
        print(tn, fp, fn, tp)
        return tn, fp, fn, tp

    def print_results(self):
        print(
            f"{self.dataset}{self.time_class}\t\t\t\t"
            f"{int(sum(self.precisions) * 10)}\t"  # / 10 * 100
            f"{int(sum(self.recalls) * 10)}\t"
            f"{int(sum(self.pfs) * 10)}"
        )

    def do_cross_validation_on_model(self):
        for train_indexes, test_indexes in KFold(n_splits=10).split(self.dataframe):
            train = self.dataframe.iloc[train_indexes, :]
            test = self.dataframe.iloc[test_indexes, :]
            self.train(x=train.iloc[:, :7], y=train['timeopen'])
            print(self.evaluate_and_get_results(test.iloc[:, :7], test['timeopen']))
            self.calculate_results(train.iloc[:, :7], test['timeopen'])
        self.print_results()


def process_all_before_classes_in_given_folder(folder_name):
    folder_path = f"{os.getcwd()}/data/{folder_name}/beforeClass"
    # print(f'filename\t\tprecision\t\trecall\t\tpf')
    for filepath in sorted(glob.glob(os.path.join(folder_path, "*.arff"))):
        filename_without_extension = filepath.split("/")[-1].split('.')[0]
        time_class = re.match('.*?([0-9]+)$', filename_without_extension).group(1)
        classifier = IssueCloseTimeClassifier(in_file=filepath, time_class=time_class)
        classifier.do_cross_validation_on_model()


# process_all_before_classes_in_given_folder('topRepos')
filepath = f"{os.getcwd()}/data/combinedRepos/beforeClass/topbefore90.arff"
filename_without_extension = filepath.split("/")[-1].split('.')[0]
time_class = re.match('.*?([0-9]+)$', filename_without_extension).group(1)
classifier = IssueCloseTimeClassifier(in_file=filepath, time_class=time_class)
classifier.do_cross_validation_on_model()