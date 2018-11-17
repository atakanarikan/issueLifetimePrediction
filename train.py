import glob
import os
import re
import keras
from keras.optimizers import Adam
from keras.utils import to_categorical
from scipy.io.arff import loadarff
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix


class IssueCloseTimeClassifier:

    def __init__(self, in_file, time_class):
        self.time_class = int(time_class)
        self.dataframe, self.dataset = self.get_dataframe_and_dataset(in_file)
        self.model = self.get_complied_neural_network_model()

    def get_dataframe_and_dataset(self, filepath):
        data = loadarff(filepath)
        dataframe = pd.DataFrame(data[0])
        dataframe['timeopen'] = dataframe['timeopen'].apply(
            lambda cls: 0 if cls == 0 else 1
        )
        return dataframe.astype(int), data[1].name

    def get_complied_neural_network_model(self):
        model = keras.Sequential()
        model.add(keras.layers.Dense(7, input_dim=7, activation='relu'))
        model.add(keras.layers.Dense(16, activation='sigmoid'))
        model.add(keras.layers.Dense(16, activation='relu'))
        model.add(keras.layers.Dense(2, activation='sigmoid'))
        model.compile(
            loss='binary_crossentropy',
            optimizer=Adam(lr=0.00005),
            metrics=['accuracy']
        )
        return model

    def train(self, X, Y):
        self.model.fit(x=X, y=Y, epochs=10, batch_size=10, verbose=0)

    def save_to_file(self, filepath):
        self.model.save(filepath)

    def evaluate_and_get_results(self, X, Y):
        scores = self.model.evaluate(X, Y)
        return "%s: %.2f%%" % (self.model.metrics_names[1], scores[1] * 100)

    def get_results(self, X, Y):
        y_predicted = [
            0 if a[0] > a[1] else 1 for a in self.model.predict(X)
        ]
        y_true = [
            0 if a[0] > a[1] else 1 for a in Y
        ]
        tn, fp, fn, tp = confusion_matrix(y_true, y_predicted, labels=[0, 1]).ravel()
        precision = 0 if tp + fp == 0 else tp / (tp + fp)
        recall = 0 if tp + fn == 0 else tp / (tp + fn)
        pf = 0 if tn + fp == 0 else fp / (tn + fp)
        return precision, recall, pf

    def do_cross_validation_on_model(self):
        precisions = []
        recalls = []
        pfs = []
        for train_indexes, test_indexes in KFold(n_splits=10).split(self.dataframe):
            train = self.dataframe.iloc[train_indexes, :]
            test = self.dataframe.iloc[test_indexes, :]
            self.train(
                train.iloc[:, :7],
                to_categorical(np.array(train['timeopen']))
            )
            precision, recall, pf = self.get_results(
                test.iloc[:, :7],
                to_categorical(np.array(test['timeopen']))
            )
            precisions.append(precision)
            recalls.append(recall)
            pfs.append(pf)
        print(
            f"{self.dataset}{self.time_class}\t\t"
            f"{int(sum(precisions)/len(precisions) * 100)}\t\t"
            f"{int(sum(recalls)/len(recalls) * 100)}\t\t"
            f"{int(sum(pfs)/len(pfs) * 100)}"
        )


def process_all_before_classes_in_given_folder(folder_name):
    folder_path = f"{os.getcwd()}/data/{folder_name}/beforeClass"
    # print(f'filename\t\tprecision\t\trecall\t\tpf')
    for filepath in sorted(glob.glob(os.path.join(folder_path, "*.arff"))):
        filename_without_extension = filepath.split("/")[-1].split('.')[0]
        time_class = re.match('.*?([0-9]+)$', filename_without_extension).group(1)
        classifier = IssueCloseTimeClassifier(in_file=filepath, time_class=time_class)
        classifier.do_cross_validation_on_model()


process_all_before_classes_in_given_folder('randomRepos')
