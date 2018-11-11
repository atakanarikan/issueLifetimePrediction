import glob
import os

import keras
from keras.optimizers import Adam
from scipy.io.arff import loadarff
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


class IssueCloseTimeClassifier:

    def __init__(self, in_file):
        self.output_layer_dim = 2
        self.dataframe, self.dataset = self.get_dataframe_and_dataset(in_file)
        self.trainX, self.testX, self.trainY, self.testY = (
            self.get_preprocessed_data()
        )
        self.model = self.get_complied_neural_network_model()

    def get_dataframe_and_dataset(self, filepath):
        data = loadarff(filepath)
        dataframe = pd.DataFrame(data[0]).astype(int)
        return dataframe, data[1].name

    def get_preprocessed_data(self):
        X = self.dataframe.iloc[:, :7]  # features
        Y = self.dataframe.iloc[:, -1]  # timeopen
        # Y = Y.map(lambda e: e if e < 90 else 90)
        if len(set(Y)) == 1:
            self.output_layer_dim = 1
        Y = OneHotEncoder(categories='auto').fit_transform(Y.values.reshape(-1, 1))
        trainX, testX = train_test_split(X, random_state=123, test_size=0.1)
        trainY, testY = train_test_split(Y, random_state=123, test_size=0.1)
        return trainX, testX, trainY, testY

    def get_complied_neural_network_model(self):
        model = keras.Sequential()
        model.add(keras.layers.Dense(7, input_dim=7, activation='relu'))
        model.add(keras.layers.Dense(16, activation='sigmoid'))
        model.add(keras.layers.Dense(16, activation='relu'))
        model.add(keras.layers.Dense(self.output_layer_dim, activation='sigmoid'))
        model.compile(
            loss='binary_crossentropy',
            optimizer=Adam(lr=0.00005),
            metrics=['accuracy']
        )
        return model

    def train(self):
        self.model.fit(
            x=self.trainX,
            y=self.trainY,
            epochs=10,
            batch_size=10,
            verbose=0
        )

    def save_to_file(self, filepath):
        self.model.save(filepath)

    def evaluate_and_get_results(self):
        scores = self.model.evaluate(self.trainX, self.trainY)
        return "%s: %.2f%%" % (self.model.metrics_names[1], scores[1] * 100)


def process_all_before_classes_in_given_folder(folder_name):
    project_prefix = '/home/arikan/thesis/code/issue-lifetime-prediction-dl'
    folder_path = f"{project_prefix}/data/{folder_name}/beforeClass"
    for filepath in glob.glob(os.path.join(folder_path, "*.arff")):
        filename_without_extension = filepath.split("/")[-1].split('.')[0]
        out_file = (
            f"{project_prefix}/out/models/{folder_name}/beforeClass/{filename_without_extension}.h5"
        )
        classifier = IssueCloseTimeClassifier(in_file=filepath)
        print(f"Training a model for: {filename_without_extension}")
        classifier.train()
        classifier.save_to_file(out_file)
        print(f"Saved the trained model! {classifier.evaluate_and_get_results()}")
        print("========================================")


process_all_before_classes_in_given_folder('riivo')


# classifier = IssueCloseTimeClassifier('/home/arikan/thesis/code/issue-lifetime-prediction-dl/data/randomRepos/beforeClass/fuzzybefore90.arff')  # noqa
# classifier.train()
# classifier.evaluate_and_get_results()
