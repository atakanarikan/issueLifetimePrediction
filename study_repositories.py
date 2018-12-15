import os
import matplotlib.pyplot as plt

import pandas as pd
from scipy.io.arff import loadarff
import glob

FIXED_ISSUES = '/home/arikan/thesis/code/github-issue-lifetime-prediction-master/data_and_code/issue_data/fixed_issues.csv'  # noqa


class DataLoader:

    def __init__(self, folder_name):
        in_path = os.path.join(os.getcwd(), 'data', folder_name)
        self.out_path = os.path.join(os.getcwd(), 'out', 'plots', folder_name)
        self.arff_files = self.collect_arff_filenames_in_the_folder(in_path)
        self.combined = self.get_combined_dataframe()

    def collect_arff_filenames_in_the_folder(self, folder_path):
        arff_files = []
        for file in glob.glob(os.path.join(folder_path, "*.arff")):
            arff_files.append(file)
        return arff_files

    def get_combined_dataframe(self):
        dataframes = [
            self.collect_dataframe_for_combined_files(filepath)
            for filepath in self.arff_files
        ]
        return pd.concat(dataframes, axis=1, sort=False)

    def get_dataframe_by_dataset(self):
        result = {}
        for filepath in self.arff_files:
            filename_without_extension = filepath.split("/")[-1].split('.')[0]
            result[filename_without_extension] = pd.DataFrame(loadarff(filepath)[0]).astype(int)
        return result

    def collect_dataframe_for_combined_files(self, filepath):
        dataset = loadarff(filepath)
        combined = pd.DataFrame(dataset[0])
        combined.timeopen = combined.timeopen.astype(int)
        combined.timeopen = pd.to_numeric(combined.timeopen)
        combined = combined.timeopen.value_counts().sort_index().to_frame().T
        combined.columns = combined.columns.astype(str)
        existing_columns = []
        if '180' in combined.columns.values:
            combined[' > 90'] = combined['180']
            existing_columns.append('180')
        if '365' in combined.columns.values:
            combined[' > 90'] = combined[' > 90'] + combined['365']
            existing_columns.append('365')
        if '1000' in combined.columns.values:
            combined[' > 90'] = combined[' > 90'] + combined['365']
            existing_columns.append('1000')
        combined = combined.rename(
            index=str,
            columns={
                "1": "< 1",
                "7": "1 - 7",
                "14": "7 - 14",
                "30": "14 - 30",
                "90": "30 - 90",
            }
        )
        combined = combined.drop(existing_columns, axis=1).T
        return combined.rename(index=str, columns={"timeopen": dataset[1].name})


class DataPlotter(DataLoader):

    def plot(self, save=False, stacked=True, title='', prefix=''):
        size = (15, 10) if save else None
        filename = f'{prefix}Stacked.png' if stacked else f'{prefix}.png'
        if not stacked:
            columns = list(self.combined.columns.values)
            columns.pop()
            self.combined.T.plot(
                title=title,
                kind='bar',
                legend=True,
                secondary_y=columns,
                figsize=size
            )
        else:
            self.combined.T.plot(
                title=title,
                kind='bar',
                legend=True,
                stacked=True,
                figsize=size
            )
        if save:
            plt.savefig(os.path.join(self.out_path, filename))
        else:
            plt.show()


class DataStudier(DataLoader):

    def get_training_details_for_datasets(self):
        for dataset, dataframe in self.get_dataframe_by_dataset().items():
            df = dataframe.timeopen.value_counts()
            col1, col2 = dataframe.timeopen.value_counts().sort_index().to_frame().T.columns.values
            print(f"{dataset}\t {df[col1]}\t{df[col2]}")


class RepositorySetSelector:

    def __init__(self):
        self.issues_df = pd.read_csv(FIXED_ISSUES, skipinitialspace=True)

    def _top_ten_repos(self):
        return list(
            self.issues_df.groupby('rid').count().sort_values(
                ['cnt'], ascending=False
            ).reset_index().head(10)['rid']
        )

    def _random_ten_repos(self):
        return list(self.issues_df.groupby('rid').count().sample(10)['rid'])


dp = DataPlotter('riivo')
print(dp.combined)
for column in dp.combined.columns:
    total = dp.combined[column].sum()
    dp.combined[column] = dp.combined[column].apply(lambda val: round((val / total) * 100, 2))
print(dp.combined)
