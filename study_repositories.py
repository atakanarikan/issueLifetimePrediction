import os
import matplotlib.pyplot as plt

import pandas as pd
from scipy.io.arff import loadarff
import glob

FIXED_ISSUES = '/home/arikan/thesis/code/github-issue-lifetime-prediction-master/data_and_code/issue_data/fixed_issues.csv'  # noqa


class DataPlotter:
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

    def collect_dataframe_for_combined_files(self, filepath):
        dataset = loadarff(filepath)
        combined = pd.DataFrame(dataset[0])
        combined.timeopen = combined.timeopen.astype(int)
        combined.timeopen = pd.to_numeric(combined.timeopen)
        combined = combined.timeopen.value_counts().sort_index().to_frame().T
        combined.columns = combined.columns.astype(str)
        combined['90'] = (
            combined['90'] + combined['180'] + combined['365'] + combined['1000']  # noqa
        )
        combined = combined.drop(['180', '365', '1000'], axis=1).T
        return combined.rename(index=str, columns={"timeopen": dataset[1].name})

    def plot(self, save=False, stacked=True, title=''):
        size = (15, 10) if save else None
        filename = 'percentageStacked.png' if stacked else 'percentage.png'
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


# DataPlotter('combined').plot(False, False, title='Combined All Repositories')
dp = DataPlotter('combinedRepos')
print(dp.combined)
for column in dp.combined.columns:
    total = dp.combined[column].sum()
    dp.combined[column] = dp.combined[column].apply(lambda val: round((val / total) * 100, 2))
print(dp.combined)
dp.plot(True, False, title='Percentage of issue classes')
