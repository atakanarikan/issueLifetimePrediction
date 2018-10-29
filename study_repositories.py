import os
import matplotlib.pyplot as plt

import pandas as pd
from scipy.io.arff import loadarff
import glob

FIXED_REPOSITORIES = '/home/arikan/thesis/code/github-issue-lifetime-prediction-master/data_and_code/issue_data/fixed_repos.csv'  # noqa
FIXED_ISSUES = '/home/arikan/thesis/code/github-issue-lifetime-prediction-master/data_and_code/issue_data/fixed_issues.csv'  # noqa


class RepositoryStudier:

    def __init__(self, folder_path):
        # self.issues_df = pd.read_csv(FIXED_ISSUES, skipinitialspace=True)
        self.arff_files = self._collect_filenames(folder_path)
        # self.df = self._generate_repository_barplot()
        self._plot_top_and_random_repos_combined_together(
            '/home/arikan/thesis/code/issue-lifetime-prediction-dl/combined_random_10_repos.arff',  # noqa
            '/home/arikan/thesis/code/issue-lifetime-prediction-dl/combined_top_10_repos.arff',  # noqa
        )
        # print(self.df.columns.values)
        # self.number_of_issues_by_close_time = self.calculate_number_of_issues() # noqa

    def top_ten_repos(self):
        return list(
            self.issues_df.groupby(
                'rid'
            ).count().sort_values(
                ['cnt'], ascending=False
            ).reset_index().head(1000).sample(10)['rid']
        )

    def random_ten_repos(self):
        return list(self.issues_df.groupby('rid').count().sample(10)['rid'])

    def _collect_filenames(self, folder_path):
        arff_files = []
        for file in glob.glob(os.path.join(folder_path, "*.arff")):
            arff_files.append(file)
        return arff_files

    def _generate_repository_histogram(self):
        result = {}
        for file in self.arff_files:
            arff_data, dataset = loadarff(file)
            temp_df = pd.DataFrame(arff_data)
            temp_df.timeopen = pd.to_numeric(temp_df.timeopen)
            self._generate_plot_and_save_it_to_out_file(
                dataframe=temp_df,
                dataset_name=dataset.name,
                out_file_path=None
            )
            result[dataset.name] = arff_data
        plt.savefig(
            '/home/arikan/thesis/code/issue-lifetime-prediction-dl'
            '/menziesResults/plots/combinedTogether.png'
        )
        return 1

    def _collect_dataframe_for_combined_files(self, filepath, dataset_name):
        combined = pd.DataFrame(loadarff(filepath)[0])
        combined.timeopen = combined.timeopen.astype(int)
        combined.timeopen = pd.to_numeric(combined.timeopen)
        combined = combined.timeopen.value_counts().sort_index().to_frame()
        return combined.rename(index=str, columns={"timeopen": dataset_name})

    def _plot_top_and_random_repos_combined_together(self, random, top):
        random_repos = self._collect_dataframe_for_combined_files(random, 'random')  # noqa
        top_repos = self._collect_dataframe_for_combined_files(top, 'top')
        combined = pd.concat([random_repos, top_repos], axis=1)
        combined.plot(
            title='Combined Repositories',
            kind='bar',
            secondary_y='top',
            legend=True,
            figsize=(20, 15)
        )
        plt.show()
        plt.savefig(
            '/home/arikan/thesis/code/issue-lifetime-prediction-dl'
            '/menziesResults/plots/combinedTogether.png'
        )

    def _generate_plot_and_save_it_to_out_file(
            self,
            dataframe,
            dataset_name,
            out_file_path
    ):

        dataframe['timeopen'].value_counts().sort_index().plot(
            title=dataset_name,
            kind='bar'
        )
        plt.savefig(out_file_path)


RepositoryStudier('/home/arikan/thesis/code/issue-lifetime-prediction-dl')  # noqa
