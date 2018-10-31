import pandas as pd
import arff


df1_selected_fields = [
    'issue_id',
    'commit_before3m',
    'commit_before_project3m',
    'closed_before_project3m',
    'created_before3m',
    'closed_before3m',
    'created_before_project3m',
    'rid'
]

col_mappings = {
    'body_len_strip': 'issuecleanedbodylen',
    'commit_before3m': 'ncommitsbycreator',
    'commit_before_project3m': 'ncommitsinproject',
    'created_before3m': 'nissuesbycreator',
    'closed_before3m': 'nissuesbycreatorclosed',
    'created_before_project3m': 'nissuescreatedinproject',
    'closed_before_project3m': 'nissuescreatedinprojectclosed',
    'timeopen': 'timeopen',
    'rid': 'repo_name',
}

RIIVO_DATA_PATH = '/home/arikan/thesis/code/github-issue-lifetime-prediction-master/data_and_code/issue_data/feature_tables_days_0.csv'  # noqa
FINAL_MENTIONS = '/home/arikan/thesis/code/github-issue-lifetime-prediction-master/data_and_code/issue_data/text/final_mentions.csv'  # noqa
FIXED_ISSUES = '/home/arikan/thesis/code/github-issue-lifetime-prediction-master/data_and_code/issue_data/fixed_issues.csv'  # noqa
FIXED_REPOS = '/home/arikan/thesis/code/github-issue-lifetime-prediction-master/data_and_code/issue_data/fixed_repos.csv'  # noqa


class StudyRiivoData:

    def __init__(self, use_cache=True):
        print("Starting initialization")
        if not use_cache:
            print("Not using cache, loading files.")
            features = pd.read_csv(
                RIIVO_DATA_PATH,
                skipinitialspace=True,
                usecols=df1_selected_fields
            )
            fixed = pd.read_csv(
                FIXED_ISSUES,
                skipinitialspace=True
            )
            extra_features = pd.read_csv(
                FINAL_MENTIONS,
                skipinitialspace=True,
                usecols=['issue_id', 'body_len_strip']
            )
            time_features = self.generate_labeled_files(fixed)

            issues = pd.merge(features, extra_features, how='inner',
                              on='issue_id')
            self.df = pd.merge(issues, time_features, how='inner',
                               on='issue_id')
            self.top_ten_repos_by_count = list(
                fixed.groupby('rid').count().sort_values(
                    ['cnt'], ascending=False
                ).reset_index().head(1000).sample(10)['rid']
            )
            self.df = self.df[self.df.timeopen > 0]
            self.df = self.df.rename(index=str, columns=col_mappings)
            self.df = self.df.astype(int)
            self.df.to_csv("/home/arikan/thesis/code/issue-lifetime-prediction-dl/pandasData.csv")  # noqa
        else:
            print("Using cache.")
            self.df = pd.read_csv("/home/arikan/thesis/code/issue-lifetime-prediction-dl/pandasData.csv")  # noqa
            self.top_ten_repos_by_count = [3520215, 8346195, 11226972, 79007, 1914, 2770186, 7514235, 3360645, 1378984, 8204377]  # noqa
        self.repo_id_to_name_map = self.rid_to_repo_name()
        self.grouped = None
        print("Initialization completed!")

    def rid_to_repo_name(self):
        repos = pd.read_csv(
            FIXED_REPOS,
            skipinitialspace=True,
            usecols=['rid', 'name']
        )
        return {
            row['rid']: row['name']
            for index, row in
            repos[repos.rid.isin(self.top_ten_repos_by_count)].iterrows()
        }

    def assign_num_dates_open(self, row):
        if row['created_at']:
            days_alive = (row['closed_at'] - row['created_at']).days
        else:
            days_alive = 9999
        if days_alive <= 1:
            return 1
        elif days_alive <= 7:
            return 7
        elif days_alive <= 14:
            return 14
        elif days_alive <= 30:
            return 30
        elif days_alive <= 90:
            return 90
        elif days_alive <= 180:
            return 180
        elif days_alive <= 365:
            return 365
        elif days_alive <= 1000:
            return 1000
        else:
            return -1

    def generate_labeled_files(self, df):

        df['created_at'] = pd.to_datetime(df['created_at'])
        df['closed_at'] = pd.to_datetime(df['closed_at'])
        df['timeopen'] = df.apply(
            lambda row: self.assign_num_dates_open(row), axis=1
        ).astype(int)
        return df[['issue_id', 'timeopen']]

    def map_columns_and_write_to_file(self):
        self.df = self.df[
            self.df['repo_name'].isin(self.top_ten_repos_by_count)
        ]
        self.df = self.df.replace({'repo_name': self.repo_id_to_name_map})
        self.df.drop(columns=['issue_id'], inplace=True, axis=1)
        self.grouped = self.df.groupby('repo_name')
        print("Beginning arff export")
        for name, group in self.grouped:
            current = group.reset_index()
            current.drop(
                columns=['index', 'repo_name'],
                inplace=True,
                axis=1
            )
            arff.dump(
                f'randomRepos/{name}.arff',
                current.values,
                relation=name,
                names=current.columns
            )
            print(f"{name}.arff completed!")
        # arff.dump(
        #     f'riivo.arff',
        #     self.df.values,
        #     relation='riivo',
        #     names=self.df.columns
        # )
        print("Finished arff export!")


runner = StudyRiivoData(False)
runner.map_columns_and_write_to_file()
