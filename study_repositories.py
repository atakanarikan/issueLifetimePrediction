import pandas as pd

FIXED_ISSUES = '/home/arikan/thesis/code/github-issue-lifetime-prediction-master/data_and_code/issue_data/fixed_issues.csv'  # noqa
FIXED_REPOSITORIES = '/home/arikan/thesis/code/github-issue-lifetime-prediction-master/data_and_code/issue_data/fixed_repos.csv'  # noqa


issues = pd.read_csv(
    FIXED_ISSUES,
    skipinitialspace=True,
)


def top_ten_repos_with_most_issues():
    return list(
        issues.groupby(
            'rid'
        ).count().sort_values(
            ['cnt'], ascending=False
        ).reset_index().head(1000).sample(10)['rid']
    )
