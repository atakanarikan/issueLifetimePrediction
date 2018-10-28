import pandas as pd


print(
    pd.read_csv(
        "menziesResults/averages/round_robin/random_repos/before90.csv"
    ).describe()
)
