import pandas as pd


print(
    pd.read_csv(
        "out/averages/round_robin/random_repos/before90.csv"
    ).describe()
)
