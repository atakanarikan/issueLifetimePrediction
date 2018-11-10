import keras
import keras.backend as K
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


train_df = pd.read_csv("data/riivo/pandasData.csv", sep=",")
print(train_df.shape)
