import pandas as pd
import numpy as np


test_df = pd.DataFrame(pd.read_csv('./test.csv'))
train_df = pd.DataFrame(pd.read_csv('./train.csv'))

print(test_df)
print(train_df)
print(len(test_df))
print(len(train_df))
print(test_df.isna().sum())
print(train_df.isna().sum())

