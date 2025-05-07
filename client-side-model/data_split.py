from IPython.display import display
import pandas as pd
from pandasgui import show
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('../data/Synthetic_Financial_datasets_log.csv')

show(df.head())

# Clean the data
df = df.drop_duplicates()
df = df.dropna()
# df = df.drop('isFlaggedFraud')

# # Split the data into two parts (e.g., 50% train, 50% test)
# train_df, test_df = train_test_split(df, test_size=0.7, random_state=42)

# def split_dataframe(df, n):
#     chunk_size = len(df) // n
#     return [df.iloc[i * chunk_size : (i + 1) * chunk_size] for i in range(n)]

# # Split train and test into 5 parts each
# train_parts = split_dataframe(train_df, 5)
# test_parts = split_dataframe(test_df, 5)

# for i, part in enumerate(train_parts, start=1):
#     part.to_csv(f'user1_dataset{i}.csv', index=False)

# for i, part in enumerate(test_parts, start=1):
#     part.to_csv(f'user2_dataset{i}.csv', index=False)
