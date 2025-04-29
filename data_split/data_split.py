import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('./Synthetic_Financial_datasets_log.csv')

print(df.head())

# Clean the data
df = df.drop_duplicates()
df = df.dropna()
# df = df.drop('isFlaggedFraud')

# Split the data into two parts (e.g., 50% train, 50% test)
train_df, test_df = train_test_split(df, test_size=0.7, random_state=42)

train_df.to_csv('./user1.csv')
test_df.to_csv('./user2.csv')
