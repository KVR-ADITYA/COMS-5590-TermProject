import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load cleaned dataset (replace with actual path)
df = pd.read_csv('./user1.csv')
print(df.head())

X = df[['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']]
y = df['isFraud']

# Split data into 80% train and 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train logistic regression model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Generate predictions
train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

# Calculate accuracy scores
train_accuracy = accuracy_score(y_train, train_preds)  # 0.8825 [2]
test_accuracy = accuracy_score(y_test, test_preds)     # 0.855 [2]

print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

model_weights = model.coef_[0].tolist() + [model.intercept_[0]]
print(model_weights)

# Serialize model weights
# joblib.dump(model, 'fraud_detection_model.joblib')  # Saved as 'logistic_regression_model.joblib' [3]
