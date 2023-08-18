import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
import joblib


# Get the dataset
digits = datasets.load_digits()
digits_df = pd.DataFrame(digits.data)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(digits_df.iloc[:, :-1], digits_df.iloc[:, -1], test_size=0.2)

# Create and train Random Forest model
model = RandomForestClassifier(n_estimators=100, max_depth=10)
model.fit(X_train, y_train)

# Perform validation
y_pred = model.predict(X_test)
prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, pos_label=1, average='weighted', zero_division=1)

print(f'f1: {prec:.2f}')
print(f'prec: {prec:.2f}')
print(f'rec: {rec:.2f}')

# Store the model
joblib.dump(model, 'model.joblib')
print('Model saved.')
