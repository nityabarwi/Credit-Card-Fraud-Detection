import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_csv('creditcard.csv')
X = df.drop(['Class', 'Time'], axis=1)  # Drop 'Time' if it was not used in training
y = df['Class']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_sm, y_train_sm)

# Save the model using joblib
joblib.dump(model, 'model.pkl')
