import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

print("Loading NASA C-MAPSS dataset...")
data_path = os.path.join(os.path.dirname(__file__), '..', 'archive', 'train_FD001.txt')
columns = ['unit', 'cycle', 'op_set1', 'op_set2', 'op_set3'] + [f'sensor_{i}' for i in range(1, 22)]

# Load standard C-MAPSS text format separated by spaces
df = pd.read_csv(data_path, sep=r'\s+', header=None, names=columns)

# Calculate true RUL based on cycle end per unit
max_cycles = df.groupby('unit')['cycle'].max()
df['RUL'] = df.apply(lambda row: max_cycles[row['unit']] - row['cycle'], axis=1)

features = ['op_set1', 'op_set2', 'op_set3'] + [f'sensor_{i}' for i in range(1, 22)]
X = df[features]
y = df['RUL']

print("Training Universal Machine Prediction Model...")
# Use a robust Random Forest that ignores uninformative constant sensors naturally
model = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1))
])

model.fit(X, y)

model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(model, f)

print(f"Model trained on real dataset and saved to {model_path}")
