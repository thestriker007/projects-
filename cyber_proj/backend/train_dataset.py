import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

DATASET_PATH = "DSL-StrongPasswordData.csv"
MODELS_DIR = "models"

def train_models():
    print("Loading dataset...")
    df = pd.read_csv(DATASET_PATH)
    
    # Extract feature columns (index 3 to the end)
    feature_cols = df.columns[3:]
    subjects = df['subject'].unique()
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    print(f"Found {len(subjects)} subjects. Training models...")
    
    for count, subject in enumerate(subjects, 1):
        print(f"[{count}/{len(subjects)}] Training model for {subject}...")
        
        # Positive samples (the current user)
        positive_df = df[df['subject'] == subject]
        X_pos = positive_df[feature_cols].values
        y_pos = np.ones(len(X_pos))
        
        # Negative samples (all other users)
        negative_df = df[df['subject'] != subject]
        
        # We can balance the dataset by sampling an equal number of negative samples
        negative_df_sampled = negative_df.sample(n=len(X_pos), random_state=42)
        X_neg = negative_df_sampled[feature_cols].values
        y_neg = np.zeros(len(X_neg))
        
        # Combine
        X_train = np.vstack((X_pos, X_neg))
        y_train = np.concatenate((y_pos, y_neg))
        
        # Train
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        
        # Save
        model_path = os.path.join(MODELS_DIR, f"{subject}_model.pkl")
        joblib.dump(clf, model_path)
        
    print("All models trained and saved to /models !")

if __name__ == "__main__":
    train_models()
