"""
Model Training Script
Trains all 6 models and saves them as .pkl files in the model/ folder
Run this once to generate trained model files
"""

import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# Get the directory where this script is located
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

print("="*70)
print("MODEL TRAINING AND SERIALIZATION")
print("="*70)

# Load data
print("\n[1/4] Loading dataset...")
try:
    df = pd.read_csv('data/data.csv')
    # Drop any unnamed columns (index artifacts from CSV export)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    print(f"✓ Dataset loaded: {len(df)} samples, {len(df.columns)} columns")
except Exception as e:
    print(f"✗ ERROR: Could not load data/data.csv - {e}")
    exit(1)

# Prepare data
print("\n[2/4] Preparing data...")
df_processed = df.copy()
# Encode target variable (M=Malignant=1, B=Benign=0)
df_processed['diagnosis'] = (df_processed['diagnosis'] == 'M').astype(int)

# Drop ID column and separate features and target
X = df_processed.drop(['id', 'diagnosis'], axis=1)
y = df_processed['diagnosis']
print(f"✓ Features: {X.shape[1]}, Target distribution: {y.value_counts().to_dict()}")
# Split and scale data
print("\n[3/4] Splitting and scaling data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(f"✓ Train: {len(X_train)}, Test: {len(X_test)}")
print("✓ Features scaled using StandardScaler")

# Save scaler
print("\n[4/4] Training and saving models...\n")
scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"  ✓ Scaler saved to: scaler.pkl")

# Define models
models = {
    'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
    'decision_tree': DecisionTreeClassifier(random_state=42),
    'knn': KNeighborsClassifier(n_neighbors=5),
    'naive_bayes': GaussianNB(),
    'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'xgboost': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
}

# Train and save each model
for idx, (model_name, model) in enumerate(models.items(), 1):
    print(f"  [{idx}/{len(models)}] Training {model_name}...", end=" ")
    
    try:
        # Train with appropriate data
        if model_name in ['logistic_regression', 'knn']:
            model.fit(X_train_scaled, y_train)
        else:
            model.fit(X_train, y_train)
        
        # Save model
        model_path = os.path.join(MODEL_DIR, f'{model_name}.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        print(f"✓ Saved")
        
    except Exception as e:
        print(f"✗ ERROR: {str(e)}")

# Save feature column order
feature_columns_path = os.path.join(MODEL_DIR, 'feature_columns.pkl')
with open(feature_columns_path, 'wb') as f:
    pickle.dump(X.columns.tolist(), f)
print(f"  ✓ Feature columns saved to: feature_columns.pkl") 
print("\n" + "="*70)
print("✓ ALL MODELS TRAINED AND SAVED SUCCESSFULLY!")
print("="*70)
print("\nGenerated files in model/ folder:")
print("  - scaler.pkl")
print("  - logistic_regression.pkl")
print("  - decision_tree.pkl")
print("  - knn.pkl")
print("  - naive_bayes.pkl")
print("  - random_forest.pkl")
print("  - xgboost.pkl")
print("\n" + "="*70)