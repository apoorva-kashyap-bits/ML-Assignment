import streamlit as st
import pandas as pd
import numpy as np
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
from sklearn.metrics import (accuracy_score, auc, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report,
                             roc_auc_score, matthews_corrcoef)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Cancer Classification System", layout="wide")

# App title
st.title("🔬 Breast Cancer Classification System")
st.markdown("**ML Assignment 2** - Binary Classification with 6 ML Models")

# Sidebar navigation
page = st.sidebar.radio("Navigation", 
                        ["📊 Overview", "🤖 Model Training", "📈 Model Comparison", "🔮 Predictions"])

# =====================================================
# HELPER FUNCTIONS
# =====================================================

@st.cache_data
def load_data():
    """Load the breast cancer dataset"""
    try:
        df = pd.read_csv('data/data.csv')
        # Drop any unnamed columns (index artifacts from CSV export)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        return df
    except:
        st.error("Could not load data/data.csv")
        return None

def prepare_data(df):
    """Prepare data for modeling"""
    # Encode target variable (M=Malignant=1, B=Benign=0)
    df_processed = df.copy()
    df_processed['diagnosis'] = (df_processed['diagnosis'] == 'M').astype(int)
    
    # Drop ID column and separate features and target
    X = df_processed.drop(['id', 'diagnosis'], axis=1)
    y = df_processed['diagnosis']
    
    return X, y, df_processed

@st.cache_resource
def load_pretrained_models():
    """Load all pre-trained models from the model/ folder"""
    model_dir = 'model'
    models = {}
    
    model_files = {
        'Logistic Regression': 'logistic_regression.pkl',
        'Decision Tree': 'decision_tree.pkl',
        'K-Nearest Neighbor': 'knn.pkl',
        'Naive Bayes': 'naive_bayes.pkl',
        'Random Forest': 'random_forest.pkl',
        'XGBoost': 'xgboost.pkl'
    }
    
    scaler = None
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    
    try:
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
    except Exception as e:
        st.warning(f"Could not load scaler: {e}")
    
    for model_name, filename in model_files.items():
        filepath = os.path.join(model_dir, filename)
        try:
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    models[model_name] = pickle.load(f)
            else:
                st.warning(f"Model file not found: {filepath}")
        except Exception as e:
            st.warning(f"Error loading {model_name}: {e}")
    
    return models, scaler

def evaluate_pretrained_models(models, scaler, X_test, y_test):
    """Evaluate all pre-trained models"""
    results = {}
    
    for model_name, model in models.items():
        try:
            # Use scaled data for models that need it
            if model_name in ['Logistic Regression', 'K-Nearest Neighbor']:
                if scaler is not None:
                    X_test_scaled = scaler.transform(X_test)
                    y_pred = model.predict(X_test_scaled)
                    y_proba = model.predict_proba(X_test_scaled)[:, 1]
                else:
                    y_pred = model.predict(X_test)
                    y_proba = model.predict_proba(X_test)[:, 1]
            else:
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1]
            
            results[model_name] = {
                'model': model,
                'predictions': y_pred,
                'probabilities': y_proba
            }
        except Exception as e:
            st.error(f"Error evaluating {model_name}: {e}")
    
    return results

def calculate_metrics(y_true, y_pred, y_proba):
    """Calculate all required metrics"""
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'AUC': roc_auc_score(y_true, y_proba),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1': f1_score(y_true, y_pred),
        'MCC': matthews_corrcoef(y_true, y_pred)
    }
    return metrics

def train_models(X_train, X_test, y_train, y_test):
    """Train all 6 models and return results (for comparison/testing)"""
    
    results = {}
    
    # Scale features for models that need it
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 1. Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)
    y_pred_proba_lr = lr.predict_proba(X_test_scaled)[:, 1]
    
    results['Logistic Regression'] = {
        'model': lr,
        'predictions': y_pred_lr,
        'probabilities': y_pred_proba_lr,
        'scaler': scaler
    }
    
    # 2. Decision Tree Classifier
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    y_pred_proba_dt = dt.predict_proba(X_test)[:, 1]
    
    results['Decision Tree'] = {
        'model': dt,
        'predictions': y_pred_dt,
        'probabilities': y_pred_proba_dt,
        'scaler': None
    }
    
    # 3. K-Nearest Neighbor
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)
    y_pred_knn = knn.predict(X_test_scaled)
    y_pred_proba_knn = knn.predict_proba(X_test_scaled)[:, 1]
    
    results['K-Nearest Neighbor'] = {
        'model': knn,
        'predictions': y_pred_knn,
        'probabilities': y_pred_proba_knn,
        'scaler': scaler
    }
    
    # 4. Naive Bayes (Gaussian)
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_pred_nb = nb.predict(X_test)
    y_pred_proba_nb = nb.predict_proba(X_test)[:, 1]
    
    results['Naive Bayes'] = {
        'model': nb,
        'predictions': y_pred_nb,
        'probabilities': y_pred_proba_nb,
        'scaler': None
    }
    
    # 5. Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_pred_proba_rf = rf.predict_proba(X_test)[:, 1]
    
    results['Random Forest'] = {
        'model': rf,
        'predictions': y_pred_rf,
        'probabilities': y_pred_proba_rf,
        'scaler': None
    }
    
    # 6. XGBoost
    xgb = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    y_pred_proba_xgb = xgb.predict_proba(X_test)[:, 1]
    
    results['XGBoost'] = {
        'model': xgb,
        'predictions': y_pred_xgb,
        'probabilities': y_pred_proba_xgb,
        'scaler': None
    }
    
    return results, y_test

# =====================================================
# PAGE 1: OVERVIEW
# =====================================================

if page == "📊 Overview":
    st.header("Dataset Overview")
    
    df = load_data()
    
    if df is not None:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Samples", len(df))
        with col2:
            st.metric("Total Features", len(df.columns) - 2)  # Exclude ID and diagnosis
        with col3:
            st.metric("Target Classes", df['diagnosis'].nunique())
        with col4:
            # Count total missing values across all columns
            missing_count = df.isnull().sum().sum()
            st.metric("Missing Values", int(missing_count) if missing_count > 0 else 0)
        
        st.subheader("Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.subheader("Dataset Statistics")
        st.dataframe(df.describe(), use_container_width=True)
        
        # Target distribution
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Target Variable Distribution")
            target_counts = df['diagnosis'].value_counts()
            target_labels = {'M': 'Malignant', 'B': 'Benign'}
            target_counts.index = target_counts.index.map(target_labels)
            fig, ax = plt.subplots(figsize=(6, 4))
            colors = ['#FF6B6B', '#4ECDC4']
            target_counts.plot(kind='bar', ax=ax, color=colors)
            ax.set_xlabel('Cancer Diagnosis')
            ax.set_ylabel('Count')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
            st.pyplot(fig)
        
        with col2:
            st.subheader("Class Distribution (%)")
            # Get value counts and exclude any NaN values
            target_counts_dist = df['diagnosis'].value_counts(dropna=True)
            percentages = (target_counts_dist / target_counts_dist.sum()) * 100
            target_labels_percent = {'M': 'Malignant', 'B': 'Benign'}
            percentages.index = percentages.index.map(target_labels_percent)
            fig, ax = plt.subplots(figsize=(6, 4))
            colors = ['#FF6B6B', '#4ECDC4']
            ax.pie(percentages, labels=percentages.index, autopct='%1.1f%%', colors=colors)
            ax.set_ylabel('')
            st.pyplot(fig)

# =====================================================
# PAGE 2: MODEL TRAINING
# =====================================================

elif page == "🤖 Model Training":
    st.header("Model Training & Evaluation")
    
    df = load_data()
    
    if df is not None:
        # Data preparation
        X, y, df_processed = prepare_data(df)
        
        st.subheader("Data Preparation")
        col1, col2 = st.columns(2)
        with col1:
            test_size = st.slider("Test Set Size (%)", 10, 40, 20) / 100
        with col2:
            st.write(f"Training samples: {int(len(df) * (1 - test_size))}")
            st.write(f"Test samples: {int(len(df) * test_size)}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Training options
        st.subheader("Load Pre-trained Models")
        st.write("The models have been pre-trained and saved. Click below to load and evaluate them:")
        
        if st.button("📥 Load Pre-trained Models & Evaluate", use_container_width=True):
            st.subheader("Loading Pre-trained Models...")
            
            try:
                # Load pre-trained models
                pretrained_models, scaler = load_pretrained_models()
                
                if not pretrained_models:
                    st.error("No pre-trained models found in model/ folder. Please run: python model/train.py")
                else:
                    st.success(f"✓ Loaded {len(pretrained_models)} pre-trained models!")
                    
                    # Evaluate all pre-trained models
                    results = evaluate_pretrained_models(pretrained_models, scaler, X_test, y_test)
                    
                    # Calculate metrics for all models
                    all_metrics = {}
                    for model_name, result in results.items():
                        metrics = calculate_metrics(y_test, result['predictions'], result['probabilities'])
                        all_metrics[model_name] = metrics
                    
                    # Create metrics table
                    st.subheader("Evaluation Metrics for All Pre-trained Models")
                    metrics_df = pd.DataFrame(all_metrics).T
                    metrics_df = metrics_df.round(4)
                    st.dataframe(metrics_df, use_container_width=True)
                    
                    # Store in session state for later use
                    st.session_state.results = results
                    st.session_state.y_test = y_test
                    st.session_state.all_metrics = all_metrics
                    st.session_state.X_test = X_test
                    st.session_state.X_train = X_train
                    st.session_state.df = df
                    
                    st.info("✓ Results loaded to session. Navigate to 'Model Comparison' for detailed analysis.")
                    
                    # Display all trained models results
                    st.subheader("Pre-trained Models Summary")
                    metrics_df = pd.DataFrame(all_metrics).T
                    metrics_df = metrics_df.round(4)
                    st.dataframe(metrics_df, use_container_width=True)
                    
            except Exception as e:
                st.error(f"Error loading models: {e}")
        
        st.divider()
        st.subheader("Train New Models (Optional)")
        st.write("You can also train models from scratch for comparison:")
        
        if st.button("🚀 Train New Models from Scratch", use_container_width=True):
            st.subheader("Training 6 Classification Models...")
            progress_bar = st.progress(0)
            
            results, y_test_actual = train_models(X_train, X_test, y_train, y_test)
            
            progress_bar.progress(100)
            st.success("✓ All models trained successfully!")
            
            # Calculate metrics for all models
            all_metrics = {}
            for model_name, result in results.items():
                metrics = calculate_metrics(y_test_actual, result['predictions'], result['probabilities'])
                all_metrics[model_name] = metrics
            
            # Create metrics table
            st.subheader("Evaluation Metrics for Newly Trained Models")
            metrics_df = pd.DataFrame(all_metrics).T
            metrics_df = metrics_df.round(4)
            st.dataframe(metrics_df, use_container_width=True)
            
            # Store in session state for later use
            st.session_state.results = results
            st.session_state.y_test = y_test_actual
            st.session_state.all_metrics = all_metrics
            st.session_state.X_test = X_test
            st.session_state.X_train = X_train
            st.session_state.df = df
            
            st.info("✓ Results saved to session. Navigate to 'Model Comparison' for detailed analysis.")

# =====================================================
# PAGE 3: MODEL COMPARISON
# =====================================================

elif page == "📈 Model Comparison":
    st.header("Model Comparison & Analysis")
    
    if 'all_metrics' not in st.session_state:
        st.warning("No models loaded. Please go to 'Model Training' page and load pre-trained models first.")
    else:
        # Display all models metrics
        st.subheader("Performance Metrics Comparison")
        metrics_df = pd.DataFrame(st.session_state.all_metrics).T
        metrics_df = metrics_df.round(4)
        st.dataframe(metrics_df, use_container_width=True)
        
        # Metrics visualization
        st.subheader("Metrics Visualization")
        
        # Create comparison charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Accuracy Comparison**")
            fig, ax = plt.subplots(figsize=(8, 5))
            metrics_df['Accuracy'].sort_values(ascending=True).plot(kind='barh', ax=ax, color='steelblue')
            ax.set_xlabel('Accuracy Score')
            ax.set_title('Model Accuracy Comparison')
            st.pyplot(fig)
        
        with col2:
            st.write("**AUC Score Comparison**")
            fig, ax = plt.subplots(figsize=(8, 5))
            metrics_df['AUC'].sort_values(ascending=True).plot(kind='barh', ax=ax, color='orange')
            ax.set_xlabel('AUC Score')
            ax.set_title('Model AUC Comparison')
            st.pyplot(fig)
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.write("**F1 Score Comparison**")
            fig, ax = plt.subplots(figsize=(8, 5))
            metrics_df['F1'].sort_values(ascending=True).plot(kind='barh', ax=ax, color='green')
            ax.set_xlabel('F1 Score')
            ax.set_title('Model F1 Score Comparison')
            st.pyplot(fig)
        
        with col4:
            st.write("**Precision vs Recall**")
            fig, ax = plt.subplots(figsize=(8, 5))
            x = np.arange(len(metrics_df))
            width = 0.35
            ax.bar(x - width/2, metrics_df['Precision'], width, label='Precision', color='coral')
            ax.bar(x + width/2, metrics_df['Recall'], width, label='Recall', color='lightblue')
            ax.set_xlabel('Models')
            ax.set_ylabel('Score')
            ax.set_title('Precision vs Recall')
            ax.set_xticks(x)
            ax.set_xticklabels(metrics_df.index, rotation=45, ha='right')
            ax.legend()
            st.pyplot(fig)
        
        # Detailed model analysis
        st.subheader("Detailed Model Analysis")
        
        selected_model = st.selectbox("Select a model for detailed analysis:", metrics_df.index)
        
        if selected_model and 'results' in st.session_state:
            st.write(f"**{selected_model} - Detailed Report**")
            
            result = st.session_state.results[selected_model]
            metrics = st.session_state.all_metrics[selected_model]
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
            with col2:
                st.metric("AUC Score", f"{metrics['AUC']:.4f}")
            with col3:
                st.metric("F1 Score", f"{metrics['F1']:.4f}")
            
            col4, col5, col6 = st.columns(3)
            with col4:
                st.metric("Precision", f"{metrics['Precision']:.4f}")
            with col5:
                st.metric("Recall", f"{metrics['Recall']:.4f}")
            with col6:
                st.metric("MCC", f"{metrics['MCC']:.4f}")
            
            # Confusion Matrix
            st.write("**Confusion Matrix**")
            cm = confusion_matrix(st.session_state.y_test, result['predictions'])
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            st.pyplot(fig)

# =====================================================
# PAGE 4: PREDICTIONS
# =====================================================

elif page == "🔮 Predictions":
    st.header("Make Predictions")
    
    if 'all_metrics' not in st.session_state:
        st.warning("No models loaded. Please go to 'Model Training' page and load pre-trained models first.")
    else:
        df = load_data()
        X, y, _ = prepare_data(df)
        
        st.write("Enter feature values to make predictions:")
        
        # Create input fields for all features
        input_values = {}
        cols = st.columns(4)
        
        for idx, col_name in enumerate(X.columns):
            col = cols[idx % 4]
            with col:
                input_values[col_name] = st.number_input(
                    f"{col_name}",
                    value=float(X[col_name].mean()),
                    min_value=float(X[col_name].min()),
                    max_value=float(X[col_name].max()),
                    step=0.1
                )
        
        if st.button("🔮 Predict", use_container_width=True):
            # Prepare input data
            input_df = pd.DataFrame([input_values])
            
            # Load scaler
            models, scaler = load_pretrained_models()
            
            st.subheader("Prediction Results")
            
            # Make predictions with all models
            predictions = {}
            
            for model_name, model in st.session_state.results.items():
                try:
                    # Use scaled data for models that need it
                    if model_name in ['Logistic Regression', 'K-Nearest Neighbor']:
                        if scaler is not None:
                            input_scaled = scaler.transform(input_df)
                            pred = model['model'].predict(input_scaled)[0]
                            proba = model['model'].predict_proba(input_scaled)[0]
                        else:
                            pred = model['model'].predict(input_df)[0]
                            proba = model['model'].predict_proba(input_df)[0]
                    else:
                        pred = model['model'].predict(input_df)[0]
                        proba = model['model'].predict_proba(input_df)[0]
                    
                    predictions[model_name] = {
                        'prediction': 'Malignant' if pred == 1 else 'Benign',
                        'confidence': max(proba) * 100
                    }
                except Exception as e:
                    st.error(f"Error predicting with {model_name}: {e}")
            
            # Display predictions
            cols = st.columns(2)
            
            for idx, (model_name, pred_data) in enumerate(predictions.items()):
                with cols[idx % 2]:
                    color = "🔴" if pred_data['prediction'] == 'Malignant' else "🟢"
                    st.info(
                        f"{color} **{model_name}**\n\n"
                        f"Prediction: **{pred_data['prediction']}**\n\n"
                        f"Confidence: **{pred_data['confidence']:.2f}%**"
                    )
            
            # Consensus prediction
            malignant_count = sum(1 for p in predictions.values() if p['prediction'] == 'Malignant')
            consensus = 'Malignant' if malignant_count > len(predictions) / 2 else 'Benign'
            consensus_color = "🔴" if consensus == 'Malignant' else "🟢"
            
            st.divider()
            st.success(
                f"{consensus_color} **CONSENSUS PREDICTION: {consensus}**\n\n"
                f"{malignant_count}/{len(predictions)} models predict Malignant"
            )