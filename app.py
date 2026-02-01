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

def train_models(X_train, X_test, y_train, y_test):
    """Train all 6 models and return results"""
    
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
        st.subheader("Train Classification Models")
        st.write("Choose to train all models or individual models:")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.write("**Batch Training:**")
            if st.button("🚀 Train All 6 Models", key="train_all_btn", use_container_width=True):
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
                st.subheader("Evaluation Metrics for All Models")
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
        
        with col2:
            st.write("**Individual Model Training:**")
            model_names = ['Logistic Regression', 'Decision Tree', 'K-Nearest Neighbor', 
                          'Naive Bayes', 'Random Forest', 'XGBoost']
            
            # Create 3 columns x 2 rows for model buttons
            cols = st.columns(3)
            
            for idx, model_name in enumerate(model_names):
                with cols[idx % 3]:
                    if st.button(f"📊 {model_name}", key=f"train_{model_name}", use_container_width=True):
                        st.info(f"Training {model_name}...")
                        
                        # Initialize session state for individual models if not exists
                        if 'individual_results' not in st.session_state:
                            st.session_state.individual_results = {}
                            st.session_state.individual_metrics = {}
                        
                        # Scale features for models that need it
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                        
                        # Train individual model based on selection
                        if model_name == 'Logistic Regression':
                            model = LogisticRegression(max_iter=1000, random_state=42)
                            model.fit(X_train_scaled, y_train)
                            y_pred = model.predict(X_test_scaled)
                            y_proba = model.predict_proba(X_test_scaled)[:, 1]
                            scaler_used = scaler
                        
                        elif model_name == 'Decision Tree':
                            model = DecisionTreeClassifier(random_state=42)
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                            y_proba = model.predict_proba(X_test)[:, 1]
                            scaler_used = None
                        
                        elif model_name == 'K-Nearest Neighbor':
                            model = KNeighborsClassifier(n_neighbors=5)
                            model.fit(X_train_scaled, y_train)
                            y_pred = model.predict(X_test_scaled)
                            y_proba = model.predict_proba(X_test_scaled)[:, 1]
                            scaler_used = scaler
                        
                        elif model_name == 'Naive Bayes':
                            model = GaussianNB()
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                            y_proba = model.predict_proba(X_test)[:, 1]
                            scaler_used = None
                        
                        elif model_name == 'Random Forest':
                            model = RandomForestClassifier(n_estimators=100, random_state=42)
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                            y_proba = model.predict_proba(X_test)[:, 1]
                            scaler_used = None
                        
                        elif model_name == 'XGBoost':
                            model = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                            y_proba = model.predict_proba(X_test)[:, 1]
                            scaler_used = None
                        
                        # Store results
                        st.session_state.individual_results[model_name] = {
                            'model': model,
                            'predictions': y_pred,
                            'probabilities': y_proba,
                            'scaler': scaler_used
                        }
                        
                        # Calculate metrics
                        metrics = calculate_metrics(y_test, y_pred, y_proba)
                        st.session_state.individual_metrics[model_name] = metrics
                        
                        # Store overall session data if not already present
                        if 'results' not in st.session_state:
                            st.session_state.results = st.session_state.individual_results
                            st.session_state.all_metrics = st.session_state.individual_metrics
                            st.session_state.y_test = y_test
                            st.session_state.X_test = X_test
                            st.session_state.X_train = X_train
                            st.session_state.df = df
                        
                        st.success(f"✓ {model_name} trained successfully!")
                        st.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
        
        # Display all trained models results
        if 'individual_metrics' in st.session_state and st.session_state.individual_metrics:
            st.subheader("Trained Models Summary")
            metrics_df = pd.DataFrame(st.session_state.individual_metrics).T
            metrics_df = metrics_df.round(4)
            st.dataframe(metrics_df, use_container_width=True)
            
            st.info("✓ Results saved. Navigate to 'Model Comparison' for detailed analysis of trained models.")

# =====================================================
# PAGE 3: MODEL COMPARISON
# =====================================================

elif page == "📈 Model Comparison":
    st.header("Model Performance Comparison")
    
    if 'all_metrics' not in st.session_state:
        st.warning("⚠️ Please train the models first on the 'Model Training' page!")
    else:
        all_metrics = st.session_state.all_metrics
        results = st.session_state.results
        y_test = st.session_state.y_test
        
        # Display metrics table
        st.subheader("Metrics Comparison Table")
        metrics_df = pd.DataFrame(all_metrics).T
        metrics_df = metrics_df.round(4)
        st.dataframe(metrics_df, use_container_width=True)
        
        # Visualization
        st.subheader("Metrics Visualization")
        
        metrics_list = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
        
        # Create tabs for different metrics
        tab1, tab2, tab3 = st.tabs(["Individual Metrics", "Comparison Chart", "Detailed Analysis"])
        
        with tab1:
            selected_metric = st.selectbox("Select Metric", metrics_list)
            fig, ax = plt.subplots(figsize=(12, 6))
            
            models = list(all_metrics.keys())
            values = [all_metrics[model][selected_metric] for model in models]
            
            colors = plt.cm.Set3(range(len(models)))
            bars = ax.bar(models, values, color=colors, edgecolor='black', linewidth=1.5)
            ax.set_ylabel(selected_metric, fontsize=12)
            ax.set_ylim(0, 1)
            ax.set_xticklabels(models, rotation=45, ha='right')
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                       f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with tab2:
            fig, axes = plt.subplots(2, 3, figsize=(16, 10))
            axes = axes.ravel()
            
            models = list(all_metrics.keys())
            colors = plt.cm.Set3(range(len(models)))
            
            for idx, metric in enumerate(metrics_list):
                ax = axes[idx]
                values = [all_metrics[model][metric] for model in models]
                
                bars = ax.bar(models, values, color=colors, edgecolor='black', linewidth=1.5)
                ax.set_ylabel(metric, fontsize=11)
                ax.set_ylim(0, 1)
                ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
                ax.grid(axis='y', alpha=0.3)
                
                for bar, val in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                           f'{val:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with tab3:
            st.write("### Detailed Performance Analysis")
            st.write("Select a model to view its detailed metrics and confusion matrix:")
            
            selected_model = st.selectbox("Select Model", list(all_metrics.keys()))
            
            col1, col2, col3 = st.columns(3)
            
            metrics = all_metrics[selected_model]
            with col1:
                st.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
            with col2:
                st.metric("AUC Score", f"{metrics['AUC']:.4f}")
            with col3:
                st.metric("F1 Score", f"{metrics['F1']:.4f}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Precision", f"{metrics['Precision']:.4f}")
            with col2:
                st.metric("Recall", f"{metrics['Recall']:.4f}")
            with col3:
                st.metric("MCC Score", f"{metrics['MCC']:.4f}")
            
            # Confusion Matrix
            y_pred = results[selected_model]['predictions']
            cm = confusion_matrix(y_test, y_pred)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                       xticklabels=['No Disease', 'Disease'],
                       yticklabels=['No Disease', 'Disease'])
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title(f'{selected_model} - Confusion Matrix')
            st.pyplot(fig)
            
            # Classification Report
            st.write("### Classification Report")
            report = classification_report(y_test, y_pred, target_names=['No Disease', 'Disease'])
            st.text(report)
        
        # New Tab: All Confusion Matrices
        with st.expander("📊 View All Models' Confusion Matrices"):
            st.write("### Confusion Matrices for All 6 Models")
            st.write("Visualize prediction accuracy for each model:")
            
            # Create 2x3 grid for all confusion matrices
            cols = st.columns(3)
            model_names = list(results.keys())
            
            for idx, model_name in enumerate(model_names):
                with cols[idx % 3]:
                    y_pred_cm = results[model_name]['predictions']
                    cm_data = confusion_matrix(y_test, y_pred_cm)
                    
                    fig, ax = plt.subplots(figsize=(6, 5))
                    sns.heatmap(cm_data, annot=True, fmt='d', cmap='RdYlGn', ax=ax, 
                               cbar=False, annot_kws={'size': 14, 'weight': 'bold'},
                               xticklabels=['No Disease', 'Disease'],
                               yticklabels=['No Disease', 'Disease'])
                    ax.set_xlabel('Predicted', fontweight='bold')
                    ax.set_ylabel('Actual', fontweight='bold')
                    ax.set_title(f'{model_name}', fontweight='bold', fontsize=12)
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Display metrics for this model below its confusion matrix
                    metrics_data = all_metrics[model_name]
                    st.markdown(f"""
                    **Performance Metrics:**
                    - Accuracy: {metrics_data['Accuracy']:.4f}
                    - Precision: {metrics_data['Precision']:.4f}
                    - Recall: {metrics_data['Recall']:.4f}
                    - F1 Score: {metrics_data['F1']:.4f}
                    """)


# =====================================================
# PAGE 4: PREDICTIONS
# =====================================================

elif page == "🔮 Predictions":
    st.header("Make Predictions on New Data")
    
    df = load_data()
    
    if df is not None:
        if 'all_metrics' not in st.session_state:
            st.warning("⚠️ Please train the models first on the 'Model Training' page!")
        else:
            results = st.session_state.results
            
            # Option 1: Single prediction
            st.subheader("Option 1: Single Patient Prediction")
            
            # Model selector
            st.write("**Select Model for Prediction:**")
            selected_pred_model = st.selectbox(
                "Choose a classification model",
                list(results.keys()),
                key="pred_model_select"
            )
            
            # Get feature ranges from training data
            X_train = st.session_state.X_train
            feature_ranges = {}
            for col in X_train.columns:
                feature_ranges[col] = (X_train[col].min(), X_train[col].max())
            
            # Create input form
            st.write("Enter patient clinical data:")
            
            cols = st.columns(3)
            user_input = {}
            
            for idx, feature in enumerate(X_train.columns):
                with cols[idx % 3]:
                    min_val, max_val = feature_ranges[feature]
                    user_input[feature] = st.number_input(
                        feature,
                        min_value=float(min_val),
                        max_value=float(max_val),
                        value=float((min_val + max_val) / 2),
                        step=1.0
                    )
            
            if st.button("🔍 Predict", key="predict_btn"):
                # Create prediction dataframe
                user_df = pd.DataFrame([user_input])
                
                # Get selected model result
                result = results[selected_pred_model]
                if result['scaler'] is not None:
                    user_df_scaled = result['scaler'].transform(user_df)
                else:
                    user_df_scaled = user_df
                
                pred = result['model'].predict(user_df_scaled)[0]
                proba = result['model'].predict_proba(user_df_scaled)[0]
                
                # Display results for selected model
                st.subheader(f"Prediction Results - {selected_pred_model}")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    prediction_text = "🔴 Disease Present" if pred == 1 else "🟢 No Disease"
                    if pred == 1:
                        st.error(prediction_text)
                    else:
                        st.success(prediction_text)
                
                with col2:
                    st.metric("Confidence (No Disease)", f"{proba[0]:.2%}")
                
                with col3:
                    st.metric("Confidence (Disease)", f"{proba[1]:.2%}")
                
                # Display confidence chart
                fig, ax = plt.subplots(figsize=(8, 4))
                classes = ['No Disease', 'Disease']
                confidences = [proba[0], proba[1]]
                colors = ['green', 'red']
                ax.barh(classes, confidences, color=colors, alpha=0.7)
                ax.set_xlabel('Confidence Score')
                ax.set_title(f'{selected_pred_model} - Prediction Confidence')
                ax.set_xlim(0, 1)
                for i, v in enumerate(confidences):
                    ax.text(v + 0.02, i, f'{v:.2%}', va='center')
                st.pyplot(fig)
            
            # Option 2: Upload CSV for batch predictions
            st.subheader("Option 2: Batch Predictions from CSV")
            
            # Model selector for batch predictions
            st.write("**Select Model for Batch Prediction:**")
            selected_batch_model = st.selectbox(
                "Choose a classification model",
                list(results.keys()),
                key="batch_model_select"
            )
            
            uploaded_file = st.file_uploader("Upload CSV file for predictions", type=['csv'])
            
            if uploaded_file is not None:
                test_data = pd.read_csv(uploaded_file)
                
                st.write(f"Uploaded {len(test_data)} samples")
                
                if st.button("🔍 Predict All"):
                    st.subheader(f"Batch Prediction Results - {selected_batch_model}")
                    
                    # Get selected model for batch prediction
                    result = results[selected_batch_model]
                    if result['scaler'] is not None:
                        test_data_scaled = result['scaler'].transform(test_data)
                    else:
                        test_data_scaled = test_data
                    
                    preds = result['model'].predict(test_data_scaled)
                    proba = result['model'].predict_proba(test_data_scaled)
                    
                    # Create results dataframe
                    results_df = test_data.copy()
                    results_df[f'{selected_batch_model}_Prediction'] = preds
                    results_df[f'{selected_batch_model}_Prediction'] = results_df[f'{selected_batch_model}_Prediction'].map({0: 'No Disease', 1: 'Disease'})
                    results_df[f'{selected_batch_model}_Confidence_Disease'] = proba[:, 1]
                    results_df[f'{selected_batch_model}_Confidence_No_Disease'] = proba[:, 0]
                    
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="Download Results",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv"
                    )
