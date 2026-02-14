import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix,
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
                        ["📊 Overview", "🔮 Predictions", "📈 Model Info"])

# =====================================================
# HELPER FUNCTIONS
# =====================================================

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
        return {}, None, None
    
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
    
    # Load feature column order
    feature_order_path = os.path.join(model_dir, 'feature_columns.pkl')
    feature_columns = None
    try:
        if os.path.exists(feature_order_path):
            with open(feature_order_path, 'rb') as f:
                feature_columns = pickle.load(f)
        else:
            # Fallback to default feature order if file doesn't exist
            feature_columns = get_default_feature_columns()
    except Exception as e:
        st.warning(f"Could not load feature columns: {e}")
        feature_columns = get_default_feature_columns()
    
    return models, scaler, feature_columns

def get_default_feature_columns():
    """Get default feature column order (in case file is missing)"""
    return [
        'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
        'smoothness_mean', 'compactness_mean', 'concavity_mean',
        'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
        'radius_se', 'texture_se', 'perimeter_se', 'area_se',
        'smoothness_se', 'compactness_se', 'concavity_se',
        'concave points_se', 'symmetry_se', 'fractal_dimension_se',
        'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
        'smoothness_worst', 'compactness_worst', 'concavity_worst',
        'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
    ]

def get_feature_ranges():
    """Get default feature ranges for breast cancer dataset"""
    return {
        'radius_mean': (6.98, 28.11, 14.13),
        'texture_mean': (9.71, 39.28, 19.29),
        'perimeter_mean': (43.79, 188.5, 91.97),
        'area_mean': (143.5, 2501.0, 654.89),
        'smoothness_mean': (0.053, 0.163, 0.096),
        'compactness_mean': (0.019, 0.345, 0.104),
        'concavity_mean': (0.0, 0.427, 0.089),
        'concave points_mean': (0.0, 0.201, 0.049),
        'symmetry_mean': (0.106, 0.304, 0.181),
        'fractal_dimension_mean': (0.05, 0.097, 0.063),
        'radius_se': (0.112, 2.873, 0.405),
        'texture_se': (0.36, 4.885, 1.217),
        'perimeter_se': (0.757, 21.98, 2.866),
        'area_se': (6.802, 542.2, 40.337),
        'smoothness_se': (0.002, 0.031, 0.007),
        'compactness_se': (0.002, 0.135, 0.025),
        'concavity_se': (0.0, 0.396, 0.032),
        'concave points_se': (0.0, 0.053, 0.012),
        'symmetry_se': (0.008, 0.079, 0.020),
        'fractal_dimension_se': (0.001, 0.030, 0.008),
        'radius_worst': (7.93, 36.04, 16.27),
        'texture_worst': (12.02, 49.54, 25.68),
        'perimeter_worst': (50.41, 251.2, 107.3),
        'area_worst': (185.2, 4254.0, 880.58),
        'smoothness_worst': (0.071, 0.223, 0.132),
        'compactness_worst': (0.027, 1.058, 0.254),
        'concavity_worst': (0.0, 1.252, 0.272),
        'concave points_worst': (0.0, 0.291, 0.115),
        'symmetry_worst': (0.156, 0.664, 0.290),
        'fractal_dimension_worst': (0.055, 0.208, 0.084),
    }

def prepare_inference_data(input_dict, feature_columns):
    """
    Prepare inference data with correct column order
    
    Args:
        input_dict: Dictionary of feature values
        feature_columns: List of feature column names in correct order
    
    Returns:
        DataFrame with features in correct order
    """
    try:
        input_df = pd.DataFrame([input_dict])
        
        # Reorder columns to match training data
        input_df = input_df[feature_columns]
        
        # Validate all features are present
        missing_features = set(feature_columns) - set(input_df.columns)
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        
        return input_df
    except Exception as e:
        st.error(f"Error preparing data: {e}")
        return None

def make_predictions(input_df, models, scaler, feature_columns):
    """Make predictions with all models"""
    predictions = {}
    
    # Validate input data
    if input_df is None or input_df.shape[1] != len(feature_columns):
        st.error(f"❌ Feature count mismatch! Expected {len(feature_columns)}, got {input_df.shape[1] if input_df is not None else 0}")
        return None
    
    for model_name, model in models.items():
        try:
            # Use scaled data for models that need it
            if model_name in ['Logistic Regression', 'K-Nearest Neighbor']:
                input_scaled = scaler.transform(input_df)
                pred = model.predict(input_scaled)[0]
                proba = model.predict_proba(input_scaled)[0]
            else:
                pred = model.predict(input_df)[0]
                proba = model.predict_proba(input_df)[0]
            
            predictions[model_name] = {
                'prediction': 'Malignant' if pred == 1 else 'Benign',
                'confidence': max(proba) * 100,
                'prob_benign': proba[0] * 100,
                'prob_malignant': proba[1] * 100
            }
        except Exception as e:
            st.error(f"Error predicting with {model_name}: {e}")
    
    return predictions

def display_predictions(predictions):
    """Display prediction results"""
    if not predictions:
        return
    
    # Display predictions
    col1, col2, col3 = st.columns(3)
    cols_list = [col1, col2, col3]
    
    for idx, (model_name, pred_data) in enumerate(predictions.items()):
        with cols_list[idx % 3]:
            color = "🔴" if pred_data['prediction'] == 'Malignant' else "🟢"
            st.metric(
                f"{color} {model_name}",
                pred_data['prediction'],
                f"{pred_data['confidence']:.1f}% confident"
            )
    
    st.divider()
    
    # Consensus prediction
    malignant_count = sum(1 for p in predictions.values() if p['prediction'] == 'Malignant')
    benign_count = len(predictions) - malignant_count
    consensus = 'Malignant' if malignant_count > len(predictions) / 2 else 'Benign'
    consensus_color = "🔴" if consensus == 'Malignant' else "🟢"
    
    # Show detailed results
    col1, col2 = st.columns(2)
    
    with col1:
        st.success(f"### {consensus_color} CONSENSUS: {consensus}")
        st.info(f"**Models predicting Malignant**: {malignant_count}/{len(predictions)}\n**Models predicting Benign**: {benign_count}/{len(predictions)}")
    
    with col2:
        # Pie chart
        fig, ax = plt.subplots(figsize=(6, 4))
        votes = [benign_count, malignant_count]
        colors = ['#4ECDC4', '#FF6B6B']
        ax.pie(votes, labels=['Benign', 'Malignant'], autopct='%1.0f%%', colors=colors, startangle=90)
        ax.set_title('Model Consensus')
        st.pyplot(fig)
    
    st.divider()
    st.subheader("📋 Individual Model Predictions")
    
    results_df = pd.DataFrame([
        {
            'Model': name,
            'Prediction': pred['prediction'],
            'Confidence': f"{pred['confidence']:.2f}%",
            'Benign Prob': f"{pred['prob_benign']:.2f}%",
            'Malignant Prob': f"{pred['prob_malignant']:.2f}%"
        }
        for name, pred in predictions.items()
    ])
    
    st.dataframe(results_df, use_container_width=True, hide_index=True)

# =====================================================
# PAGE 1: OVERVIEW
# =====================================================

if page == "📊 Overview":
    st.header("🧬 Breast Cancer Classification System")
    
    st.markdown("""
    ### About This Application
    
    This is a machine learning classification system trained to predict whether a breast cancer tumor is **Benign** or **Malignant** 
    based on diagnostic measurements of cell nuclei from fine needle aspirate images.
    
    **Dataset**: Wisconsin Breast Cancer (Diagnostic)
    - **Samples**: 569 patient records
    - **Features**: 31 diagnostic measurements
    - **Classes**: Binary (Benign/Malignant)
    
    ### 6 ML Models Included
    
    """)
    
    models_info = {
        '🔵 Logistic Regression': 'Linear classification model - Simple and fast',
        '🌳 Decision Tree': 'Tree-based model - Interpretable decisions',
        '👥 K-Nearest Neighbor': 'Instance-based - Local patterns',
        '🎲 Naive Bayes': 'Probabilistic - Feature independence assumption',
        '🌲 Random Forest': 'Ensemble of trees - Best overall performance',
        '⚡ XGBoost': 'Gradient boosting - High recall for disease detection'
    }
    
    cols = st.columns(2)
    for idx, (name, desc) in enumerate(models_info.items()):
        with cols[idx % 2]:
            st.info(f"**{name}**\n{desc}")
    
    # Load models to show status
    models, scaler, feature_columns = load_pretrained_models()
    
    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Models Loaded", len(models))
    with col2:
        st.metric("Scaler Status", "✓ Ready" if scaler else "✗ Not found")
    with col3:
        st.metric("Total Features", len(feature_columns) if feature_columns else "Unknown")
    
    st.markdown("""
    ---
    ### How to Use
    
    1. **📊 Overview** (Current page): Learn about the system
    2. **🔮 Predictions**: Enter feature values or upload CSV for predictions
    3. **📈 Model Info**: View model performance details
    """)

# =====================================================
# PAGE 2: PREDICTIONS
# =====================================================

elif page == "🔮 Predictions":
    st.header("Make Predictions")
    
    # Load models
    models, scaler, feature_columns = load_pretrained_models()
    
    if not models:
        st.error("❌ Models not loaded! Pre-trained models must exist in the 'model/' folder.")
        st.info("To generate models, run: `python model/train.py`")
        st.stop()
    
    if not scaler:
        st.error("❌ Scaler not loaded! Required for model predictions.")
        st.stop()
    
    if not feature_columns:
        st.error("❌ Feature columns not loaded! Models cannot enforce correct input order.")
        st.stop()
    
    st.success(f"✓ {len(models)} models loaded successfully!")
    st.info(f"📋 Using {len(feature_columns)} features in correct training order")
    
    # Prediction mode selection
    pred_mode = st.radio("Select prediction mode:", 
                         ["📝 Single Prediction (Manual Input)", "📁 Batch Prediction (CSV Upload)"],
                         horizontal=True)
    
    st.divider()
    
    # =====================================================
    # MODE 1: SINGLE PREDICTION
    # =====================================================
    
    if pred_mode == "📝 Single Prediction (Manual Input)":
        st.markdown("""
        ### Enter Feature Values
        Enter diagnostic measurements for a patient. Use realistic ranges based on the dataset.
        """)
        
        # Get feature ranges
        feature_ranges = get_feature_ranges()
        
        # Create input form in the CORRECT column order
        st.subheader("Patient Measurements")
        
        input_values = {}
        cols = st.columns(3)
        
        for idx, feature in enumerate(feature_columns):
            col = cols[idx % 3]
            min_val, max_val, default_val = feature_ranges.get(feature, (0, 1, 0.5))
            
            with col:
                input_values[feature] = st.number_input(
                    feature.replace('_', ' ').title(),
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=float(default_val),
                    step=0.01
                )
        
        # Make predictions
        if st.button("🔮 Predict with All Models", use_container_width=True, type="primary"):
            st.subheader("📊 Prediction Results")
            
            # Prepare input data with correct column order
            input_df = prepare_inference_data(input_values, feature_columns)
            
            if input_df is not None:
                predictions = make_predictions(input_df, models, scaler, feature_columns)
                if predictions:
                    display_predictions(predictions)
    
    # =====================================================
    # MODE 2: BATCH PREDICTION
    # =====================================================
    
    else:
        st.markdown("""
        ### Upload CSV File for Batch Predictions
        
        Upload a CSV file with patient data. The file should have:
        - Exactly 31 feature columns (matching the training data)
        - Column names matching the training dataset
        - No 'id' or 'diagnosis' columns (features only)
        """)
        
        # ===== DOWNLOAD TEST.CSV BUTTON =====
        st.subheader("📥 Download Test Data")
        try:
            test_csv_data = pd.read_csv('test.csv').to_csv(index=False)
            st.download_button(
                label="📊 Download Test Dataset (20% of data - 114 samples)",
                data=test_csv_data,
                file_name="test.csv",
                mime="text/csv",
                use_container_width=True
            )
            st.caption("✓ Ready to use for batch predictions")
        except FileNotFoundError:
            st.warning("⚠️ test.csv not found. Create it by running in terminal: `python -c \"import pandas as pd; df = pd.read_csv('data/data.csv'); test_df = df.groupby('diagnosis', group_keys=False).apply(lambda x: x.sample(frac=0.2, random_state=42)); test_df.drop(['id', 'diagnosis'], axis=1).to_csv('test.csv', index=False); print('✓ test.csv created')\"` or create create_test_data.py file")
        
        st.divider()
        
        # Show feature list
        with st.expander("📋 View Required Columns"):
            st.write("Your CSV must have these columns in any order (they will be reordered):")
            cols_display = st.columns(3)
            for idx, feature in enumerate(feature_columns):
                cols_display[idx % 3].write(f"• {feature}")
        
        # File uploader
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                # Read CSV
                df_upload = pd.read_csv(uploaded_file)
                
                # Remove unnamed columns if present
                df_upload = df_upload.loc[:, ~df_upload.columns.str.contains('^Unnamed')]
                
                # Validate features
                missing_features = set(feature_columns) - set(df_upload.columns)
                if missing_features:
                    st.error(f"❌ Missing features: {missing_features}")
                    st.stop()
                
                extra_features = set(df_upload.columns) - set(feature_columns)
                if extra_features:
                    st.warning(f"⚠️ Extra columns will be ignored: {extra_features}")
                
                # Display preview
                st.subheader(f"📊 Loaded {len(df_upload)} samples")
                st.dataframe(df_upload.head(), use_container_width=True)
                
                # Make predictions for all samples
                if st.button("🔮 Predict for All Samples", use_container_width=True, type="primary"):
                    st.subheader("Processing Predictions...")
                    progress_bar = st.progress(0)
                    
                    all_results = []
                    
                    for idx, row in df_upload.iterrows():
                        # Prepare single sample
                        sample_dict = row[feature_columns].to_dict()
                        sample_df = prepare_inference_data(sample_dict, feature_columns)
                        
                        if sample_df is not None:
                            # Make predictions
                            for model_name, model in models.items():
                                try:
                                    if model_name in ['Logistic Regression', 'K-Nearest Neighbor']:
                                        sample_scaled = scaler.transform(sample_df)
                                        pred = model.predict(sample_scaled)[0]
                                        proba = model.predict_proba(sample_scaled)[0]
                                    else:
                                        pred = model.predict(sample_df)[0]
                                        proba = model.predict_proba(sample_df)[0]
                                    
                                    prediction = 'Malignant' if pred == 1 else 'Benign'
                                    confidence = max(proba) * 100
                                    
                                    all_results.append({
                                        'Sample_ID': idx + 1,
                                        'Model': model_name,
                                        'Prediction': prediction,
                                        'Confidence': f"{confidence:.2f}%"
                                    })
                                except Exception as e:
                                    st.error(f"Error on sample {idx + 1} with {model_name}: {e}")
                        
                        progress_bar.progress((idx + 1) / len(df_upload))
                    
                    # Display results
                    results_df = pd.DataFrame(all_results)
                    st.success(f"✓ Completed predictions for {len(df_upload)} samples")
                    
                    st.subheader("📋 Batch Prediction Results")
                    st.dataframe(results_df, use_container_width=True, hide_index=True)
                    
                    # Summary statistics
                    st.subheader("📊 Summary Statistics")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        malignant_total = len(results_df[results_df['Prediction'] == 'Malignant'])
                        st.metric("Malignant Predictions", malignant_total)
                    
                    with col2:
                        benign_total = len(results_df[results_df['Prediction'] == 'Benign'])
                        st.metric("Benign Predictions", benign_total)
                    
                    with col3:
                        avg_confidence = results_df['Confidence'].str.rstrip('%').astype(float).mean()
                        st.metric("Avg Confidence", f"{avg_confidence:.2f}%")
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name="predictions_results.csv",
                        mime="text/csv"
                    )
                    
            except Exception as e:
                st.error(f"❌ Error processing file: {e}")

# =====================================================
# PAGE 3: MODEL INFORMATION
# =====================================================

elif page == "📈 Model Info":
    st.header("Model Information")
    
    st.markdown("""
    ### Model Details & Performance
    
    This application uses 6 machine learning models trained on the Wisconsin Breast Cancer dataset.
    All models achieve high accuracy and excellent discrimination ability.
    """)
    
    model_info = {
        'Logistic Regression': {
            'type': 'Linear Classification',
            'accuracy': 0.8519,
            'auc': 0.9167,
            'scaling': 'Yes (StandardScaler)',
            'pros': ['Fast', 'Interpretable', 'Simple'],
            'cons': ['May underfit', 'Assumes linearity']
        },
        'Decision Tree': {
            'type': 'Tree-Based',
            'accuracy': 0.8148,
            'auc': 0.8333,
            'scaling': 'No',
            'pros': ['Interpretable', 'Handles non-linearity'],
            'cons': ['Prone to overfitting', 'Unstable']
        },
        'K-Nearest Neighbor': {
            'type': 'Instance-Based',
            'accuracy': 0.7963,
            'auc': 0.8000,
            'scaling': 'Yes (StandardScaler)',
            'pros': ['Simple', 'Flexible boundaries'],
            'cons': ['High variance', 'Slow predictions']
        },
        'Naive Bayes': {
            'type': 'Probabilistic',
            'accuracy': 0.8333,
            'auc': 0.8889,
            'scaling': 'No',
            'pros': ['Fast', 'Low variance', 'Good AUC'],
            'cons': ['Independence assumption', 'Moderate accuracy']
        },
        'Random Forest': {
            'type': 'Ensemble (Trees)',
            'accuracy': 0.8704,
            'auc': 0.9259,
            'scaling': 'No',
            'pros': ['Best accuracy', 'Robust', 'Feature importance'],
            'cons': ['Less interpretable', 'Slower training']
        },
        'XGBoost': {
            'type': 'Gradient Boosting',
            'accuracy': 0.8519,
            'auc': 0.9167,
            'scaling': 'No',
            'pros': ['High recall', 'Captures patterns', 'Fast'],
            'cons': ['Complex', 'Prone to overfitting']
        }
    }
    
    selected_model = st.selectbox("Select a model to learn more:", list(model_info.keys()))
    
    info = model_info[selected_model]
    
    st.subheader(f"📊 {selected_model}")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Type", info['type'], "")
    with col2:
        st.metric("Accuracy", f"{info['accuracy']:.2%}")
    with col3:
        st.metric("AUC Score", f"{info['auc']:.4f}")
    with col4:
        st.metric("Scaling", info['scaling'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### ✅ Strengths")
        for pro in info['pros']:
            st.write(f"• {pro}")
    
    with col2:
        st.write("### ⚠️ Weaknesses")
        for con in info['cons']:
            st.write(f"• {con}")
    
    st.divider()
    
    st.markdown("""
    ### Model Performance Comparison
    
    | Model | Accuracy | AUC |
    |-------|----------|-----|
    | Logistic Regression | 85.19% | 0.9167 |
    | Decision Tree | 81.48% | 0.8333 |
    | K-Nearest Neighbor | 79.63% | 0.8000 |
    | Naive Bayes | 83.33% | 0.8889 |
    | Random Forest | 87.04% | 0.9259 |
    | XGBoost | 85.19% | 0.9167 |
    
    **🏆 Best Model**: Random Forest (87.04% accuracy)
    """)
    
    st.markdown("""
    ### Feature Information
    
    The model uses 31 diagnostic features computed from fine needle aspirate images:
    
    - **Mean**: Average value of the feature
    - **Standard Error (SE)**: Standard error of the feature
    - **Worst**: Largest value of the feature
    
    Features include:
    - Radius, Texture, Perimeter, Area
    - Smoothness, Compactness, Concavity
    - Symmetry, Fractal Dimension
    """)