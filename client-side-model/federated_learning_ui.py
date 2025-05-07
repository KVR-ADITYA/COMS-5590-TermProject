import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import boto3
import pickle
import tenseal as ts
import os
import datetime
import time
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import requests
import json

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Federated Fraud Detection System",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #F3F4F6;
        margin-bottom: 1rem;
    }
    .success-msg {
        color: green;
        font-weight: 600;
    }
    .error-msg {
        color: red;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# State management
if 'client' not in st.session_state:
    st.session_state.client = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'data' not in st.session_state:
    st.session_state.data = None
if 'X' not in st.session_state:
    st.session_state.X = None
if 'y' not in st.session_state:
    st.session_state.y = None
if 'context' not in st.session_state:
    st.session_state.context = None
if 'local_accuracy' not in st.session_state:
    st.session_state.local_accuracy = None
if 'aggregated_accuracy' not in st.session_state:
    st.session_state.aggregated_accuracy = None

# Title
st.markdown("<h1 class='main-header'>Federated Fraud Detection System</h1>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("<h2 class='sub-header'>Configuration</h2>", unsafe_allow_html=True)
    
    aws_profile = st.text_input("AWS Profile Name", value="client1_user_5590")
    client_id = st.text_input("Client ID", value="client1")
    
    # API endpoint
    api_endpoint = st.text_input("API Endpoint", value=os.getenv("API_ENDPOINT", "https://api.example.com/v1"))
    
    # Upload data file
    uploaded_file = st.file_uploader("Upload Data CSV", type=["csv"])
    
    if uploaded_file is not None:
        # Save the file to a temporary location
        with open("temp_data.csv", "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("File uploaded successfully!")
        
        # Load the data
        try:
            df = pd.read_csv("temp_data.csv")
            st.session_state.data = df
            st.success(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")

# Main content
tabs = st.tabs(["Setup", "Data Analysis", "Local Training", "Federated Learning", "Results"])

# Setup tab
with tabs[0]:
    st.markdown("<h2 class='sub-header'>System Setup</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3>AWS Connection</h3>", unsafe_allow_html=True)
        
        connect_aws = st.button("Connect to AWS")
        
        if connect_aws:
            try:
                session = boto3.Session(profile_name=aws_profile)
                st.session_state.session = session
                st.success("Successfully connected to AWS!")
            except Exception as e:
                st.error(f"Failed to connect to AWS: {str(e)}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3>Homomorphic Encryption Keys</h3>", unsafe_allow_html=True)
        
        key_option = st.radio(
            "Key Option",
            ["Generate New Keys", "Use Existing Keys"]
        )
        
        setup_keys = st.button("Setup Keys")
        
        if setup_keys:
            with st.spinner("Setting up encryption keys..."):
                try:
                    session = st.session_state.get('session')
                    if session is None:
                        st.error("Please connect to AWS first!")
                    else:
                        s3 = session.client('s3')
                        
                        if key_option == "Generate New Keys":
                            # Implementation based on keygen.py
                            st.info("Generating new HE keys...")
                            context = ts.context(
                                ts.SCHEME_TYPE.CKKS, 
                                poly_modulus_degree=8192, 
                                coeff_mod_bit_sizes=[50, 30, 30, 50]
                            )
                            context.generate_galois_keys()
                            context.global_scale = 2**30

                            # Store private context
                            key_data = {
                                'private_context': context.serialize(save_secret_key=True),
                                'public_context': context.serialize(save_secret_key=False)
                            }
                            
                            s3.put_object(
                                Bucket='fraud-detection-encrypted-keys',
                                Key=f'he-keys/{client_id}.bin',
                                Body=pickle.dumps(key_data),
                            )
                            
                            s3.put_object(
                                Bucket='fraud-detection-encrypted-weights',
                                Key=f'public-key/{client_id}_public_context.pkl',
                                Body=key_data['public_context']
                            )
                            
                            st.session_state.context = context
                            st.success("New encryption keys generated successfully!")
                            
                        else:  # Use Existing Keys
                            # Implementation based on core_model.py's get_he_keys method
                            try:
                                response = requests.get(f"{api_endpoint}/keys/client")
                                
                                if response.status_code != 200:
                                    st.error("Failed to get key reference")
                                else:
                                    s3_obj = s3.get_object(
                                        Bucket=response.json()["bucket"],
                                        Key=response.json()["s3_key"]
                                    )
                                    
                                    key_data = pickle.loads(s3_obj['Body'].read())
                                    context = ts.context_from(key_data['private_context'])
                                    st.session_state.context = context
                                    st.success("Existing encryption keys loaded successfully!")
                            except Exception as e:
                                st.error(f"Failed to load existing keys: {str(e)}")
                                
                except Exception as e:
                    st.error(f"Error setting up keys: {str(e)}")
        
        st.markdown("</div>", unsafe_allow_html=True)

# Data Analysis tab
with tabs[1]:
    st.markdown("<h2 class='sub-header'>Data Analysis</h2>", unsafe_allow_html=True)
    
    if st.session_state.data is not None:
        df = st.session_state.data
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<h3>Data Preview</h3>", unsafe_allow_html=True)
            st.dataframe(df.head())
            
            st.markdown("<h4>Data Statistics</h4>", unsafe_allow_html=True)
            st.dataframe(df.describe())
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<h3>Data Visualization</h3>", unsafe_allow_html=True)
            
            # Fraud distribution
            fig, ax = plt.subplots(figsize=(10, 6))
            fraud_counts = df['isFraud'].value_counts()
            ax.pie(fraud_counts, labels=['Legitimate', 'Fraud'], autopct='%1.1f%%', explode=[0, 0.1], 
                   colors=['#4CAF50', '#F44336'], shadow=True)
            ax.set_title('Distribution of Fraud vs Legitimate Transactions')
            st.pyplot(fig)
            
            # Feature distributions
            if st.checkbox("Show Feature Distributions"):
                feature = st.selectbox("Select Feature", 
                                       ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest'])
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(data=df, x=feature, hue='isFraud', bins=30, alpha=0.7)
                ax.set_title(f'Distribution of {feature} by Fraud Status')
                st.pyplot(fig)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
        # Prepare data for modeling
        if st.button("Prepare Data for Modeling"):
            # Based on core_model.py
            X = df[['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 
                    'oldbalanceDest', 'newbalanceDest']]
            y = df['isFraud']
            
            st.session_state.X = X
            st.session_state.y = y
            
            st.success("Data prepared for modeling!")
    else:
        st.info("Please upload a dataset in the sidebar first.")

# Local Training tab
with tabs[2]:
    st.markdown("<h2 class='sub-header'>Local Model Training</h2>", unsafe_allow_html=True)
    
    if st.session_state.X is not None and st.session_state.y is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<h3>Training Parameters</h3>", unsafe_allow_html=True)
            
            max_iter = st.slider("Max Iterations", min_value=1, max_value=20, value=5, step=1)
            random_state = st.slider("Random State", min_value=0, max_value=100, value=42)
            test_size = st.slider("Test Size", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<h3>Train Local Model</h3>", unsafe_allow_html=True)
            
            if st.button("Train Model"):
                with st.spinner("Training local model..."):
                    try:
                        # Split data
                        X_train, X_test, y_train, y_test = train_test_split(
                            st.session_state.X, st.session_state.y, 
                            test_size=test_size, random_state=random_state
                        )
                        
                        # Train model
                        model = LogisticRegression(max_iter=max_iter, random_state=random_state)
                        model.fit(X_train, y_train)
                        
                        # Evaluate
                        preds = model.predict(X_test)
                        accuracy = accuracy_score(y_test, preds)
                        
                        st.session_state.model = model
                        st.session_state.local_accuracy = accuracy
                        
                        st.success(f"Local model trained successfully! Accuracy: {accuracy:.4f}")
                    except Exception as e:
                        st.error(f"Error training model: {str(e)}")
            
            if st.session_state.local_accuracy is not None:
                st.markdown(f"<p class='success-msg'>Local Model Accuracy: {st.session_state.local_accuracy:.4f}</p>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        if st.session_state.model is not None:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<h3>Model Evaluation</h3>", unsafe_allow_html=True)
            
            X_train, X_test, y_train, y_test = train_test_split(
                st.session_state.X, st.session_state.y, 
                test_size=test_size, random_state=random_state
            )
            
            preds = st.session_state.model.predict(X_test)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Confusion Matrix
                fig, ax = plt.subplots(figsize=(8, 6))
                cm = confusion_matrix(y_test, preds)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                           xticklabels=['Not Fraud', 'Fraud'],
                           yticklabels=['Not Fraud', 'Fraud'])
                ax.set_ylabel('True Label')
                ax.set_xlabel('Predicted Label')
                ax.set_title('Confusion Matrix')
                st.pyplot(fig)
            
            with col2:
                # Classification Report
                report = classification_report(y_test, preds, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df)
            
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("Please prepare your data for modeling first in the Data Analysis tab.")

# Federated Learning tab
with tabs[3]:
    st.markdown("<h2 class='sub-header'>Federated Learning</h2>", unsafe_allow_html=True)
    
    if st.session_state.model is not None and st.session_state.context is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<h3>Encrypt and Upload Model Weights</h3>", unsafe_allow_html=True)
            
            if st.button("Encrypt & Upload Weights"):
                with st.spinner("Encrypting and uploading model weights..."):
                    try:
                        session = st.session_state.session
                        s3 = session.client('s3')
                        model = st.session_state.model
                        context = st.session_state.context
                        
                        # Encrypt weights (based on core_model.py)
                        coef_array = np.array(model.coef_[0])
                        intercept_array = np.array([model.intercept_[0]])
                        
                        weights = np.concatenate([coef_array, intercept_array]).tolist()
                        
                        # Encrypt weights vector
                        encrypted_weights = ts.ckks_vector(context, weights)
                        serialized_weights = encrypted_weights.serialize()
                        
                        # Upload to S3
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        s3_key = f"client_weights/{client_id}/{timestamp}_weights.pkl"
                        
                        s3.put_object(
                            Bucket='fraud-detection-encrypted-weights',
                            Key=s3_key,
                            Body=serialized_weights
                        )
                        
                        # Notify aggregator
                        try:
                            response = requests.post(
                                f"{api_endpoint}/aggregator",
                                json={
                                    'client_id': client_id,
                                    's3_key': s3_key
                                }
                            )
                            
                            if response.status_code != 200:
                                st.error(f"Failed to notify aggregator: {response.status_code}")
                            else:
                                st.success(f"Weights uploaded successfully! Progress: {response.json().get('progress')}%")
                        except Exception as e:
                            st.error(f"Error notifying aggregator: {str(e)}")
                    except Exception as e:
                        st.error(f"Error encrypting/uploading weights: {str(e)}")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<h3>Download Aggregated Model</h3>", unsafe_allow_html=True)
            
            if st.button("Download Aggregated Model"):
                with st.spinner("Downloading and decrypting aggregated model..."):
                    try:
                        session = st.session_state.session
                        s3 = session.client('s3')
                        model = st.session_state.model
                        context = st.session_state.context
                        
                        # Get latest aggregated model from S3
                        response = s3.get_object(
                            Bucket='fraud-detection-encrypted-weights',
                            Key='aggregated/latest_aggregated_model.pkl'
                        )
                        encrypted_aggregated = response['Body'].read()
                        
                        # Deserialize and decrypt
                        aggregated_weights = ts.ckks_vector_from(context, encrypted_aggregated)
                        decrypted_weights = np.array(aggregated_weights.decrypt())
                        
                        # Update local model with proper array types
                        n_features = len(model.coef_[0])
                        model.coef_ = np.array([decrypted_weights[:n_features]])
                        model.intercept_ = np.array([decrypted_weights[-1]])
                        
                        # Evaluate new model
                        X_train, X_test, y_train, y_test = train_test_split(
                            st.session_state.X, st.session_state.y, 
                            test_size=0.2, random_state=42
                        )
                        
                        preds = model.predict(X_test)
                        accuracy = accuracy_score(y_test, preds)
                        
                        st.session_state.model = model
                        st.session_state.aggregated_accuracy = accuracy
                        
                        st.success(f"Aggregated model downloaded successfully! Accuracy: {accuracy:.4f}")
                    except Exception as e:
                        st.error(f"Error downloading aggregated model: {str(e)}")
            
            if st.session_state.aggregated_accuracy is not None:
                st.markdown(f"<p class='success-msg'>Aggregated Model Accuracy: {st.session_state.aggregated_accuracy:.4f}</p>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        if st.session_state.model is None:
            st.info("Please train your local model first.")
        if st.session_state.context is None:
            st.info("Please set up your encryption keys first.")

# Results tab
with tabs[4]:
    st.markdown("<h2 class='sub-header'>Results & Comparison</h2>", unsafe_allow_html=True)
    
    if st.session_state.local_accuracy is not None or st.session_state.aggregated_accuracy is not None:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3>Model Performance Comparison</h3>", unsafe_allow_html=True)
        
        comparison_data = {
            'Model': ['Local Model', 'Federated Aggregated Model'],
            'Accuracy': [
                st.session_state.local_accuracy if st.session_state.local_accuracy is not None else 0,
                st.session_state.aggregated_accuracy if st.session_state.aggregated_accuracy is not None else 0
            ]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        bar_colors = ['#2196F3', '#4CAF50']
        bars = sns.barplot(x='Model', y='Accuracy', data=comparison_df, palette=bar_colors)
        
        # Add value labels
        for i, bar in enumerate(bars.patches):
            if comparison_data['Accuracy'][i] > 0:
                bars.text(
                    bar.get_x() + bar.get_width()/2.,
                    bar.get_height() + 0.01,
                    f"{comparison_data['Accuracy'][i]:.4f}",
                    ha='center'
                )
        
        ax.set_ylim(0, 1)
        ax.set_title('Model Accuracy Comparison')
        st.pyplot(fig)
        
        # Analysis
        if st.session_state.local_accuracy is not None and st.session_state.aggregated_accuracy is not None:
            diff = st.session_state.aggregated_accuracy - st.session_state.local_accuracy
            if diff > 0:
                st.success(f"The federated model shows an improvement of {diff:.4f} in accuracy compared to your local model!")
            elif diff < 0:
                st.warning(f"The federated model shows a decrease of {abs(diff):.4f} in accuracy compared to your local model.")
            else:
                st.info("The federated model shows no change in accuracy compared to your local model.")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Model deployment options
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3>Model Deployment</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Export Model"):
                if st.session_state.model is not None:
                    # Save model to pickle file
                    with open("fraud_detection_model.pkl", "wb") as f:
                        pickle.dump(st.session_state.model, f)
                    
                    # Create a download button
                    with open("fraud_detection_model.pkl", "rb") as f:
                        st.download_button(
                            label="Download Model",
                            data=f,
                            file_name="fraud_detection_model.pkl",
                            mime="application/octet-stream"
                        )
        
        with col2:
            if st.button("Generate Inference API"):
                code = """
# Flask API for model inference
from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
with open('fraud_detection_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array([
        data['step'],
        data['amount'],
        data['oldbalanceOrg'],
        data['newbalanceOrig'],
        data['oldbalanceDest'],
        data['newbalanceDest']
    ]).reshape(1, -1)
    
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]
    
    return jsonify({
        'fraud_prediction': int(prediction),
        'fraud_probability': float(probability)
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
"""
                st.code(code, language="python")
                
                # Create a download button for the API code
                st.download_button(
                    label="Download API Code",
                    data=code,
                    file_name="fraud_detection_api.py",
                    mime="text/plain"
                )
        
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("Train your models to see performance comparison.")

# Footer
st.markdown("---")
st.markdown("Federated Fraud Detection System | Powered by Homomorphic Encryption")