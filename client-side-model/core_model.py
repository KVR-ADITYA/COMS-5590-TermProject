import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import boto3
import pickle
import tenseal as ts  # For homomorphic encryption
import os
import datetime
import time
import numpy as np
import json
import requests
import argparse
from botocore.exceptions import ClientError

from dotenv import load_dotenv, dotenv_values 


load_dotenv()

API_ENDPOINT = os.getenv("API_ENDPOINT")

class FederatedClient:
    def __init__(self, session, client_id, data_path):
        self.client_id = client_id
        self.data_path = data_path
        self.model = None
        self.context = None  # Homomorphic encryption context
        
        # Initialize AWS clients
        self.s3 = session.client('s3')
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        
        # Load or create HE keys
        self.setup_cryptography()
        
    def setup_cryptography(self):
        """Load or create homomorphic encryption keys"""
        try:
            self.context = self.get_he_keys()
            
        except Exception as e:
            self.context = self.create_he_keys()
    
    def get_he_keys(self):
    
        response = requests.get(f"{API_ENDPOINT}/keys/client")
        
        if response.status_code != 200:
            raise Exception("Failed to get key reference")
                
        # fraud-detection-encrypted-keys
        s3_obj = self.s3.get_object(
            Bucket=response.json()["bucket"],
            Key=response.json()["s3_key"]
        )
        
        key_data = pickle.loads(s3_obj['Body'].read())
        
        # Contains both private and public
        return ts.context_from(key_data['private_context'])

    def create_he_keys(self):
        
        response = requests.post(f"{API_ENDPOINT}/keys/client")
        
        # fraud-detection-encrypted-keys
        s3_obj = self.s3.get_object(
            Bucket=response.json()["bucket"],
            Key=response.json()["s3_key"]
        )
        
        key_data = pickle.loads(s3_obj['Body'].read())
        
        return ts.context_from(key_data['private_context'])
                
    def load_data(self):
        """Load client's local data"""
        df = pd.read_csv(self.data_path)
        self.X = df[['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 
                    'oldbalanceDest', 'newbalanceDest']]
        self.y = df['isFraud']
    
    def train_local_model(self):
        self.model.fit(self.X, self.y)
        
        # Evaluate local model
        preds = self.model.predict(self.X)
        accuracy = accuracy_score(self.y, preds)
        print(f"Local model accuracy: {accuracy:.4f}")
    
    def train_downloaded_model(self):
        self.model.fit(self.X, self.y)
        
        preds = self.model.predict(self.X)
        accuracy = accuracy_score(self.y, preds)
        print(f"Updated model accuracy: {accuracy:.4f}")
    
    def train_or_update_model(self):
        """Try to download latest model first, fall back to new training"""
        try:
            # Attempt to download and load aggregated model
            self.download_aggregated_model()
            print("Loaded aggregated model as starting point")
            
            # Fine-tune the downloaded model
            self.train_downloaded_model()
            
            print("Fine-tuned existing model on local data")
            
        except Exception as e:
            print(f"No aggregated model found ({str(e)}), training new model")
            self.train_local_model()
                
    
    def encrypt_weights(self):
        """Encrypt model weights using HE"""
        coef_array = np.array(self.model.coef_[0])
        intercept_array = np.array([self.model.intercept_[0]])
        
        weights = np.concatenate([coef_array, intercept_array]).tolist()
        
        # Encrypt weights vector
        encrypted_weights = ts.ckks_vector(self.context, weights)
        return encrypted_weights.serialize()
    
    def upload_weights(self, encrypted_weights):
        """Upload encrypted weights to aggregator"""
        # Store encrypted weights in S3
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        s3_key = f"client_weights/{self.client_id}/{timestamp}_weights.pkl"
        
        self.s3.put_object(
            Bucket='fraud-detection-encrypted-weights',
            Key=s3_key,
            Body=encrypted_weights
        )
        
        try:
            response = requests.post(
                f"{API_ENDPOINT}/aggregator",
                json={
                    'client_id': self.client_id,
                    's3_key': s3_key
                }
            )
            if response.status_code != 200:
                print(f"Failed to notify aggregator")
            else:
                print(f"Update successful. Progress: {response.json().get('progress')}")
        except Exception as e:
            print(f"Error notifying aggregator: {str(e)}")
        
    
    def download_aggregated_model(self):
        """Download and decrypt aggregated model"""
        # Get latest aggregated model from S3
        
        try:
            response = self.s3.get_object(
                Bucket='fraud-detection-encrypted-weights',
                Key='aggregated/latest_aggregated_model.pkl'
            )
            encrypted = response['Body'].read()
            
            weights = ts.ckks_vector_from(self.context, encrypted)
            decrypted = np.array(weights.decrypt())
            
            n_features = len(self.X.columns)
            self.model.coef_ = np.array([decrypted[:n_features]])
            self.model.intercept_ = np.array([decrypted[-1:]])
        except Exception as e:
            raise Exception(f"Download failed: {str(e)}")


def main(client_id, data_path):
    session = boto3.Session(profile_name='client1_user_5590')
    
    client = FederatedClient(session, client_id, data_path=data_path)
    
    client.load_data()
    
    # Try to load aggregated model first
    client.train_or_update_model()
    
    encrypted_weights = client.encrypt_weights()
    client.upload_weights(encrypted_weights)
    

parser = argparse.ArgumentParser()
parser.add_argument('--client_id', type=str, required=True, help="Client ID for sending updates")
parser.add_argument('--data', type=str, required=True, help="CSV file path")
args = parser.parse_args()
main(args.client_id, args.data)
    