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
from botocore.exceptions import ClientError

from dotenv import load_dotenv, dotenv_values 

from keygen import generate_keys, retrieve_he_keys


load_dotenv() 

class FederatedClient:
    def __init__(self, session, client_id, data_path):
        self.client_id = client_id
        self.data_path = data_path
        self.model = None
        self.context = None  # Homomorphic encryption context
        
        # Initialize AWS clients
        self.s3 = session.client('s3')
        self.secrets_manager = session.client('secretsmanager')
        self.api_gateway = session.client('apigatewaymanagementapi', 
                                       endpoint_url=os.getenv('API_GW_ENDPOINT'))
        
        # Load or create HE keys
        self._setup_cryptography()
        
    def _setup_cryptography(self):
        """Load or create homomorphic encryption keys"""
        try:
            # Try to load existing keys from Secrets Manager
            self.context = retrieve_he_keys(self.client_id)
            
        except ClientError as e:
            self.context = generate_keys(self.client_id)
                
    def load_data(self):
        """Load client's local data"""
        df = pd.read_csv(self.data_path)
        self.X = df[['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 
                    'oldbalanceDest', 'newbalanceDest']]
        self.y = df['isFraud']
    
    def train_local_model(self):
        """Train logistic regression model on local data"""
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        self.model.fit(self.X, self.y)
        
        # Evaluate local model
        preds = self.model.predict(self.X)
        accuracy = accuracy_score(self.y, preds)
        print(f"Local model accuracy: {accuracy:.4f}")
                
    
    def encrypt_weights(self):
        """Encrypt model weights using HE"""
        coef_array = np.array(self.model.coef_[0])
        intercept_array = np.array([self.model.intercept_[0]])
        
        # Combine and convert to list
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
        
    
    def download_aggregated_model(self):
        """Download and decrypt aggregated model"""
        # Get latest aggregated model from S3
        response = self.s3.get_object(
            Bucket='fraud-detection-encrypted-weights',
            Key='aggregated/latest_aggregated_model.pkl'
        )
        encrypted_aggregated = response['Body'].read()
        
        # Deserialize and decrypt
        aggregated_weights = ts.ckks_vector_from(self.context, encrypted_aggregated)
        decrypted_weights = np.array(aggregated_weights.decrypt())  # Convert to NumPy array
        
        # Update local model with proper array types
        n_features = len(self.model.coef_[0])
        self.model.coef_ = np.array([decrypted_weights[:n_features]])  # 2D array
        self.model.intercept_ = np.array([decrypted_weights[-1]])



def main():
    session = boto3.Session(profile_name='client1_user_5590')
    
    client = FederatedClient(session, client_id=1, data_path="../user1.csv")
    
    client.load_data()
    client.train_local_model()
    encrypted_weights = client.encrypt_weights()
    
    client.upload_weights(encrypted_weights)
    
    
    client.s3.put_object(
        Bucket='fraud-detection-encrypted-weights',
        Key='aggregated/latest_aggregated_model.pkl',
        Body=encrypted_weights
    )
    
    time.sleep(15)
    
    
    client.download_aggregated_model()
    # Evaluate new model
    preds = client.model.predict(client.X)
    accuracy = accuracy_score(client.y, preds)
    print(f"Updated model accuracy: {accuracy:.4f}")
    

main()
    