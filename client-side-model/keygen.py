import boto3
import pickle
import tenseal as ts  # For homomorphic encryption
import os
import sys
import datetime
from botocore.exceptions import ClientError

from dotenv import load_dotenv, dotenv_values 


s3 = boto3.client('s3')
secrets_manager = boto3.client('secretsmanager')
api_gateway = boto3.client('apigatewaymanagementapi', 
                                endpoint_url=os.getenv('API_GW_ENDPOINT'))


def generate_keys(client_id):

    print("Generating new HE keys...")
    context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[50, 30, 30, 50])
    context.generate_galois_keys()
    context.global_scale = 2**30

    # Store private context (should be kept secure)
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
    
    return context

def retrieve_he_keys(client_id):
    s3 = boto3.client('s3')
    
    response = s3.get_object(
        Bucket='fraud-detection-encrypted-keys',
        Key=f'he-keys/{client_id}.bin'
    )
    
    key_data = pickle.loads(response['Body'].read())
    
    return ts.context_from(key_data['private_context'])