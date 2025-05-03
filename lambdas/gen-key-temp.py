import boto3
import pickle
import os
import sys
import datetime
from botocore.exceptions import ClientError
import tenseal as ts
import json
import tenseal as ts


from dotenv import load_dotenv, dotenv_values 


session = boto3.Session(profile_name='client1_user_5590')

#s3 = boto3.client('s3')

s3 = session.client('s3')



def generate_keys(client_id):

    print("Generating new HE keys...")
    context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[50, 30, 30, 50])
    context.generate_galois_keys()
    context.global_scale = 2**30

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

    key = f'he-keys/{client_id}.bin'
    
    return key

def get_s3_he_keys(client_id):
    key = f'he-keys/{client_id}.bin'
    
    return key


generate_keys(12)