from flask import Flask, request, jsonify
import tenseal as ts
import boto3
import pickle
from botocore.exceptions import ClientError
import logging

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

s3 = boto3.client('s3')

KEYS_BUCKET = 'fraud-detection-encrypted-keys'
WEIGHTS_BUCKET = 'fraud-detection-encrypted-weights'
MASTER_KEY_PATH = 'he-keys/master-key.bin'
PUBLIC_KEY_PATH = 'public-key/public-context.bin'

@app.route('/keys', methods=['POST'])
def initialize_keys():
    """Initialize global keys (only needs to be called once)"""
    try:
        s3.head_object(Bucket=KEYS_BUCKET, Key=MASTER_KEY_PATH)
        logger.info("Keys already exist")
        return jsonify({
            "status": "exists",
            "s3_key": MASTER_KEY_PATH,
            "bucket": KEYS_BUCKET
        }), 200
        
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            context = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=8192,
                coeff_mod_bit_sizes=[50, 30, 30, 50]
            )
            context.generate_galois_keys()
            context.global_scale = 2**30

            key_data = {
                'private_context': context.serialize(save_secret_key=True),
                'public_context': context.serialize(save_secret_key=False)
            }
            
            s3.put_object(
                Bucket=KEYS_BUCKET,
                Key=MASTER_KEY_PATH,
                Body=pickle.dumps(key_data)
            )
            
            s3.put_object(
                Bucket=WEIGHTS_BUCKET,
                Key=PUBLIC_KEY_PATH,
                Body=key_data['public_context']
            )
            
            logger.info("Successfully generated global keys")
            return jsonify({"status": "created",
            "s3_key": MASTER_KEY_PATH,
            "bucket": KEYS_BUCKET}), 201
        else:
            logger.error(f"S3 access error: {str(e)}")
            return jsonify({"error": "S3 access error"}), 500

@app.route('/keys', methods=['GET'])
def get_key():
    """Get the S3 key for the complete key pair"""
    try:
        s3.head_object(Bucket=KEYS_BUCKET, Key=MASTER_KEY_PATH)
        return jsonify({
            "status": "exists",
            "s3_key": MASTER_KEY_PATH,
            "bucket": KEYS_BUCKET
        }), 200
    except ClientError as e:
        return jsonify({"error": "Key pair not found"}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)