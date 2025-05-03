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

S3_BUCKET = 'fraud-detection-encrypted-keys'
KEY_PREFIX = 'he-keys/'

@app.route('/keys', methods=['POST'])
def generate_keys():
    
    client_id = request.json.get('client_id')
    if not client_id:
        logger.error("Client ID not provided")
        return jsonify({"error": "client_id is required"}), 400
    
    s3_key = f'{KEY_PREFIX}{client_id}.bin'
    
    try:
        s3.head_object(Bucket=S3_BUCKET, Key=s3_key)
        logger.info(f"Keys already exist for client: {client_id}")
        return jsonify({
            "status": "exists",
            "message": "Key already exists",
            "s3_key": s3_key
        }), 200
        
    except ClientError as e:
        
        logger.info(f"Generating new keys for client: {client_id}")
        
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
            Bucket=S3_BUCKET,
            Key=s3_key,
            Body=pickle.dumps(key_data),
        )
        
        logger.info(f"Successfully stored keys for client: {client_id}")
        return jsonify({
            "status": "created",
            "message": "New keys generated",
            "s3_key": s3_key
        }), 201
        
    except Exception as e:
        logger.error(f"Key generation failed: {str(e)}", exc_info=True)
        return jsonify({
            "error": f"Key generation failed: {str(e)}"
        }), 500
        
@app.route('/keys', methods=['GET'])
def get_key_metadata():
    
    client_id = request.args.get('client_id')
    
    try:
        s3_key = f'{KEY_PREFIX}{client_id}.bin'
        s3.head_object(Bucket=S3_BUCKET, Key=s3_key)
        
        return jsonify({
            "status": "exists",
            "s3_key": s3_key
        }), 200
        
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            return jsonify({
                "status": "not_found",
                "s3_key": s3_key
            }), 404
        else:
            return jsonify({
                "status": "error",
                "s3_key": s3_key,
                "message": "S3 access error"
            }), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)