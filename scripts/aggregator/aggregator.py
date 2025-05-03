from flask import Flask, request, jsonify
import tenseal as ts
import boto3
import numpy as np
from botocore.exceptions import ClientError
import logging
from datetime import datetime, timedelta
import requests
import os

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EXPECTED_CLIENTS = {"client1", "client2"}
WEIGHTS_BUCKET = "fraud-detection-encrypted-weights"
API_ENDPOINT = os.getenv("API_ENDPOINT")

received_updates = set()
s3 = boto3.client("s3")


class Aggregator:
    def __init__(self):
        self.context = None
        self.setup_cryptography()
        
    def setup_cryptography(self):
        """Get public context from key service"""
        try:
            response = requests.get(f"{API_ENDPOINT}/keys/aggregator")
            self.context = ts.context_from(response.content)
        except Exception as e:
            logger.error(f"Failed to get public context: {str(e)}")
            raise

    def aggregate_weights(self, client_keys):
        """Aggregate weights from specific client uploads"""
        if not client_keys:
            return False
            
        # Initialize with first client's weights
        first_obj = s3.get_object(Bucket=WEIGHTS_BUCKET, Key=client_keys[0])
        first_weights = ts.ckks_vector_from(self.context, first_obj['Body'].read())
        aggregated = np.array(first_weights.decrypt())
        
        # Aggregate remaining weights
        for key in client_keys[1:]:
            enc_weights = s3.get_object(Bucket=WEIGHTS_BUCKET, Key=key)
            weights = ts.ckks_vector_from(self.context, enc_weights['Body'].read())
            aggregated += np.array(weights.decrypt())
        
        aggregated /= len(client_keys)
        
        # Re-encrypt with public context
        enc_aggregated = ts.ckks_vector(self.context, aggregated.tolist())
        
        # Upload new model
        s3.put_object(
            Bucket=WEIGHTS_BUCKET,
            Key='aggregated/latest_aggregated_model.pkl',
            Body=enc_aggregated.serialize()
        )
        return True

@app.route('/update', methods=['POST'])
def client_update():
    """Endpoint for clients to notify they've uploaded weights"""
    global received_updates
    
    try:
        data = request.json
        client_id = data.get('client_id')
        s3_key = data.get('s3_key')
        
        if not client_id or not s3_key:
            return jsonify({"error": "Missing client_id or s3_key"}), 400
            
        if client_id not in EXPECTED_CLIENTS:
            return jsonify({"error": "Unknown client"}), 400
        
        received_updates.add((client_id, s3_key))
        logger.info(f"Received update from {client_id} (Total: {len(received_updates)}/{len(EXPECTED_CLIENTS)})")
        
        # Check if all clients have reported
        if {client for client, _ in received_updates} == EXPECTED_CLIENTS:
            logger.info("All clients reported - starting aggregation")
            aggregator = Aggregator()
            
            # Get all the S3 keys for this round
            client_keys = [key for _, key in received_updates]
            success = aggregator.aggregate_weights(client_keys)
            
            # Reset for next round
            received_updates = set()
            
            return jsonify({
                "status": "aggregation_complete",
                "success": success
            })
            
        return jsonify({
            "status": "update_received",
            "progress": f"{len({client for client, _ in received_updates})}/{len(EXPECTED_CLIENTS)}"
        })
        
    except Exception as e:
        logger.error(f"Error processing update: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)