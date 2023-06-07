from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import subprocess
import os
from cnn.test import runTest

def save_matrix(matrix):
    directory = './cnn'
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, 'matrix_data.json')
    with open(file_path, 'w') as file:
        json.dump(matrix, file)

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})  # Allow all origins for /api routes

@app.route('/api/matrix', methods=['POST'])
def receive_matrix():
    matrix = request.get_json()
    print(matrix)
    # save_matrix(matrix)
    # subprocess.run(["python3", "./cnn/test.py"])
    runTest(matrix)
    return jsonify({'message': 'Matrix received successfully'}), 200

if __name__ == '__main__':
    app.run(debug=True, port=5000)
