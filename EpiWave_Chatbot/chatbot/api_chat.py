from flask import Flask, request, jsonify
from flask_cors import CORS
from agents.respond import stream_response
from agents.retrieve import retrieve_chunks
from agents.verify import verify_chunks
from rag.vectorize import _get_chroma_collection

app = Flask(__name__)

# Enable CORS for all origins - more explicit
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/chat', methods=['POST', 'OPTIONS'])
def chat():
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        return '', 200
        
    data = request.json
    user_query = data.get('query', '')
    history = data.get('history', [])
    
    has_scans = _get_chroma_collection().count() > 0
    retrieved = retrieve_chunks(user_query)
    verified = verify_chunks(retrieved)
    
    response_text = ""
    for token in stream_response(verified, history, user_query, has_scans=has_scans):
        response_text += token
    
    return jsonify({'response': response_text})

@app.route('/', methods=['GET'])
def home():
    return "EpiWave AI Server is running!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)