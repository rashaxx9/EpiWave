from flask import Flask, request, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/chat', methods=['POST', 'OPTIONS'])
def chat():
    if request.method == 'OPTIONS':
        return '', 200
        
    data = request.json
    user_query = data.get('query', '')
    
    # Return a simple response without any AI logic
    return jsonify({'response': f"You said: {user_query}"})

@app.route('/health', methods=['GET'])
def health():
    return {"status": "ok"}, 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False)
