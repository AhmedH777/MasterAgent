from flask import Flask, request, jsonify
from flask_cors import CORS
from master_agent.chat import parse_message

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get("message", "")
    model_type = data.get("model", "llama3.2")  # Default to llama3.2

    # Empty Input
    if not user_message:
        return jsonify({"error": "Empty message"}), 400
    
    # Master Agent Response
    bot_reply = parse_message(user_message, model=model_type)

    return jsonify({"response": bot_reply})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
