from flask import Flask, request, jsonify
from flask_cors import CORS
from master_agent.Agent import Agent

# Initialize the Agent
agent = Agent(model="gpt-4-turbo",
              max_memory_context_buffer=10,
              role="assistant",
              description="You are a helpful AI assistant.")

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
    
    ################################################################
    ####################### Deploy the Agent #######################
    ################################################################
    # Check if agent model is the same as the user model
    if agent.get_model() != model_type:
        agent.set_model(model_type)
    
    # Chat with the user
    bot_reply = agent.chat(user_message)
    ################################################################

    return jsonify({"response": bot_reply})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
