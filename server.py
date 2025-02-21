from flask import Flask, request, jsonify, Response, send_from_directory
from master_agent.Agent import Agent
from flask_cors import CORS
from queue import Queue
import tempfile
import logging
import json
import os
import re

app = Flask(__name__)
CORS(app)

# Create a queue to hold log messages
log_queue = Queue()

# Custom logger
class QueueHandler(logging.Handler):
    def emit(self, record):
        log_entry = self.format(record)
        log_queue.put(log_entry)

logger = logging.getLogger("app_logger")
logger.setLevel(logging.DEBUG)
queue_handler = QueueHandler()
queue_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(queue_handler)

# Custom logging functions with source tagging
def log_with_source(source, message):
    logger.info(f"[{source}] {message}")

# Initialize the Agent
agent = Agent(model="gpt-4o",
              max_memory_size=30,
              summary_trigger=10,
              preserve_last_n_context=4,
              role="assistant",
              description="You are a helpful AI assistant.",
              logger=logger)

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get("message", "")
    model_type = data.get("model", "gpt-4o")

    log_with_source("SYSTEM", f"Received message: {user_message} with model: {model_type}")

    if not user_message:
        log_with_source("SYSTEM", "Received empty message")
        return jsonify({"error": "Empty message"}), 400

    if agent.get_model() != model_type:
        agent.set_model(model_type)
        log_with_source("AGENT", f"Switched model to {model_type}")

    try:
        bot_reply = agent.chat(user_message)
        log_with_source("LLM", f"Bot reply: {bot_reply}")

        # Directly return the bot reply without JSON parsing
        return jsonify({"response": bot_reply})

    except Exception as e:
        log_with_source("LLM", f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/logs', methods=['GET'])
def stream_logs():
    def log_stream():
        while True:
            log_message = log_queue.get()
            yield f"data: {log_message}\n\n"
    return Response(log_stream(), mimetype='text/event-stream')

@app.route('/api/save_memory', methods=['POST'])
def save_memory():
    try:
        log_with_source("AGENT", "Saving conversation memory...")
        
        # Assuming the agent has a method to save memory
        agent.end_chat()

        # Clean Project Temp Directory
        project_dir = os.path.dirname(os.path.abspath(__file__))
        temp_dir = os.path.join(project_dir, 'temp')

        for filename in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

        return jsonify({"status": "success", "message": "Memory saved successfully."})
    except Exception as e:
        log_with_source("AGENT", f"Error saving memory: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

# Serve the generated plot image
@app.route('/temp/<filename>')
def serve_temp_file(filename):
    # Serve from project 'temp' folder
    project_dir = os.path.dirname(os.path.abspath(__file__))
    temp_dir = os.path.join(project_dir, 'temp')
    return send_from_directory(temp_dir, filename)


if __name__ == '__main__':
    app.run(debug=True, port=5000)