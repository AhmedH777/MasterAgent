from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from master_agent.Agent import Agent
import logging
import time
from queue import Queue

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
agent = Agent(model="gpt-4-turbo",
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
    model_type = data.get("model", "gpt-4-turbo")

    log_with_source("SYSTEM", f"Received message: {user_message} with model: {model_type}")

    if not user_message:
        log_with_source("SYSTEM", "Received empty message")
        return jsonify({"error": "Empty message"}), 400

    if agent.get_model() != model_type:
        agent.set_model(model_type)
        log_with_source("AGENT", f"Switched model to {model_type}")

    bot_reply = agent.chat(user_message)
    log_with_source("LLM", f"Bot reply: {bot_reply}")

    return jsonify({"response": bot_reply})


@app.route('/api/logs', methods=['GET'])
def stream_logs():
    def log_stream():
        while True:
            log_message = log_queue.get()
            yield f"data: {log_message}\n\n"
    return Response(log_stream(), mimetype='text/event-stream')


if __name__ == '__main__':
    app.run(debug=True, port=5000)