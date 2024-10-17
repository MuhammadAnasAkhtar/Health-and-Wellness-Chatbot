from flask import Flask, render_template, request, jsonify
from langchain_community.llms import HuggingFaceHub  # Updated import
from dotenv import load_dotenv
import os

load_dotenv()  # Load Hugging Face API token from .env file

app = Flask(__name__)

# Load Hugging Face API token
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Initialize the LLM from Hugging Face Hub with the Llama-3.1 model
llm = HuggingFaceHub(
    repo_id="meta-llama/Llama-3.2-1B",  # Specify your model here
    model_kwargs={"temperature": 0.7, "max_length": 150},  # Adjust parameters as needed
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
)

@app.route("/")
def index():
    return render_template('chat.html')  # Create a simple HTML page for interaction

@app.route("/get", methods=["POST"])
def chat():
    user_message = request.form["msg"]  # Get the user's message
    response = llm(user_message)  # Get the model's response
    return jsonify({"reply": response.strip()})  # Send back the reply

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
    
