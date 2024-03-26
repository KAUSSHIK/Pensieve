import os
import json
import streamlit as st
from pensieve.audio import transcribe_audio
from pensieve.llm import load_model, generate_response, train_model

st.title("Pensieve: Your AI Memory Companion")

# Set up logging directory
logs_dir = "data/logs"
os.makedirs(logs_dir, exist_ok=True)

# Load the model and tokenizer
tokenizer, model = load_model()

# Load existing logs from the logs directory
logs = []
for log_file in os.listdir(logs_dir):
    with open(os.path.join(logs_dir, log_file), "r") as f:
        log_data = json.load(f)
        logs.append(log_data["text"])

# Sidebar
with st.sidebar:
    st.header("Settings")
    max_length = st.slider("Max Response Length", 50, 200, 100)
    train_button = st.button("Train Model")

# Main content
st.header("Record a Memory")
input_method = st.radio("Select Input Method", ("Speak", "Type"))

if input_method == "Speak":
    if st.button("Start Recording"):
        text = transcribe_audio()
        if text:
            log_entry = {"text": text}
            log_filename = f"log_{len(logs)}.json"
            with open(os.path.join(logs_dir, log_filename), "w") as f:
                json.dump(log_entry, f)
            logs.append(text)
            st.success(f"Memory recorded: {text}")
else:
    text_input = st.text_area("Type your memory:")
    if st.button("Save Memory"):
        if text_input:
            log_entry = {"text": text_input}
            log_filename = f"log_{len(logs)}.json"
            with open(os.path.join(logs_dir, log_filename), "w") as f:
                json.dump(log_entry, f)
            logs.append(text_input)
            st.success(f"Memory recorded: {text_input}")

st.header("Chat with Your AI Companion")
user_input = st.text_input("You:")
if user_input:
    response = generate_response(model, tokenizer, f"{user_input} {' '.join(logs)}", max_length)
    st.markdown(f"**Pensieve:** {response}")

st.header("Memory Logs")
for log in logs:
    st.write(f"- {log}")

# Train the model if the train button is clicked
if train_button:
    train_model(model, tokenizer, logs)
    st.success("Model training completed.")