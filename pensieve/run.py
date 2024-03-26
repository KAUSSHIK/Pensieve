import os
import json
from pensieve.audio import transcribe_audio
from pensieve.llm import load_model, generate_response, train_model

# Instructiosn to run the code and use the AI-powered personal journal:
# 1. Run the code snippet in a Python environment.
# 2. Speak into the microphone to record your thoughts.
# 3. Choose whether to train the model or ask a question.
# 4. If training, wait for the model to complete training.
# 5. If asking a question, enter your question and receive a response.
# 6. Continue recording thoughts and interacting with the AI companion.
# 7. To stop the program, press Ctrl+C or close the Python environment.

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

print("Pensieve: An AI-powered personal journal. Start recording your thoughts.")

while True:
    # Transcribe user's audio input
    text = transcribe_audio()
    print(f"Transcribed text: {text}")

    # Save the transcribed text as a log entry
    log_entry = {"text": text}
    log_filename = f"log_{len(logs)}.json"
    with open(os.path.join(logs_dir, log_filename), "w") as f:
        json.dump(log_entry, f)
    logs.append(text)

    # Check if the user wants to train the model or ask a question
    user_input = input("Do you want to train the model or ask a question? (train(T or t)/ask(A or a)): ")

    if user_input.lower() == "train" or user_input.lower() == "t":
        # Fine-tune the model on the updated logs
        train_model(model, tokenizer, logs)
        print("Model training completed. Check back later to ask questions.")
    elif user_input.lower() == "ask" or user_input.lower() == "a":
        # Ask the user for a question
        question = input("What would you like to ask about your past experiences? ")

        # Generate a response based on the question and logs
        response = generate_response(model, tokenizer, f"{question} {' '.join(logs)}")
        print(f"Pensieve: {response}")
    else:
        print("Invalid input. Please enter 'train' or 'ask'.")