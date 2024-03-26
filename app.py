import streamlit as st
from pensieve.audio import transcribe_audio
from pensieve.llm import load_model, generate_response

st.title("Pensieve: Your AI Memory Companion")

if "memory_logs" not in st.session_state:
    st.session_state.memory_logs = []

# Load the model and tokenizer
tokenizer, model = load_model()

# Sidebar
with st.sidebar:
    st.header("Settings")
    max_length = st.slider("Max Response Length", 50, 200, 100)

# Main content
st.header("Record a Memory")
if st.button("Start Recording"):
    text = transcribe_audio()  # Call the function to get the transcribed text
    if text:
        st.session_state.memory_logs.append(text)
        st.success(f"Memory recorded: {text}")

st.header("Chat with Your AI Companion")
user_input = st.text_input("You:")
if user_input:
    response = generate_response(model, tokenizer, user_input, max_length)
    st.markdown(f"**Pensieve:** {response}")

st.header("Memory Logs")
for log in st.session_state.memory_logs:
    st.write(f"- {log}")