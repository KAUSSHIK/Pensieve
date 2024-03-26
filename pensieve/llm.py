from transformers import GPT2TokenizerFast, TFGPT2LMHeadModel # Hugging Face GPT-2
import tensorflow as tf
from transformers import DefaultDataCollator # Hugging Face data collator

# Load GPT-2 model and tokenizer
def load_model(model_name="gpt2"):
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name, padding_side='left')
    model = TFGPT2LMHeadModel.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token # Set padding token to end-of-sequence token
    return tokenizer, model

# Preprocess text for GPT-2
def preprocess_text(text, tokenizer, max_length=128): # max_length is the maximum number of tokens (words) in the input
    input_ids = tokenizer.encode(text, max_length=max_length, truncation=True, padding="max_length")
    return input_ids

# Generate response from GPT-2 model
def generate_response(model, tokenizer, prompt, max_length=100):
    input_ids = preprocess_text(prompt, tokenizer)
    input_ids = tf.constant([input_ids])
    attention_mask = tf.ones_like(input_ids, dtype=tf.int32)

    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        pad_token_id=tokenizer.pad_token_id,
        max_length=max_length,
        max_new_tokens=max_length,
        num_return_sequences=1,
    )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

def create_dataset(logs, tokenizer, max_length=128, batch_size=4): # batch_size is the number of logs to process in parallel
    input_ids = [preprocess_text(log, tokenizer, max_length) for log in logs]
    labels = input_ids.copy()
    dataset = tf.data.Dataset.from_tensor_slices((input_ids, labels))
    dataset = dataset.shuffle(len(input_ids)).batch(batch_size)
    return dataset

def train_model(model, tokenizer, logs, epochs=3):
    dataset = create_dataset(logs, tokenizer)
    data_collator = DefaultDataCollator(return_tensors="tf")
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
    model.compile(optimizer=optimizer)
    
    model.fit(dataset, epochs=epochs, data_collator=data_collator)
    model.save_pretrained("data/models/pensieve")
    tokenizer.save_pretrained("data/models/pensieve")