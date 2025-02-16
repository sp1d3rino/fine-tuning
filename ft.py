import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments,BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training 
from datasets import load_dataset, Dataset
import streamlit as st

# Streamlit App
st.title("LLM Fine-Tuning with LoRA")

# Function to download or load a model locally
def load_or_download_model(model_name):
    try:
        bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        )
        # Load the model and tokenizer from Hugging Face
        model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                     #load_in_8bit=True,
                                                     quantization_config=bnb_config, 
                                                     device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        st.success(f"Model and tokenizer loaded successfully!")
        return model, tokenizer
    except Exception as e:
        st.error(f"Failed to load the model: {e}")
        return None, None

# LoRA Configuration
def create_lora_config():
    return LoraConfig(
        r=8,  # Rank of the low-rank matrices
        lora_alpha=32,  # Scaling factor
        
        target_modules=None,  # Target modules for LoRA
        lora_dropout=0.1,  # Dropout rate
        bias="none",  # Bias handling
        task_type="CAUSAL_LM"  # Task type
    )

# Prepare Dataset
def prepare_dataset(dataset_path, tokenizer):
    # Load dataset from local path
    if dataset_path.endswith(".json"):
        dataset = Dataset.from_json(dataset_path)
    elif dataset_path.endswith(".csv"):
        dataset = Dataset.from_csv(dataset_path)
    elif dataset_path.endswith(".txt"):
        try:
            # Read the text file with UTF-8 encoding
            with open(dataset_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            dataset = Dataset.from_dict({"text": lines})
        except UnicodeDecodeError:
            st.error("Failed to read the text file. Please ensure the file is encoded in UTF-8.")
            return None
    else:
        st.error("Unsupported dataset format. Use .json, .csv, or .txt.")
        return None

    # Tokenize the dataset
    def tokenize_function(examples):
        # Tokenize the input text and create labels for causal language modeling
        tokenized_inputs = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
        tokenized_inputs["labels"] = tokenized_inputs["input_ids"]  # Use input_ids as labels for causal LM
        return tokenized_inputs
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset

# Fine-Tuning Function
def fine_tune_model(model, tokenizer, dataset, output_dir):
    # Prepare model for LoRA fine-tuning
    model = prepare_model_for_kbit_training(model)
    lora_config = create_lora_config()
    model = get_peft_model(model, lora_config)

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        num_train_epochs=3,
        logging_dir="./logs",
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        fp16=True,  # Use mixed precision
        push_to_hub=False,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    # Fine-Tune
    trainer.train()

    # Save Model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    st.success(f"Model saved to {output_dir}")

# Chat Function
def chat_with_model(model, tokenizer):
    st.subheader("Chat with the Model")
    user_input = st.text_input("You: ", "")
    if user_input:
        # Generate a response
        inputs = tokenizer(user_input, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_length=100, num_return_sequences=1)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.text_area("Model:", value=response, height=100)

# Streamlit UI
def main():
    st.sidebar.header("Configuration")
    model_name = st.sidebar.text_input("Model Name (Hugging Face Repo ID)", "gpt2")
    dataset_path = st.sidebar.text_input("Local Dataset Path (e.g., ./data/dataset.txt)", "./dataset/book1.txt")
    output_dir = st.sidebar.text_input("Output Directory", "./fine_tuned_model")

    if st.sidebar.button("Load Model and Dataset"):
        with st.spinner("Loading model and dataset..."):
            # Load or download the model
            model, tokenizer = load_or_download_model(model_name)
            if model is None or tokenizer is None:
                st.error("Failed to load or download the model.")
                return

            # Set a padding token if not already defined
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token  # Use EOS token as padding token
                st.warning(f"Padding token not found. Using EOS token ({tokenizer.eos_token}) as padding token.")

            # Load and prepare the dataset
            dataset = prepare_dataset(dataset_path, tokenizer)
            if dataset is None:
                st.error("Failed to load dataset.")
                return
            st.success("Model and dataset loaded successfully!")
            st.session_state["base_model"] = model
            st.session_state["tokenizer"] = tokenizer
            st.session_state["dataset"] = dataset

    if st.sidebar.button("Start Fine-Tuning"):
        if "base_model" not in st.session_state or "dataset" not in st.session_state:
            st.error("Please load the model and dataset first.")
        else:
            with st.spinner("Fine-tuning in progress..."):
                fine_tune_model(
                    st.session_state["base_model"],
                    st.session_state["tokenizer"],
                    st.session_state["dataset"],
                    output_dir
                )
                st.session_state["fine_tuned_model"] = True  # Mark fine-tuning as completed

    # Chat with Base Model
    if st.sidebar.button("Chat Base Model"):
        if "base_model" not in st.session_state:
            st.error("Please load the base model first.")
        else:
            st.session_state["chat_mode"] = "base"
            st.session_state["current_model"] = st.session_state["base_model"]

    # Chat with Fine-Tuned Model
    if st.sidebar.button("Chat Fine-Tuned Model", disabled="fine_tuned_model" not in st.session_state):
        if "fine_tuned_model" not in st.session_state:
            st.error("Please complete fine-tuning first.")
        else:
            # Load the fine-tuned model
            fine_tuned_model = AutoModelForCausalLM.from_pretrained(output_dir, device_map="auto")
            fine_tuned_tokenizer = AutoTokenizer.from_pretrained(output_dir)
            st.session_state["chat_mode"] = "fine_tuned"
            st.session_state["current_model"] = fine_tuned_model
            st.session_state["current_tokenizer"] = fine_tuned_tokenizer

    # Chat Interface
    if "chat_mode" in st.session_state:
        if st.session_state["chat_mode"] == "base":
            st.subheader("Chatting with Base Model")
            chat_with_model(st.session_state["base_model"], st.session_state["tokenizer"])
        elif st.session_state["chat_mode"] == "fine_tuned":
            st.subheader("Chatting with Fine-Tuned Model")
            chat_with_model(st.session_state["current_model"], st.session_state["current_tokenizer"])

if __name__ == "__main__":
    main()