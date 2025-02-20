import pandas as pd
import numpy as np
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, TrainingArguments, Trainer
import evaluate
import os
import json
import logging
import torch
from sklearn.preprocessing import LabelEncoder

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_dataset(dataset):
    # Convert sentiment labels to numerical values
    label_encoder = LabelEncoder()
    
    # Extract text and labels
    texts = dataset['text']
    # Assuming 'sentiment' is your label column - adjust if different
    labels = label_encoder.fit_transform(dataset['label_text'])
    
    return texts, labels, label_encoder

def tokenize_function(examples, tokenizer, max_length=128):
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

def main():
    # Load dataset
    logger.info("Loading dataset...")
    dataset = load_dataset("mteb/tweet_sentiment_extraction")
    
    # Initialize tokenizer
    logger.info("Initializing tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    # Add padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Prepare datasets
    train_texts, train_labels, label_encoder = prepare_dataset(dataset['train'])
    eval_texts, eval_labels, _ = prepare_dataset(dataset['test'])
    
    # Create complete datasets with texts and labels
    train_dataset = {
        'text': train_texts,
        'labels': train_labels
    }
    eval_dataset = {
        'text': eval_texts,
        'labels': eval_labels
    }
    
    # Tokenize datasets
    logger.info("Tokenizing datasets...")
    train_encodings = tokenize_function(train_dataset, tokenizer)
    eval_encodings = tokenize_function(eval_dataset, tokenizer)
    
    # Create torch datasets
    class SentimentDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = SentimentDataset(train_encodings, train_labels)
    eval_dataset = SentimentDataset(eval_encodings, eval_labels)
    
    # Initialize model
    logger.info("Initializing model...")
    num_labels = len(label_encoder.classes_)
    model = GPT2ForSequenceClassification.from_pretrained(
        "gpt2", 
        num_labels=num_labels,
        pad_token_id=tokenizer.pad_token_id
    )
    
    # Resize token embeddings
    model.resize_token_embeddings(len(tokenizer))
    
    # Setup metrics
    metric = evaluate.load("accuracy")
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="test_trainer",
        evaluation_strategy="steps",
        eval_steps=500,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        num_train_epochs=3,
        weight_decay=0.01,
        learning_rate=2e-5,
        warmup_steps=500,
        save_strategy="steps",
        save_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_dir="./logs",
        logging_steps=100,
    )
    
    # Initialize trainer
    logger.info("Setting up trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
    
    # Train and evaluate
    logger.info("Starting training...")
    trainer.train()
    
    logger.info("Evaluating model...")
    eval_results = trainer.evaluate()
    logger.info(f"Evaluation results: {eval_results}")
    
    # Save everything
    output_dir = "Fine_Tuned_Models"
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Saving model and tokenizer to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save label mapping
    label_map = {i: label for i, label in enumerate(label_encoder.classes_)}
    with open(os.path.join(output_dir, "label_map.json"), "w") as f:
        json.dump(label_map, f)
    
    logger.info("Training complete! Model and related files saved.")

if __name__ == "__main__":
    main()