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
from huggingface_hub import HfFolder
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_dataset(dataset):
    # Convert sentiment labels to numerical values
    label_encoder = LabelEncoder()
    
    # Extract text and labels
    texts = dataset['text']
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

def upload_to_hub(trainer, tokenizer, label_map, repository_id, token):
    """
    Upload the model, tokenizer, and label mapping to HuggingFace Hub
    """
    try:
        # Set the token
        HfFolder.save_token(token)
        
        # Push to hub
        logger.info(f"Uploading model to HuggingFace Hub: {repository_id}")
        trainer.push_to_hub(
            repository_id,
            commit_message="Upload fine-tuned GPT2 model for sentiment analysis"
        )
        
        # Upload label mapping
        with open("label_map.json", "w") as f:
            json.dump(label_map, f)
            
        logger.info("Model successfully uploaded to HuggingFace Hub!")
        
    except Exception as e:
        logger.error(f"Error uploading to HuggingFace Hub: {str(e)}")
        raise

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Train and upload a sentiment analysis model')
    parser.add_argument('--hub_token', type=str, help='HuggingFace Hub token')
    parser.add_argument('--repository_name', type=str, help='Repository name for HuggingFace Hub')
    parser.add_argument('--push_to_hub', action='store_true', help='Whether to push the model to HuggingFace Hub')
    
    args = parser.parse_args()

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
        push_to_hub=args.push_to_hub,
        hub_token=args.hub_token if args.push_to_hub else None,
        hub_model_id=args.repository_name if args.push_to_hub else None,
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
    
    # Save everything locally
    output_dir = "Fine_Tuned_Models"
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Saving model and tokenizer to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save label mapping
    label_map = {i: label for i, label in enumerate(label_encoder.classes_)}
    with open(os.path.join(output_dir, "label_map.json"), "w") as f:
        json.dump(label_map, f)
    
    # Upload to HuggingFace Hub if requested
    if args.push_to_hub and args.hub_token and args.repository_name:
        upload_to_hub(
            trainer=trainer,
            tokenizer=tokenizer,
            label_map=label_map,
            repository_id=args.repository_name,
            token=args.hub_token
        )
    
    logger.info("Training complete! Model and related files saved.")

if __name__ == "__main__":
    main()