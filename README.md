# Fine tune base model for sentimental analysis


This is a LLM trainer that generates a finetuned model starting from GTP2. After the training phase you can submit sentences to the model to classify the sentiment (positive, neutral, negative)



Installation phase. Install python modules



pip install pandas numpy datasets transformers evaluate torch huggingface_hub argparse scikit-learn

pip install accelerator -U
