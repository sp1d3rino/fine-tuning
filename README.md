# Fine tune base model for sentimental analysis


This is a LLM trainer that generates a finetuned model starting from GTP2. After the training phase you can submit sentences to the model to classify the sentiment (positive, neutral, negative)



Installation phase. Install python modules



pip install pandas numpy datasets transformers evaluate torch huggingface_hub argparse scikit-learn

pip install accelerator -U

pip install streamlit


## Train the model

python trainer.py

## Download the fine tuned Model

cd Fine_Tuned_Models

git clone https://huggingface.co/fabras/sentimental-gpt

## Run chat with fine tuned model
streamlit run chat-LLM.py
