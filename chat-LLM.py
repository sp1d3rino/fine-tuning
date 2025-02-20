import streamlit as st
import torch
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
import json
import os

class SentimentAnalyzer:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.model = GPT2ForSequenceClassification.from_pretrained(model_path).to(self.device)
        
        # Load label mapping
        with open(os.path.join(model_path, "label_map.json"), "r") as f:
            self.label_map = json.load(f)

    def predict(self, text):
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=-1).item()
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            confidence = probabilities[0][prediction].item() * 100
        
        # Map to label
        sentiment = self.label_map[str(prediction)]
        return sentiment, confidence

def main():
    st.set_page_config(page_title="Sentiment Analysis Chat", page_icon="🤖")
    
    st.title("💭 Sentiment Analysis Chat")
    st.write("Enter your text below to analyze its sentiment!")

    # Initialize session state for chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Initialize the model
    @st.cache_resource
    def load_model():
        return SentimentAnalyzer("Fine_Tuned_Models/sentimental-gpt")

    try:
        analyzer = load_model()
        
        # Text input
        user_input = st.text_area("Enter your text:", height=100)
        
        if st.button("Analyze Sentiment"):
            if user_input:
                # Get prediction
                sentiment, confidence = analyzer.predict(user_input)
                
                # Add to chat history
                st.session_state.chat_history.append({
                    "text": user_input,
                    "sentiment": sentiment,
                    "confidence": confidence
                })
                
                # Clear input
                st.text_area("Enter your text:", value="", height=100, key="clear_input")

        # Display chat history
        st.subheader("Analysis History")
        for i, item in enumerate(reversed(st.session_state.chat_history)):
            with st.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**Text:** {item['text']}")
                with col2:
                    # Color code the sentiment
                    if item['sentiment'] == 'positive':
                        sentiment_color = 'green'
                    elif item['sentiment'] == 'negative':
                        sentiment_color = 'red'
                    else:
                        sentiment_color = 'grey'
                    
                    st.markdown(f"**Sentiment:** :{sentiment_color}[{item['sentiment']}]")
                    st.write(f"**Confidence:** {item['confidence']:.2f}%")
                st.divider()

        # Add a clear history button
        if st.button("Clear History"):
            st.session_state.chat_history = []
            st.experimental_rerun()

    except Exception as e:
        st.error(f"Error loading model: Make sure you have trained the model and saved it in the 'Fine_Tuned_Models' directory.")
        st.error(f"Error details: {str(e)}")

if __name__ == "__main__":
    main()