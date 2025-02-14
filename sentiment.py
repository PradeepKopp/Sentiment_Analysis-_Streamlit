import streamlit as st
from transformers import pipeline

@st.cache_resource
def load_sentiment_model():
    #Load pre_trained sentimnt analysis pipeline from Hugging face
    return pipeline("sentiment-analysis")

def main():
    st.title("Real Time Sentiment Analysis")
    st.markdown("""
                This App performs real time sentiment analysis on the input text.
                Simply type or paste your text in the box below to analyze its sentiment""")
    
    #Load Sentiment analysis model
    sentiment_analyzer=load_sentiment_model()

    #Input text
    user_input=st.text_area("Enter your text here:",height=200)

    if st.button("Analysis Sentiment") and user_input:
        #Perform Sentiment Analysis
        result=sentiment_analyzer(user_input)

        #Display the result
        sentiment=result[0]['label']
        score=result[0]['score']
        st.markdown(f"Sentiment:,{sentiment}")
        st.markdown(f"Confidence score:,{score:.2f}")
if __name__=="__main__":
    main()