import streamlit as st
import pandas as pd
from utils import preprocess_essay_text, predict_score

# Streamlit App Title and Description
st.title("Essay Scorer Pro")
st.write("A tool to score essays using AI. Enter text or upload a CSV file to get predictions on scale 1-6.")

# Option to select input method
input_method = st.radio(
    "Select Input Method:",
    ("Enter Text", "Upload CSV File")
)

if input_method == "Enter Text":
    # Text Input
    essay_text = st.text_area("Enter your essay text here:")
    if st.button("Predict Score"):
        if essay_text.strip():
            score = predict_score(essay_text)
            st.success(f"Predicted Score: {score}")
        else:
            st.warning("Please enter some text!")

elif input_method == "Upload CSV File":
    # File Upload
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file:
        # Read the uploaded file
        data = pd.read_csv(uploaded_file)
        
        if 'full_text' not in data.columns:
            st.error("The CSV file must have a 'full_text' column.")
        else:
            # Make predictions
            st.write("Processing...")
            data['Predicted_Score'] = data['full_text'].apply(predict_score)
            
            # Display the predictions
            st.write("Predictions:")
            st.dataframe(data)
            
            # Download option
            csv = data.to_csv(index=False)
            st.download_button(
                label="Download Predictions as CSV",
                data=csv,
                file_name="predicted_scores.csv",
                mime="text/csv"
            )
