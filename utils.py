import torch
import re
from text_unidecode import unidecode
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load the saved model and tokenizer
Model_path = "jatinmehra/Smollm2-360M-Essay-Scoring" 
model = AutoModelForSequenceClassification.from_pretrained(Model_path)
tokenizer = AutoTokenizer.from_pretrained(Model_path)

# Preprocessing Functions
def resolve_encodings_and_normalize(text: str) -> str:
    """Resolve encoding problems and normalize abnormal characters."""
    text = (
        text.encode("raw_unicode_escape")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
        .encode("cp1252", errors="replace_encoding_with_utf8")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
    )
    text = unidecode(text)  # Convert accented characters to ASCII
    return text

def preprocess_essay_text(text: str) -> str:
    """
    Prepares essay text for scoring by cleaning non-essential issues without altering quality indicators.
    """
    text = resolve_encodings_and_normalize(text)
    text = re.sub(r'\s+', ' ', text.strip())  # Normalize whitespace
    text = re.sub(r'\s+([?.!,"])', r'\1', text)  # Remove spaces before punctuation
    text = re.sub(r',([^\s])', r', \1', text)    # Add space after commas
    return text

# Prediction Function
def predict_score(text: str) -> int:
    # Preprocess the text
    processed_text = preprocess_essay_text(text)

    # Tokenize the input text
    encoding = tokenizer(
        processed_text,
        padding='max_length',
        truncation=True,
        max_length=512,
        return_tensors='pt'
    )

    # Get input IDs and attention mask
    input_ids = encoding['input_ids'].squeeze(0).unsqueeze(0)  # Add batch dimension
    attention_mask = encoding['attention_mask'].squeeze(0).unsqueeze(0)  # Add batch dimension

    # Move tensors to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    # Perform inference
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=-1).cpu().numpy()

    # Convert prediction to score (adjust based on your scoring range)
    score = prediction[0] + 1  # Scores range from 1 to 6 | Model predicts from 0 to 5.
    return score
