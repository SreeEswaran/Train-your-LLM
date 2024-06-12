# Python script for data preprocessing
# Include functions for cleaning and tokenizing text data

import re
from transformers import GPT2Tokenizer

# Example text data
example_text = """
Lorem Ipsum is simply dummy text of the printing and typesetting industry. 
Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, 
when an unknown printer took a galley of type and scrambled it to make a type specimen book.
"""

def clean_text(text):
    # Remove special characters, digits, and extra whitespace
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    text = re.sub(r'\d+', '', text)      # remove digits
    text = text.strip()                  # remove leading/trailing whitespace
    text = re.sub(' +', ' ', text)       # remove extra spaces
    return text

def tokenize_text(text):
    # Tokenize text using GPT-2 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt-4')
    tokens = tokenizer.encode(text, return_tensors='pt')
    return tokens

if __name__ == "__main__":
    # Example usage
    cleaned_text = clean_text(example_text)
    tokenized_text = tokenize_text(cleaned_text)
    
    print("Cleaned Text:")
    print(cleaned_text)
    print("\nTokenized Text:")
    print(tokenized_text)
