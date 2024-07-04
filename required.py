from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import Tuple

# Set device to GPU if available, otherwise CPU
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load the FinBERT tokenizer and model from ProsusAI
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device)

# Define sentiment labels
labels = ["positive", "negative", "neutral"]

def estimate_sentiment(news: list) -> Tuple[torch.Tensor, str]:
    """
    Estimate sentiment from a list of news headlines.

    :param news: List of news headlines.
    :return: Tuple containing the probability of the most likely sentiment and the sentiment label.
    """
    if news:
        # Tokenize the input news headlines
        tokens = tokenizer(news, return_tensors="pt", padding=True).to(device)

        # Get model predictions
        logits = model(tokens["input_ids"], attention_mask=tokens["attention_mask"])["logits"]

        # Aggregate and apply softmax to logits
        aggregated_logits = torch.sum(logits, 0)
        probabilities = torch.nn.functional.softmax(aggregated_logits, dim=-1)

        # Determine the most likely sentiment
        max_probability = probabilities[torch.argmax(probabilities)]
        sentiment_label = labels[torch.argmax(probabilities)]

        return max_probability, sentiment_label
    else:
        return torch.tensor(0), labels[-1]  # Return neutral if no news

if __name__ == "__main__":
    # Example usage
    #headlines = ['responds negative to the news!', 'traders are sad!']
    #probability, sentiment = estimate_sentiment(headlines)
    #print(probability, sentiment)

    # Check if CUDA is available
    print(torch.cuda.is_available())
