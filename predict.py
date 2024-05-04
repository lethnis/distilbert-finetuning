from colorama import Fore

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def main():
    """Check how model performs on a set of random news."""

    # load model and tokenizer from chosen checkpoint
    checkpoint = "checkpoints/full_model/checkpoint-6500"
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    # extract index to label dict
    id2label = model.config.id2label

    # read news from a file
    with open("news/test.txt") as f:
        news = f.read().split("\n")

    # tokenize news and make them the same size
    tokens = tokenizer(news, padding=True, truncation=True, return_tensors="pt")
    # predict -> use softmax to convert logits to probas -> argmax to get max index
    preds = torch.softmax(model(**tokens).logits, dim=-1).argmax(dim=-1)

    # for each news print predicted label and the news
    for i in range(len(news)):
        print(f"Predicted label: {Fore.CYAN + id2label[int(preds[i])]}" + Fore.RESET)
        print(news[i])
        print()


if __name__ == "__main__":
    main()
