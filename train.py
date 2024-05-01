from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)


def tokenize_headlines(data):
    return tokenizer(data["news_headline"], truncation=True)


dataset = (
    Dataset.from_csv("news/data.csv")
    .remove_columns("Unnamed: 0")
    .rename_column("news_category", "labels")
    .class_encode_column("labels")
    .train_test_split(test_size=0.1, stratify_by_column="labels")
)

checkpoint = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
data_collator = DataCollatorWithPadding(tokenizer)

dataset = dataset.map(tokenize_headlines, batched=True)

training_args = TrainingArguments("runs", num_train_epochs=1)

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=7, ignore_mismatched_sizes=True)

model.requires_grad_(False)
model.classifier.requires_grad_(True)

trainer = Trainer(
    model,
    training_args,
    data_collator,
    dataset["train"],
    dataset["test"],
    tokenizer,
)

trainer.train()
