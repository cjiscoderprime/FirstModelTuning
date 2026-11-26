import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

model_id = (
    "distilbert/"
    "distilbert-base-uncased-finetuned-sst-2-english"
) 

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForSequenceClassification.from_pretrained(
    model_id
)

model.eval()

text = "I love building software"

inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

probabilities = torch.softmax(outputs.logits, dim=-1)

print("Tokens: ", tokenizer.tokenize(text))
print("Token IDs: ", inputs["input_ids"])
print("Logits: ", outputs.logits)
print("Probabilities: ", probabilities)
print("Labels: ", model.config.id2label)

print(
    "Parameters: ",
    sum(p.numel() for p in model.parameters())
)

print(model)