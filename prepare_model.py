import torch
from transformers import AutoModelForSequenceClassification

torch.manual_seed(42)

#General Pretrained Model
model_id = "distilbert/distilbert-base-uncased"

id2label = {
    0: "BEARISH",
    1: "BULLISH",
    2: "NEUTRAL"
}

label2id = {
    label: idx for idx, label in id2label.items()
}

model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    num_labels=3,
    id2label=id2label,
    label2id = label2id,
    ignore_mismatched_sizes=True
)

print("\nClassification layer: ")
print(model.classifier)

print("\nClassification weights: ")
print(model.classifier.weight)

print("\nWeight dimensions")
print(model.classifier.weight.shape)

print("\nBias dimensions")
print(model.classifier.bias.shape)

print("\nLabel mappings: ")
print(model.config.id2label)

print("\nTotal parameters: ")
print(sum(p.numel() for p in model.parameters()))

print("\nTrainable parameters: ")
print(
    sum(p.numel() for p in model.parameters()
        if p.requires_grad)
)
