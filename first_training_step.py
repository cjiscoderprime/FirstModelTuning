import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, DataCollatorWithPadding
from prepare_model import model, model_id

#Load previous dataset
dataset = load_from_disk("data/tokenized")

tokenizer = AutoTokenizer.from_pretrained(model_id)

collator = DataCollatorWithPadding(
    tokenizer = tokenizer,
    return_tensors = "pt"
)

# Select 4 training examples for practice
samples = dataset["train"].select(range(4))

features = [
    {
    "input_ids": sample["input_ids"],
    "attention_mask": sample["attention_mask"],
    "label": sample["label"]
    }
    for sample in samples
]

batch = collator(features)

print("\nBatch dimensions: ")
for key, value in batch.items():
    print(key, value.shape)

# Enable training
model.train()

# Initialize optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=2e-5
)

# Save classifier weights before training
weights_before = model.classifier.weight.detach().clone()

# Clear existing gradients
optimizer.zero_grad()

#Forward pass
outputs = model(**batch)

print("\nLogits: ")
print(outputs.logits)

print("\nCorrect Labels: ")
print(batch["labels"])

print("\nLoss: ")
print(outputs.loss.item())

#Backpropogation
outputs.loss.backward()

print("\nClassifier gradient norm: ")
print(model.classifier.weight.grad.norm().item())

#Update parameters
optimizer.step()

weights_after = model.classifier.weight.detach()

print("\nTotal absolute change in classifier weights: ")
print((weights_after - weights_before).abs().sum().item())

print("\nFirst Training step completed. ")





