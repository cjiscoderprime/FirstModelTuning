from datasets import load_dataset
from transformers import AutoTokenizer

dataset = load_dataset(
    "zeroshot/twitter-financial-news-sentiment"
)

model_id = (
    "distilbert/"
    "distilbert-base-uncased-finetuned-sst-2-english"
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

sample = dataset["train"][0]

encoded = tokenizer(sample["text"])

print("Original text: ")
print(sample["text"])

print("\nOriginal label: ")
print(sample["label"])

print("\nTokens:")
print(tokenizer.convert_ids_to_tokens(encoded["input_ids"]))

print("\nToken IDs: ")
print(encoded["input_ids"])

print("\nAttention Mask: ")
print(encoded["attention_mask"])

print("\nSequence Length: ")
print(len(encoded["input_ids"]))



