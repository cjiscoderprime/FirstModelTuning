from datasets import load_dataset
from transformers import AutoTokenizer

dataset = load_dataset(
    "zeroshot/twitter-financial-news-sentiment"
)

model_id = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = AutoTokenizer.from_pretrained(model_id)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length = 128,
        return_token_type_ids=False
    )

tokenized_dataset = dataset.map(
    tokenize_function,
    batched = True
)

print("Tokenized dataset: ")
print(tokenized_dataset)

print("\nFirst tokenized example: ")
print(tokenized_dataset["train"][0])

tokenized_dataset.save_to_disk("data/tokenized")




