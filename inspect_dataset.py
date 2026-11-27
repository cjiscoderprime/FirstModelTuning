from datasets import load_dataset
from collections import Counter

dataset = load_dataset(
    "zeroshot/twitter-financial-news-sentiment"
)

labels = {
    0: "Bearish",
    1: "Bullish",
    2: "Neutral"
}

print("Dataset Structure: ")
print(dataset)

print("\nDataset features: ")
print(dataset["train"].features)

print("\nLabel distribution: ")
for split in dataset:
    counts = Counter(dataset[split]["label"])

    print(f"\n{split}: ")
    for label, count in sorted(counts.items()):
        print(labels[label], count)

print("\nRandom training examples: ")
samples = dataset["train"].shuffle(seed=42)

for i in range(10):
    sample = samples[i]
    print("Text: ", sample["text"])
    print("Label: ", labels[sample["label"]])
    print()