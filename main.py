from transformers import pipeline

classifier = pipeline(
    "sentiment-analysis",
    model = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

while True:
    text = input("\nEnter Text: ")

    if text == "exit":
        break

    result = classifier(text)
    print(result)