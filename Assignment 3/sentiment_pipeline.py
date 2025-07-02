from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import torch
import evaluate

# 1. Load the IMDb dataset
dataset = load_dataset("imdb")

# 2. Load tokenizer and tokenize the data
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(example):
    return tokenizer(example["text"], truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 3. Load the pre-trained BERT model
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# 4. Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 5. Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions)
    }

# 6. Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"].shuffle(seed=42).select(range(2000)),  # for quicker training
    eval_dataset=tokenized_datasets["test"].select(range(500)),
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 7. Train the model
trainer.train()

# 8. Evaluate the model
metrics = trainer.evaluate()
print("\nEvaluation Metrics:", metrics)

# 9. Save the model and tokenizer
model.save_pretrained("./sentiment-model")
tokenizer.save_pretrained("./sentiment-model")

# 10. Load and use the model for inference
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class_id = logits.argmax().item()
    return "positive" if predicted_class_id == 1 else "negative"

# Example usage
sample_text = "This movie was absolutely wonderful!"
print("Sample prediction:", predict_sentiment(sample_text))
