#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from tqdm import tqdm


# In[ ]:


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Avoid pad token issues
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# In[ ]:


get_ipython().system('nvidia-smi      # switching to 4t - gpu')


# In[ ]:


dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
train_texts = dataset["train"]["text"]

# Filter out empty lines
train_texts = [t for t in train_texts if len(t.strip()) > 0]

# Tokenize
tokenized = [tokenizer.encode(t, return_tensors="pt").squeeze(0) for t in train_texts if len(t) > 10]


# In[ ]:


class GPTDataset(Dataset):
    def __init__(self, tokenized_texts, block_size=64):
        self.samples = []
        for text in tokenized_texts:
            for i in range(0, len(text) - block_size, block_size):
                self.samples.append(text[i:i+block_size])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return sample, sample  # input and target are same

train_dataset = GPTDataset(tokenized)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=lambda batch: pad_sequence([x[0] for x in batch], batch_first=True, padding_value=tokenizer.pad_token_id))


# In[ ]:


optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

model.train()
for epoch in range(1):
    loop = tqdm(train_loader, desc="Training")
    for batch in loop:
        batch = batch.to(device)
        outputs = model(input_ids=batch, labels=batch)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loop.set_postfix(loss=loss.item())


# In[ ]:


model.eval()
prompt = "The future of AI is"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
output = model.generate(input_ids, max_new_tokens=20)
print(tokenizer.decode(output[0]))


# In[ ]:


import math


# In[ ]:


def evaluate_perplexity(model, eval_loader):
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            batch = batch.to(device)
            outputs = model(input_ids=batch, labels=batch)
            loss = outputs.loss
            total_loss += loss.item() * batch.size(0)
            total_tokens += batch.size(0)

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return perplexity

# Example usage
perplexity = evaluate_perplexity(model, train_loader)  # or test_loader
print(f"Perplexity: {perplexity:.2f}")


# In[ ]:


def evaluate_perplexity(model, eval_loader):
    import math
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for i, batch in enumerate(tqdm(eval_loader, desc="Evaluating")):
            batch = batch.to(device)
            outputs = model(input_ids=batch, labels=batch)
            loss = outputs.loss
            total_loss += loss.item() * batch.size(0)
            total_tokens += batch.size(0)

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return perplexity

# Usage:
perplexity = evaluate_perplexity(model, train_loader)
print(f"Perplexity (small eval): {perplexity:.2f}")


# In[ ]:


def top_k_accuracy(model, loader, k=5):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="Top-k Accuracy")):
            batch = batch.to(device)
            outputs = model(batch)
            logits = outputs.logits

            shift_logits = logits[:, :-1, :]
            shift_labels = batch[:, 1:]

            top_k_preds = torch.topk(shift_logits, k, dim=-1).indices
            match = (top_k_preds == shift_labels.unsqueeze(-1)).any(dim=-1)
            correct += match.sum().item()
            total += match.numel()

    accuracy = correct / total
    return accuracy

# Usage:
acc = top_k_accuracy(model, train_loader, k=5)
print(f"Top-5 Accuracy (small eval): {acc:.2%}")

