import torch
import transformers
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from dataset import CodeDataSet
from torch.utils.data import DataLoader

#load data
code = open('test.txt', 'r').read()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_length = 128
dataset = CodeDataSet(code, tokenizer, max_length)

model = BertForSequenceClassification.from_pretrained('bert-based-untrained', num_labels=len(set(label for _, label in code)))
optimizer = AdamW(model.parameters(), lr=5e-5)


train_data = int(0.8 * len(dataset))
val_data = len(dataset) - train_data

train_dataset,val_dataset = torch.utils.data.random_split(dataset, [train_data,val_data])

batch = 4
train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=True)

epochs = 3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


for epoch in range(epochs):
    model.train()
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        total, correct = 0,0
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = correct/total
        print()