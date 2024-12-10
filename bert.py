import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

# Load the dataset
def load_data():
    df = pd.read_csv('quora_question_pairs.csv')  # Adjust path if necessary
    questions1 = df['question1'].values
    questions2 = df['question2'].values
    labels = df['is_duplicate'].values
    return questions1, questions2, labels

# Dataset class to handle input encoding
class QADataset(Dataset):
    def _init_(self, question1, question2, labels, tokenizer, max_len):
        self.question1 = question1
        self.question2 = question2
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def _len_(self):
        return len(self.labels)

    def _getitem_(self, item):
        question1 = str(self.question1[item])
        question2 = str(self.question2[item])
        
        encoding = self.tokenizer.encode_plus(
            question1,
            question2,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()
        
        label = torch.tensor(self.labels[item], dtype=torch.long)
        
        return {'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': label}

# Initialize tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Prepare data
questions1, questions2, labels = load_data()
X_train, X_val, y_train, y_val = train_test_split(questions1, questions2, labels, test_size=0.1)

# Create Dataset objects and DataLoaders
train_dataset = QADataset(X_train, X_val, y_train, tokenizer, max_len=128)
val_dataset = QADataset(X_val, X_val, y_val, tokenizer, max_len=128)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Optimizer setup
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# Training Loop
for epoch in range(3):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
    print(f"Epoch {epoch + 1} completed with loss: {loss.item()}")

# Evaluation on validation set
model.eval()
total_eval_accuracy = 0
for batch in val_loader:
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    
    logits = outputs.logits
    preds = torch.argmax(logits, dim=1).flatten()
    total_eval_accuracy += (preds == labels).sum().item()

print(f"Validation Accuracy: {total_eval_accuracy / len(val_dataset):.4f}")

# Save the model
model.save_pretrained('quora_bert_model')

# Load the model
model = BertForSequenceClassification.from_pretrained('quora_bert_model')
model.to(device)