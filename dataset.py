from torch import tensor
from torch.utils.data import Dataset
from transformers import BertTokenizer

class CodeDataSet(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        code, label = self.data[index]
        encoding = self.tokenizer.encode_plus(code, add_special_token=True, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(), "label": tensor(label)}
    
            