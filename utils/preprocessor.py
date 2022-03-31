import torch
from torch.utils.data import Dataset

class Preprocessor(Dataset):

  def __init__(self, documents, targets, tokenizer, max_len):
    self.documents = documents
    self.targets = targets
    self.tokenizer = tokenizer
    self.max_len = max_len
  
  def __len__(self):
    return len(self.documents)
  
  def __getitem__(self, item):
    document = self.documents[item]
    target = self.targets[item]

    encoding = self.tokenizer.encode_plus(
      document,
      add_special_tokens=True,
      max_length=self.max_len,
      return_token_type_ids=False,
      pad_to_max_length=True,
      return_attention_mask=True,
      return_tensors='pt',
    )

    return {
      'text': document,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'targets': torch.tensor(target, dtype=torch.long)
    }

