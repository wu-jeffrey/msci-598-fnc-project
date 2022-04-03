import transformers
from transformers import pipeline
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup

import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

from utils.preprocessor import Preprocessor
from utils.dataset import DataSet
from utils.generate_train_test_splits import train_test_split, get_stances_for_split
from utils.score import report_score, LABELS, score_submission

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache() 

PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
MAX_LEN = 512 # 4788+40
BATCH_SIZE = 7
EPOCHS = 10

class StanceClassifier(nn.Module):
  def __init__(self, num_classes):
    super(StanceClassifier, self).__init__()
    self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
    self.drop = nn.Dropout(p=0.3)
    self.out = nn.Linear(self.bert.config.hidden_size, num_classes)
  
  def forward(self, input_ids, attention_mask):
    _, pooled_output = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask,
      return_dict=False
    )
    output = self.drop(pooled_output)
    return self.out(output)

def create_data_loader(stances, bodies, labels, tokenizer, max_len, batch_size):
  documents = []
  targets = []

  for stance in stances:
    headline = stance["Headline"]
    body = bodies[stance["Body ID"]]
    label = labels.index(stance["Stance"])
    doc = headline + ' ' + tokenizer.sep_token + ' ' + body

    documents.append(doc)
    targets.append(label)

  data = Preprocessor(
    documents=np.array(documents),
    targets=np.array(targets),
    tokenizer=tokenizer,
    max_len=max_len
  )

  return DataLoader(
    data,
    batch_size=batch_size,
    num_workers=4
  )

def train_epoch(
  model, 
  data_loader, 
  loss_fn, 
  optimizer, 
  device, 
  scheduler, 
  num_examples
):
  model = model.train()

  losses = []
  correct_predictions = 0
  
  for d in tqdm(data_loader):
    input_ids = d["input_ids"].to(device)
    attention_mask = d["attention_mask"].to(device)
    targets = d["targets"].to(device)

    outputs = model(
      input_ids=input_ids,
      attention_mask=attention_mask
    )

    _, preds = torch.max(outputs, dim=1)
    loss = loss_fn(outputs, targets)

    correct_predictions += torch.sum(preds == targets)
    losses.append(loss.item())

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

  return correct_predictions.double() / num_examples, np.mean(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples):
  model = model.eval()

  losses = []
  correct_predictions = 0

  with torch.no_grad():
    for d in data_loader:
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      targets = d["targets"].to(device)

      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
      )
      _, preds = torch.max(outputs, dim=1)

      loss = loss_fn(outputs, targets)

      correct_predictions += torch.sum(preds == targets)
      losses.append(loss.item())

  return correct_predictions.double() / n_examples, np.mean(losses)

def main():
  # Load the training dataset and break it into train/validation sets
  ds = DataSet()
  training_ids, validation_ids = train_test_split(ds)
  training_stances, validation_stances = get_stances_for_split(ds, training_ids, validation_ids)

  # Create pytorch dataloaders  
  tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
  train_data_loader = create_data_loader(training_stances, ds.articles, LABELS, tokenizer, MAX_LEN, BATCH_SIZE)
  val_data_loader = create_data_loader(validation_stances, ds.articles, LABELS, tokenizer, MAX_LEN, BATCH_SIZE)

  # load the model
  model = StanceClassifier(len(LABELS))
  model = model.to(device)

  # Training
  optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
  total_steps = len(train_data_loader) * EPOCHS

  scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
  )

  loss_fn = nn.CrossEntropyLoss().to(device)

  history = defaultdict(list)
  best_accuracy = 0

  for epoch in range(EPOCHS):

    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)

    train_acc, train_loss = train_epoch(
      model,
      train_data_loader,    
      loss_fn, 
      optimizer, 
      device, 
      scheduler, 
      len(training_stances)
    )

    print(f'Train loss {train_loss} accuracy {train_acc}')

    val_acc, val_loss = eval_model(
      model,
      val_data_loader,
      loss_fn, 
      device, 
      len(validation_stances)
    )

    print(f'Val   loss {val_loss} accuracy {val_acc}')
    print()

    history['train_acc'].append(train_acc)
    history['train_loss'].append(train_loss)
    history['val_acc'].append(val_acc)
    history['val_loss'].append(val_loss)

    if val_acc > best_accuracy:
      torch.save(model.state_dict(), 'best_model_state.bin')
      best_accuracy = val_acc


if __name__ == "__main__":
  main()


