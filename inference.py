import transformers
from transformers import pipeline
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup

import numpy as np
import pandas as pd
from collections import defaultdict
import pickle
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

def eval_model(model, data_loader, loss_fn, device, n_examples):
  model = model.eval()

  losses = []
  predictions = []
  correct_predictions = 0

  with torch.no_grad():
    for d in tqdm(data_loader):
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      targets = d["targets"].to(device)

      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
      )
      _, preds = torch.max(outputs, dim=1)

      predictions.append(preds)

      loss = loss_fn(outputs, targets)

      correct_predictions += torch.sum(preds == targets)
      losses.append(loss.item())

  return correct_predictions.double() / n_examples, np.mean(losses), predictions

def main():
  # Load the competition dataset
  competition_dataset = DataSet("competition_test")

  # # Create pytorch dataloaders  
  # tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
  # data_loader = create_data_loader(competition_dataset.stances, competition_dataset.articles, LABELS, tokenizer, MAX_LEN, BATCH_SIZE)

  # # load the model
  # model = StanceClassifier(len(LABELS))
  # model.load_state_dict(torch.load('best_model_state.bin'))
  # model = model.to(device)

  # loss_fn = nn.CrossEntropyLoss().to(device)

  # val_acc, val_loss, predictions = eval_model(
  #   model,
  #   data_loader,
  #   loss_fn, 
  #   device, 
  #   len(competition_dataset.stances)
  # )

  # print(f'loss {val_loss} accuracy {val_acc}')
  # print()

  # batched_predictions = list(map(lambda x: x.cpu().tolist(), predictions))
  # flat_list = [item for sublist in batched_predictions for item in sublist]

  # with open('competition_predictions.pkl', 'wb') as f:
  #   pickle.dump(flat_list, f)

  with open('competition_predictions.pkl', 'rb') as f:
    flat_list = pickle.load(f)

  predicted = [LABELS[int(a)] for a in flat_list]

  answer = {'Headline': [], 'Body ID': [], 'Stance': []}
  for i, x in enumerate(competition_dataset.stances):
      answer['Headline'].append(x['Headline'])
      answer['Body ID'].append(x['Body ID'])
      answer['Stance'].append(predicted[i])

  df = pd.DataFrame(data=answer)

  df.to_csv('answer.csv', index=False, encoding='utf-8')

if __name__ == "__main__":
  main()
