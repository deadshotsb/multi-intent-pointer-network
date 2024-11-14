import random
import os
import numpy as np
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, multilabel_confusion_matrix
import pandas as pd
from sklearn import metrics


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

import pandas as pd

## change the dataset file name, I used .tsv files to avoid any problems with comma
df_train = pd.read_csv('<train file path>')   # train data path
df_test = pd.read_csv('<test file path>')  # test data path
df = pd.concat([df_train, df_test], axis = 0)
n_train = df_train.shape[0]
n_test = df_test.shape[0]

# l = df["intent1"].unique()
# print(sorted(l))

# Column names of intent classes Labels in the dataset
req = ['<intent1>', '<intent2>'] #-> fill the intent columns 

import matplotlib.pyplot as plt

for x in reversed(req):
    counts = []
    classes = df[x].unique()
#     print(classes)
    for i in classes:
        count = len(df[df[x]==i])
        counts.append(count)
    print(len(classes), classes, counts)

## Change the file_name accordingly. This can be used during inference for label mapping
with open('<file_path>', 'w') as f:
    for x in classes:
        f.write(x+"\n")

sample_size = int(len(df))
sampleDf = df.sample(sample_size, random_state=23)
x = sampleDf.Text.values
y_1 = sampleDf[req[1]].values
y_2 = sampleDf[req[0]].values

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
encoder.fit(y_1)
encoder.fit(y_2)
num_classes = len(encoder.classes_)
print("Number of classes = " + str(num_classes))


class Triage(Dataset):
    def __init__(self, dataframe, req, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __getitem__(self, index):
        title = str(self.data.Text[index])
        title = " ".join(title.split())
        inputs = self.tokenizer.encode_plus(
            title,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {
            'text': title,
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'target_1': torch.tensor(self.data[req[0]][index], dtype=torch.long),
            'target_2': torch.tensor(self.data[req[1]][index], dtype=torch.long)
        } 
    
    def __len__(self):
        return self.len

class BERTClass(torch.nn.Module):
    def __init__(self, num_classes, model_name):
        super(BERTClass, self).__init__()
        self.l1 = AutoModel.from_pretrained(model_name)
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.2)
        self.classifier = torch.nn.Linear(768, num_classes)
        self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, input_ids, attention_mask):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        output = self.softmax(output)
        return output

def train_model(epoch, loader, model, loss_function, optimizer):
    tr_loss = 0
    tgts_1 = []
    tgts_2 = []
    preds_1 = []
    preds_2 = []
    texts = []
    model.train()
    for _, data in enumerate(tqdm(loader)):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        targets_1 = data['target_1'].to(device, dtype = torch.long)
        targets_2 = data['target_2'].to(device, dtype = torch.long)
        texts.extend(data['text'])

        outputs = model(ids, mask)
#         print(outputs, targets_1)
        loss_1 = loss_function(outputs, targets_1)
#         print(outputs, targets_2)
        loss_2 = loss_function(outputs, targets_2)
        loss = DOM_WT*loss_1 + loss_2
        tr_loss += loss.item()
        top_probs, top_indices = torch.topk(outputs, k=2, dim=1, largest=True, sorted=True)

        dominant_index = top_indices[:, 0]
        non_dominant_index = top_indices[:, 1]

        t1 = np.array([transform_dict[x] for x in targets_1.detach().cpu().numpy()])
        t2 = np.array([transform_dict[x] for x in targets_2.detach().cpu().numpy()])
        tgts_1.extend(t1.tolist())
        tgts_2.extend(t2.tolist())

        p1 = np.array([transform_dict[x] for x in dominant_index.detach().cpu().numpy()])
        p2 = np.array([transform_dict[x] for x in non_dominant_index.detach().cpu().numpy()])
        preds_1.extend(p1.tolist())
        preds_2.extend(p2.tolist())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
#     acc_1 = round(np.logical_or(metrics.accuracy_score(tgts_1, preds_1), metrics.accuracy_score(tgts_1, preds_2)) * 100, 2)
#     f1_1 = round(np.logical_or(metrics.f1_score(tgts_1, preds_1, average='macro'), metrics.f1_score(tgts_1, preds_2, average='macro')) * 100, 2)
#     acc_2 = round(np.logical_or(metrics.accuracy_score(tgts_2, preds_2), metrics.accuracy_score(tgts_2, preds_1)) * 100, 2)
#     f1_2 = round(np.logical_or(metrics.f1_score(tgts_2, preds_2, average='macro'), metrics.f1_score(tgts_2, preds_1, average='macro'))*100, 2)
    acc_1 = round(metrics.accuracy_score(tgts_1, preds_1) * 100, 2)
    f1_1 = round(metrics.f1_score(tgts_1, preds_1, average='macro') * 100, 2)
    acc_2 = round(metrics.accuracy_score(tgts_2, preds_2) * 100, 2)
    f1_2 = round(metrics.f1_score(tgts_2, preds_2, average='macro')*100, 2)
    print(f"Train Loss: {tr_loss}; Train Accuracy: {acc_1} / {f1_1}, {acc_2} / {f1_2}")
    return texts, tgts_1, tgts_2, preds_1, preds_2

def valid(model, loader):
    model.eval()
    te_loss = 0
    dom_loss = 0
    non_dom_loss = 0
    tgts_1 = []
    tgts_2 = []
    preds_1 = []
    preds_2 = []
    probs_1 = []
    probs_2 = []
    texts = []
    res = {}
    with torch.no_grad():
        for _, data in enumerate(tqdm(loader)):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            targets_1 = data['target_1'].to(device, dtype = torch.long)
            targets_2 = data['target_2'].to(device, dtype = torch.long)
            texts.extend(data['text'])

            outputs = model(ids, mask)
            loss_1 = loss_function(outputs, targets_1)
            loss_2 = loss_function(outputs, targets_2)
            loss = DOM_WT*loss_1 + loss_2
            te_loss += loss.item()
            dom_loss += loss_1.item()
            non_dom_loss += loss_2.item()
            top_probs, top_indices = torch.topk(outputs, k=2, dim=1, largest=True, sorted=True)
#             print(top_probs, top_indices)

            dominant_index = top_indices[:, 0]
            non_dominant_index = top_indices[:, 1]

            t1 = np.array([transform_dict[x] for x in targets_1.detach().cpu().numpy()])
            t2 = np.array([transform_dict[x] for x in targets_2.detach().cpu().numpy()])
            tgts_1.extend(t1.tolist())
            tgts_2.extend(t2.tolist())

            p1 = np.array([transform_dict[x] for x in dominant_index.detach().cpu().numpy()])
            p2 = np.array([transform_dict[x] for x in non_dominant_index.detach().cpu().numpy()])
            preds_1.extend(p1.tolist())
            preds_2.extend(p2.tolist())
            
            pb1 = top_probs[:, 0].detach().cpu().numpy()
            pb2 = top_probs[:, 1].detach().cpu().numpy()
            probs_1.extend(pb1.tolist())
            probs_2.extend(pb2.tolist())

        acc_1 = round(metrics.accuracy_score(tgts_1, preds_1)*100, 2)
        f1_1 = round(metrics.f1_score(tgts_1, preds_1, average='macro')*100, 2)
        acc_2 = round(metrics.accuracy_score(tgts_2, preds_2)*100, 2)
        f1_2 = round(metrics.f1_score(tgts_2, preds_2, average='macro')*100, 2)
        avg_acc = round((acc_1+acc_2)/2, 2)
        avg_f1 = round((f1_1+f1_2)/2, 2)
        res = {'Dom': f'{acc_1}/{f1_1}', 'Non-Dom': f'{acc_2}/{f1_2}', 'avg':f'{avg_acc}/{avg_f1}'}
        print(f"Test Loss: {te_loss}; Dom loss: {dom_loss}; Non dom loss: {non_dom_loss}; Test Accuracy: {acc_1} / {f1_1}, {acc_2} / {f1_2}")
    return texts, tgts_1, tgts_2, preds_1, preds_2, probs_1, probs_2, res

my_dict = {v: k for k, v in enumerate(classes)}
transform_dict = {k:v for k,v in enumerate(classes)}

def update_cat(x):
    return my_dict[x]

MAX_LEN = 265
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 8
EPOCHS = 15
LEARNING_RATE = 2e-5
model_name = 'distilbert-base-uncased' #'bert-base-uncased' #'roberta-base' #'google/electra-base-discriminator'#'distilbert-base-uncased'

train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': False,
                'num_workers': 0
                }

test_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': False,
                'num_workers': 0
                }


import numpy as np
from torch import cuda
from sklearn.model_selection import train_test_split

df = df[['Text']+req]
print(df.columns)
train_df = df.head(n_train)
test_df = df.tail(len(df) - n_train)
# train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

for x in req:
    train_df[x] = train_df[x].apply(lambda x: update_cat(x))
    test_df[x] = test_df[x].apply(lambda x: update_cat(x))

tokenizer = AutoTokenizer.from_pretrained(model_name)

training_set = Triage(train_df, req, tokenizer, MAX_LEN)
testing_set = Triage(test_df, req, tokenizer, MAX_LEN)

training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)

results = {}
for W in [1, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]:
    DOM_WT = W
    print(DOM_WT)
    device = 'cuda' if cuda.is_available() else 'cpu'
    model = BERTClass(num_classes, model_name)
    model.to(device)

    loss_function = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        tr1, t11, t21, p11, p21 = train_model(epoch, training_loader, model, loss_function, optimizer)
    texts, tgts_1, tgts_2, preds_1, preds_2, probs_1, probs_2, res = valid(model, testing_loader)
    results[DOM_WT] = res
    preds = np.array([preds_1, preds_2]).T
    targets = np.array([tgts_1, tgts_2]).T
    exact_match_ratio = np.mean(np.all(targets == preds, axis=1))
#     preds_T = np.array([preds_2, preds_1]).T
#     exact_match_ratio_T = np.mean(np.all(targets == preds_T, axis=1))
    print("Overall Accuracy:", round(exact_match_ratio*100, 2))
    results[DOM_WT]['OA'] = round(exact_match_ratio*100, 2)
#     print("Overall Accuracy:", round(np.logical_or(exact_match_ratio, exact_match_ratio_T)*100, 2))
#     results[DOM_WT]['OA'] = round(np.logical_or(exact_match_ratio, exact_match_ratio_T)*100, 2)
#     preds = np.sort(preds, axis=1)
#     targets = np.sort(targets, axis=1)
#     exact_match_ratio = np.mean(np.all(targets == preds, axis=1))
#     print("Exact Match Ratio:", round(exact_match_ratio*100, 2))
#     results[DOM_WT]['EM'] = round(exact_match_ratio*100, 2)
    df_test = pd.DataFrame.from_dict({'Text': texts, 'pred_dom': preds_1, 'pred_non_dom': preds_2, 'dom': tgts_1, 'non_dom': tgts_2, 'probs_dom':probs_1, 'probs_non_dom': probs_2})
    ##################### CHANGE THE FILE NAME HERE, <model>_<coarse/fine>...
    torch.save(model, f'../model_checkpoints/<dataset>/<model_name>_{DOM_WT}_checkpoint.pt')
    df_test.to_csv(f'../result_csv/<dataset>/<model_name>_<dataset>_{DOM_WT}.csv', index=False)
    del model
    print("=========================================================================================")

for k, v in results.items():
    print(f'Dom Weight: {k}, Results: {v}')