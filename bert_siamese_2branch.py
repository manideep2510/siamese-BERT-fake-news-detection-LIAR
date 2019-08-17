import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from random import randrange
import torch.nn.functional as F
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt

#home = str(Path.home())

train_path = '/fake_news/LIAR-PLUS/dataset/train2.tsv'
test_path = '/fake_news/LIAR-PLUS/dataset/test2.tsv'
val_path = '/fake_news/LIAR-PLUS/dataset/val2.tsv'


train_df = pd.read_csv(train_path, sep="\t", header=None)
test_df = pd.read_csv(test_path, sep="\t", header=None)
val_df = pd.read_csv(val_path, sep="\t", header=None)

# Fill nan (empty boxes) with 0
train_df = train_df.fillna('None')
test_df = test_df.fillna('None')
val_df = val_df.fillna('None')

train = train_df.values
test = test_df.values
val = val_df.values


labels = {'train':[train[i][2] for i in range(len(train))], 'test':[test[i][2] for i in range(len(test))], 'val':[val[i][2] for i in range(len(val))]}
statements = {'train':[train[i][3] for i in range(len(train))], 'test':[test[i][3] for i in range(len(test))], 'val':[val[i][3] for i in range(len(val))]}
subjects = {'train':[train[i][4] for i in range(len(train))], 'test':[test[i][4] for i in range(len(test))], 'val':[val[i][4] for i in range(len(val))]}
speakers = {'train':[train[i][5] for i in range(len(train))], 'test':[test[i][5] for i in range(len(test))], 'val':[val[i][5] for i in range(len(val))]}
jobs = {'train':[train[i][6] for i in range(len(train))], 'test':[test[i][6] for i in range(len(test))], 'val':[val[i][6] for i in range(len(val))]}
states = {'train':[train[i][7] for i in range(len(train))], 'test':[test[i][7] for i in range(len(test))], 'val':[val[i][7] for i in range(len(val))]}
affiliations = {'train':[train[i][8] for i in range(len(train))], 'test':[test[i][8] for i in range(len(test))], 'val':[val[i][8] for i in range(len(val))]}
credits = {'train':[train[i][9:14] for i in range(len(train))], 'test':[test[i][9:14] for i in range(len(test))], 'val':[val[i][9:14] for i in range(len(val))]}
contexts = {'train':[train[i][14] for i in range(len(train))], 'test':[test[i][14] for i in range(len(test))], 'val':[val[i][14] for i in range(len(val))]}
justification = {'train':[train[i][15] for i in range(len(train))], 'test':[test[i][15] for i in range(len(test))], 'val':[val[i][15] for i in range(len(val))]}


def to_onehot(a):
    a_cat = [0]*len(a)
    for i in range(len(a)):
        if a[i]=='true':
            a_cat[i] = [1,0,0,0,0,0]
        elif a[i]=='mostly-true':
            a_cat[i] = [0,1,0,0,0,0]
        elif a[i]=='half-true':
            a_cat[i] = [0,0,1,0,0,0]
        elif a[i]=='barely-true':
            a_cat[i] = [0,0,0,1,0,0]
        elif a[i]=='false':
            a_cat[i] = [0,0,0,0,1,0]
        elif a[i]=='pants-fire':
            a_cat[i] = [0,0,0,0,0,1]
        else:
            print('Incorrect label')
    return a_cat


def to_onehot(a):
    a_cat = [0]*len(a)
    for i in range(len(a)):
        if a[i]=='true':
            a_cat[i] = [1,0]
        elif a[i]=='mostly-true':
            a_cat[i] = [1,0]
        elif a[i]=='half-true':
            a_cat[i] = [1,0]
        elif a[i]=='barely-true':
            a_cat[i] = [0,1]
        elif a[i]=='false':
            a_cat[i] = [0,1]
        elif a[i]=='pants-fire':
            a_cat[i] = [0,1]
        else:
            print('Incorrect label')
    return a_cat


labels_onehot = {'train':to_onehot(labels['train']), 'test':to_onehot(labels['test']), 'val':to_onehot(labels['val'])}


# Preparing meta data

#credit['train'][2]

metadata = {'train':[0]*len(train), 'val':[0]*len(val), 'test':[0]*len(test)}

for i in range(len(train)):
    subject = subjects['train'][i]
    if subject == 0:
        subject = 'None'
    
    speaker = speakers['train'][i]
    if speaker == 0:
        speaker = 'None'
    
    job = jobs['train'][i]
    if job == 0:
        job = 'None'
    
    state = states['train'][i]
    if state == 0:
        state = 'None'

    affiliation = affiliations['train'][i]
    if affiliation == 0:
        affiliation = 'None'

    context = contexts['train'][i]
    if context == 0 :
        context = 'None'
        
    meta = justification['train'][i] + ' ' + subject + ' ' + speaker + ' ' + job + ' ' + state + ' ' + affiliation + ' ' + context
    
    metadata['train'][i] = meta
    
for i in range(len(val)):
    subject = subjects['val'][i]
    if subject == 0:
        subject = 'None'
    
    speaker = speakers['val'][i]
    if speaker == 0:
        speaker = 'None'
    
    job = jobs['val'][i]
    if job == 0:
        job = 'None'
    
    state = states['val'][i]
    if state == 0:
        state = 'None'

    affiliation = affiliations['val'][i]
    if affiliation == 0:
        affiliation = 'None'

    context = contexts['val'][i]
    if context == 0 :
        context = 'None'
        
    meta = justification['val'][i] + ' ' + subject + ' ' + speaker + ' ' + job + ' ' + state + ' ' + affiliation + ' ' + context
    
    metadata['val'][i] = meta
    
for i in range(len(test)):
    subject = subjects['test'][i]
    if subject == 0:
        subject = 'None'
    
    speaker = speakers['test'][i]
    if speaker == 0:
        speaker = 'None'
    
    job = jobs['test'][i]
    if job == 0:
        job = 'None'
    
    state = states['test'][i]
    if state == 0:
        state = 'None'

    affiliation = affiliations['test'][i]
    if affiliation == 0:
        affiliation = 'None'

    context = contexts['test'][i]
    if context == 0 :
        context = 'None'
        
    meta = justification['test'][i] + ' ' + subject + ' ' + speaker + ' ' + job + ' ' + state + ' ' + affiliation + ' ' + context
    
    metadata['test'][i] = meta
    
    
# Credit score calculation
credit_score = {'train':[0]*len(train), 'val':[0]*len(val), 'test':[0]*len(test)}
for i in range(len(train)):
    credit = credits['train'][i]
    if sum(credit) == 0:
        score = 0.5
    else:
        score = (credit[3]*0.2 + credit[2]*0.5 + credit[0]*0.75 + credit[1]*0.9 + credit[4]*1)/(sum(credit))
    credit_score['train'][i] = [score for i in range(1536)]
    
for i in range(len(val)):
    credit = credits['val'][i]
    if sum(credit) == 0:
        score = 0.5
    else:
        score = (credit[3]*0.2 + credit[2]*0.5 + credit[0]*0.75 + credit[1]*0.9 + credit[4]*1)/(sum(credit))
    credit_score['val'][i] = [score for i in range(1536)]
    
for i in range(len(test)):
    credit = credits['test'][i]
    if sum(credit) == 0:
        score = 0.5
    else:
        score = (credit[3]*0.2 + credit[2]*0.5 + credit[0]*0.75 + credit[1]*0.9 + credit[4]*1)/(sum(credit))
    credit_score['test'][i] = [score for i in range(1536)]


class BertLayerNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-12):
            """Construct a layernorm module in the TF style (epsilon inside the square root).
            """
            super(BertLayerNorm, self).__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.variance_epsilon = eps

        def forward(self, x):
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.weight * x + self.bias
        

class BertForSequenceClassification(nn.Module):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.
    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.
    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary. Items in the batch should begin with the special "CLS" token. (see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels].
    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].
    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
    num_labels = 2
    model = BertForSequenceClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, num_labels=2): # Change number of labels here.
        super(BertForSequenceClassification, self).__init__()
        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size*2, num_labels)
        #self.fc1 = nn.Linear(config.hidden_size*2, 512)
        nn.init.xavier_normal_(self.classifier.weight)
        
    '''def forward_once(self, x):
        # Forward pass 
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output'''
        
    def forward_once(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        #logits = self.classifier(pooled_output)

        return pooled_output
    
    def forward(self, input_ids1, input_ids2, credit_sc):
        # forward pass of input 1
        output1 = self.forward_once(input_ids1, token_type_ids=None, attention_mask=None, labels=None)
        # forward pass of input 2
        output2 = self.forward_once(input_ids2, token_type_ids=None, attention_mask=None, labels=None)
        
        out = torch.cat((output1, output2), 1)
        #print(out.shape)
        
        # Multiply the credit score with the output after concatnation
                    
        out = torch.add(credit_sc, out)
        
        #out = self.fc1(out)
        logits = self.classifier(out)
        
        return logits
    
    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False
    
    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True


from pytorch_pretrained_bert import BertConfig

config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

num_labels = 2
model = BertForSequenceClassification(num_labels)


# Loading the statements
X_train = statements['train']
y_train = labels_onehot['train']

X_val = statements['val']
y_val = labels_onehot['val']

X_train = X_train + X_val
y_train = y_train + y_val


X_test = statements['test']
y_test = labels_onehot['test']

# Loading the justification
X_train_just = justification['train']

X_val_just = justification['val']

X_train_just = X_train_just + X_val_just

X_test_just = statements['test']


# Loading the meta data
X_train_meta = metadata['train']
X_val_meta = metadata['val']
X_train_meta = X_train_meta + X_val_meta
X_test_meta = metadata['test']

# Loading Credit scores

X_train_credit = credit_score['train']
X_val_credit = credit_score['val']
X_train_credit = X_train_credit+X_val_credit
X_test_credit = credit_score['test']


# Small data partitioned for debugging
'''X_train = X_train[:100]
y_train = y_train[:100]

X_test = X_test[:100]
y_test = y_test[:100]

X_train_just = X_train_just[:100]
X_test_just = X_test_just[:100]

X_train_meta = X_train_meta[:100]
X_test_meta = X_test_meta[:100]

X_train_credit = X_train_credit[:100]
X_test_credit = X_test_credit[:100]'''

max_seq_length = 128
class text_dataset(Dataset):
    def __init__(self,x_y_list, transform=None):
        
        self.x_y_list = x_y_list
        self.transform = transform
        
    def __getitem__(self,index):
        
        # Tokenize statements
        tokenized_review = tokenizer.tokenize(self.x_y_list[0][index])
        
        if len(tokenized_review) > max_seq_length:
            tokenized_review = tokenized_review[:max_seq_length]
            
        ids_review  = tokenizer.convert_tokens_to_ids(tokenized_review)

        padding = [0] * (max_seq_length - len(ids_review))
        
        ids_review += padding
        
        assert len(ids_review) == max_seq_length
        
        #print(ids_review)
        ids_review = torch.tensor(ids_review)
        
        sentiment = self.x_y_list[3][index] # color        
        list_of_labels = [torch.from_numpy(np.array(sentiment))]
        
        # Tokenize metadata
     
        tokenized_review_meta = tokenizer.tokenize(self.x_y_list[1][index])
        
        if len(tokenized_review_meta) > max_seq_length:
            tokenized_review_meta = tokenized_review_meta[:max_seq_length]
            
        ids_review_meta  = tokenizer.convert_tokens_to_ids(tokenized_review_meta)

        padding = [0] * (max_seq_length - len(ids_review_meta))
        
        ids_review_meta += padding
        
        assert len(ids_review_meta) == max_seq_length
        
        #print(ids_review)
        ids_review_meta = torch.tensor(ids_review_meta)
        
        sentiment = self.x_y_list[3][index] # color        
        list_of_labels = [torch.from_numpy(np.array(sentiment))]
        
        credit_scr = self.x_y_list[2][index] # Credit score
        
        #ones_768 = np.ones((768))
        #credit_scr = credit_scr * ones_768
        credit_scr = torch.tensor(credit_scr)
        
        return [ids_review, ids_review_meta, credit_scr], list_of_labels[0]
    
    def __len__(self):
        return len(self.x_y_list[0])

batch_size = 8

# Train Statements and Justifications
train_lists = [X_train, X_train_meta, X_train_credit, y_train]

# Test Statements and Justifications
test_lists = [X_test,, X_train_meta, X_test_credit, y_test]

# Preparing the data (Tokenize)
training_dataset = text_dataset(x_y_list = train_lists)
test_dataset = text_dataset(x_y_list = test_lists)


# Prepare the training dictionaries
dataloaders_dict = {'train': torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=0),
                   'val':torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
                   }
dataset_sizes = {'train':len(train_lists[0]),
                'val':len(test_lists[0])}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

train_acc = []
val_acc = []
train_loss = []
val_loss = []

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    print('starting')
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 100
    best_acc = 0

    for epoch in range(num_epochs):
        epoch_start = time.time()
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            
            sentiment_corrects = 0
            
            
            # Iterate over data.
            for inputs, sentiment in dataloaders_dict[phase]:
                
                inputs1 = inputs[0] # News statement input
                inputs2 = inputs[1] # Meta data input
                inputs3 = inputs[2] # Credit scores input
                
                inputs1 = inputs1.to(device) 
                inputs2 = inputs2.to(device) 
                inputs3 = inputs3.to(device) 

                sentiment = sentiment.to(device)
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    #print(inputs)
                    outputs = model(inputs1, inputs2, inputs3,)

                    outputs = F.softmax(outputs,dim=1)
                    
                    loss = criterion(outputs, torch.max(sentiment.float(), 1)[1])
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs1.size(0)

                
                sentiment_corrects += torch.sum(torch.max(outputs, 1)[1] == torch.max(sentiment, 1)[1])

                
            epoch_loss = running_loss / dataset_sizes[phase]

            
            sentiment_acc = sentiment_corrects.double() / dataset_sizes[phase]

            print('{} total loss: {:.4f} '.format(phase,epoch_loss ))
            print('{} sentiment_acc: {:.4f}'.format(
                phase, sentiment_acc))

            # Saving training acc and loss for each epoch
            sentiment_acc1 = sentiment_acc.data
            sentiment_acc1 = sentiment_acc1.cpu()
            sentiment_acc1 = sentiment_acc1.numpy()
            train_acc.append(sentiment_acc1)
            
            #epoch_loss1 = epoch_loss.data
            #epoch_loss1 = epoch_loss1.cpu()
            #epoch_loss1 = epoch_loss1.numpy()
            train_loss.append(epoch_loss)
                
            if phase == 'val' and sentiment_acc > best_acc:
                print('Saving with accuracy of {}'.format(sentiment_acc),
                      'improved over previous {}'.format(best_acc))
                best_acc = sentiment_acc
                
                # Saving val acc and loss for each epoch
                sentiment_acc1 = sentiment_acc.data
                sentiment_acc1 = sentiment_acc1.cpu()
                sentiment_acc1 = sentiment_acc1.numpy()
                val_acc.append(sentiment_acc1)
            
                #epoch_loss1 = epoch_loss.data
                #epoch_loss1 = epoch_loss1.cpu()
                #epoch_loss1 = epoch_loss1.numpy()
                val_loss.append(epoch_loss)
                
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), 'bert_model_test_noFC1_triBERT.pth')

        print('Time taken for epoch'+ str(epoch+1)+ ' is ' + str((time.time() - epoch_start)/60) + ' minutes')
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(float(best_acc)))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, train_acc, val_acc, train_loss, val_loss

model.to(device)


lrlast = .0005
lrmain = .00001
optim1 = optim.Adam(
    [
        {"params":model.bert.parameters(),"lr": lrmain},
        {"params":model.classifier.parameters(), "lr": lrlast},
       
   ])

#optim1 = optim.Adam(model.parameters(), lr=0.001)#,momentum=.9)
# Observe that all parameters are being optimized
optimizer_ft = optim1
criterion = nn.CrossEntropyLoss()

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=3, gamma=0.1)


model_ft1, train_acc, val_acc, train_loss, val_loss = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,num_epochs=20)

# Accuracy plots

'''print(val_acc)
print(val_loss)

#plt.plot(train_acc)
plt.plot(val_acc)
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['val'], loc='upper left')
#plt.show()
plt.savefig('accuracy.png')
plt.close()

print('Saved Accuracy plot')

# Loss plots

#plt.plot(train_loss)
plt.plot(val_loss)
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['val'], loc='upper right')
#plt.show()
plt.savefig('loss.png')
plt.close()

print('Saved Loss plot')'''
