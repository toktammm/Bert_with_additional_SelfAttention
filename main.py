from os import replace
import pandas as pd
import numpy as np
import sys
import warnings
import math
from sklearn.utils import shuffle
import torch
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix, f1_score, accuracy_score, precision_recall_fscore_support
from transformers import AutoModel, AutoTokenizer
from transformers import AdamW
from torch.nn import BCEWithLogitsLoss, BCELoss
from tqdm import tqdm, trange
from transformers import BertTokenizer
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch import nn
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight
if not sys.warnoptions:
    warnings.simplefilter("ignore") 

def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='weighted')

def calculate_accuracy(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    n_correct_elems = np.sum(preds_flat == labels_flat)
    return n_correct_elems / len(labels)

def main():
    fraction = 100
    epochs = 10
    batch_size = 64
    learningrate = 2e-5
    max_length = 50
    freeze = True
    freeze_epoch = 4
    DROPOUT = 0.2
    heads = 8
    pretrained_model_name = "vinai/bertweet-base"
    data_file = 'twitter_hatespeech_2018.csv'
      

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        n_gpu = torch.cuda.device_count()    
        torch.cuda.get_device_name(0)

    '''load data'''
    # df = pd.read_csv('usc_covid_misinfo_BertweetNormalized.csv', index_col=0)
    df = pd.read_csv(data_file)
    df = df.sample(frac=1).reset_index(drop=True)   #shuffle rows    

    if device.type == "cpu":
        df = df.iloc[:500]                          # JUST FOR CPU TO DECREASE TIME FOR TEST

    num_labels = len(set(df.label.values))
    categories = list(set(df.label.values))
    n_classes = len(categories)
    labels_to_categorical = {categories[i]: i for i in range(num_labels)}
    df = df.sample(frac=1).reset_index(drop=True) #shuffle rows
    df['class'] = df.label.replace(labels_to_categorical)
    labels = list(df['class'].values)
    comments = list(df.normal.values)       # comments = list(df.normal_lower.values) <----------- change to lowercase if needed

    '''define the tokenizer'''
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name, use_fast=False) 

    encodings = tokenizer.batch_encode_plus(comments,max_length=max_length,pad_to_max_length=True, truncation=True)
    input_ids = encodings['input_ids']              # tokenized and encoded sentences
    attention_masks = encodings['attention_mask']   # encoded attention masks

    '''Identifying indices class label entries that only occur once (to stratify train and validation splits)'''
    label_counts = df['class'].astype(str).value_counts()

    '''split our data into train and validation sets'''
    train_inputs, validation_inputs, train_labels, validation_labels, train_masks, validation_masks = train_test_split(input_ids, labels, attention_masks,random_state=2020, test_size=0.20, stratify = labels)

    validation_inputs = torch.tensor(validation_inputs)
    validation_labels = torch.tensor(validation_labels)
    validation_masks = torch.tensor(validation_masks)

    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels) 
    validation_dataloader = DataLoader(validation_data, shuffle=False, batch_size=batch_size) # also can use SequentialSampler(validation_data) instead of shuffle=False
    torch.save(validation_dataloader,'validation_data_loader')

    '''loop over data to add random data in each step'''
    chunk_size = [int(len(train_inputs) // (100 // fraction))] * (100 // fraction)
    train_inputs_chunks = np.split(train_inputs, np.cumsum(chunk_size[:-1]))
    train_labels_chunks = np.split(train_labels, np.cumsum(chunk_size[:-1]))
    train_masks_chunks = np.split(train_masks, np.cumsum(chunk_size[:-1]))

    validation_losses = []
    validation_f1s = []
    validation_accs = []
    train_loss_set = []

    '''training epochs for the pretrained model (recommended between 2 and 4). 
    So we freeze the pretrained_model parameters after freeze_epoch times!'''
    for c in range(len(train_inputs_chunks)):
        
        pretrained_model = AutoModel.from_pretrained(pretrained_model_name)   
        model, n_bert_pars = build_model(device, pretrained_model, heads=heads)
        optimizer = build_optimizer(model, learningrate)
        
        if c > 0:
            train_inputs_chunks[c] = np.concatenate((train_inputs_chunks[c-1], train_inputs_chunks[c]), axis=0)
            train_labels_chunks[c] = np.concatenate((train_labels_chunks[c-1], train_labels_chunks[c]), axis=0)
            train_masks_chunks[c] = np.concatenate((train_masks_chunks[c-1], train_masks_chunks[c]), axis=0)

        train_inputs_chunk = list(train_inputs_chunks[c])
        train_labels_chunk = list(train_labels_chunks[c])
        train_masks_chunk = list(train_masks_chunks[c])

        '''Convert data into torch tensors'''
        train_inputs_chunk = torch.tensor(train_inputs_chunk)
        train_labels_chunk = torch.tensor(train_labels_chunk)
        train_masks_chunk = torch.tensor(train_masks_chunk)

        '''Create an iterator of data with torch DataLoader'''
        train_data = TensorDataset(train_inputs_chunk, train_masks_chunk, train_labels_chunk)
        train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size) # also can use RandomSampler(train_data) instead of shuffle=True

        torch.save(train_dataloader,'train_data_loader')

        for e in trange(epochs, desc="Epoch"):
            
            '''To stop fine-tuning the loaded pretrained model, we freeze the parameters for that part in epoch number freeze_epoch'''
            if e == freeze_epoch - 1:
                '''freeze all the parameters except the parameter gradiants for the self-attention layers and classification head'''
                if freeze:
                    for p, param in enumerate(model.parameters()):
                        if p < n_bert_pars:
                            param.requires_grad = False     
                    optimizer = build_optimizer(model)
            
            '''Training'''
            '''Set model to training mode'''
            model.train()

            '''Tracking variables'''
            tr_loss = 0 
            nb_tr_examples, nb_tr_steps = 0, 0
            true_labels,pred_labels, pred_labels_soft = [], [], []
            
            '''Train the data for one epoch'''
            for step, batch in enumerate(train_dataloader):
                
                '''Add batch to GPU'''
                batch = tuple(t.to(device) for t in batch)
                
                '''Unpack the inputs from dataloader'''
                b_input_ids, b_input_mask, b_labels = batch 
                
                '''Clear out the gradients (by default they accumulate)'''
                optimizer.zero_grad()

                '''Forward pass for multiclass classification'''
                pred = model(b_input_ids, b_input_mask)
                loss_func = nn.NLLLoss() 
                loss = loss_func(pred, b_labels)
                pred = pred.detach().cpu().numpy()
                b_labels = b_labels.to('cpu').numpy()
                train_loss_set.append(loss.item())    
                pred_labels.append(pred)
                true_labels.append(b_labels)
                
                '''Backward pass'''
                loss.backward()
                
                '''Update parameters and take a step using the computed the gradient'''
                optimizer.step()

                tr_loss += loss.item()
                nb_tr_examples += b_input_ids.size(0)
                nb_tr_steps += 1
            
            pred_labels = np.concatenate(pred_labels, axis=0)
            true_labels = np.concatenate(true_labels, axis=0)
            tr_f1_accuracy = f1_score_func(pred_labels, true_labels)
            print('\n')
            print('data size: ', len(train_inputs_chunks[c])/len(train_inputs), " Train F1 Score (Weighted): {}".format(tr_f1_accuracy))
            print('data size: ', len(train_inputs_chunks[c])/len(train_inputs), " Train loss: {}".format(tr_loss/nb_tr_steps))

            '''Validation'''
            '''put model on validation mode'''
            model.eval()

            logit_preds,true_labels,pred_labels,pred_labels_soft = [],[],[],[]

            '''Predict'''
            loss_val_total = 0
            for i, batch in enumerate(validation_dataloader):
                batch = tuple(t.to(device) for t in batch)
                # Unpack the inputs from our dataloader
                b_input_ids, b_input_mask, b_labels = batch #, b_token_types
                with torch.no_grad():
                    
                    '''Forward pass'''
                    pred = model(b_input_ids, b_input_mask)
                    loss = loss_func(pred, b_labels)
                    pred = pred.detach().cpu().numpy()
                    loss_val_total += loss.item()
                    b_labels = b_labels.to('cpu').numpy()
                    pred_labels.append(pred)
                    true_labels.append(b_labels)
            
            loss_val_avg = loss_val_total/len(validation_dataloader)
            pred_labels = np.concatenate(pred_labels, axis=0)
            true_labels = np.concatenate(true_labels, axis=0)
            val_f1_accuracy = f1_score_func(pred_labels, true_labels)
            val_acc = calculate_accuracy(pred_labels, true_labels)

            print('data size: ', len(train_inputs_chunks[c])/len(train_inputs), ' F1 Validation Score (Weighted): ', val_f1_accuracy)
            print('data size: ', len(train_inputs_chunks[c])/len(train_inputs), ' Validation accuracy: ', val_acc)
            print('data size: ', len(train_inputs_chunks[c])/len(train_inputs), ' Validation Loss: ', loss_val_avg)

        validation_losses.append(loss_val_avg)
        validation_f1s.append(val_f1_accuracy)
        validation_accs.append(val_acc)
        
        del model


    print('validation f1 accuracies: ')
    print(validation_f1s)

    print('validation flat accuracies: ')
    print(validation_losses)

    print('validation accuracies: ')
    print(validation_accs)

if __name__ == "__main__":
    main()