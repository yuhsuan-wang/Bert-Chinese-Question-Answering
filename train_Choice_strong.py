import json
import torch
from argparse import ArgumentParser, Namespace
import random
import numpy as np
from pathlib import Path
from transformers import BertTokenizer,BertConfig, BertForMultipleChoice, AdamW, get_linear_schedule_with_warmup
from utils import createSelection
from torch.utils.data import Dataset,DataLoader
import pickle
import time
from datetime import datetime
from tqdm import tqdm
import torch.nn.functional as F

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

CONTEXT = "context"
TRAIN = "train"
SPLITS = [CONTEXT, TRAIN]
MAX_NUM = 2


def main(args):
    PRETRAINED_MODEL_NAME = args.pretrained_model_name
    NUM_EPOCHS = args.num_epoch
    TRAIN_BATCH_SIZE = args.batch_size
    NUM_LABELS = 2 # start/end
    q_max_len  = args.q_max_len
    max_token_len = args.max_token_len


    torch.cuda.set_device(1)
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(torch.cuda.current_device())


    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_name,do_lower_case=True)
    with open(args.data_dir / f"context.json") as f:
        context = json.load(f)
    with open(args.data_dir / f"train.json") as f:
        train_data = json.load(f)
    
    split = int(len(train_data) * 0.8)
    valid_data = train_data[split:]
    train_data = train_data[:split]

    trainSelection_path = args.dir/'Selection_train_strong.pickle'
    validSelection_path = args.dir/'Selection_valid_strong.pickle'


    if Path(trainSelection_path).exists() and Path(validSelection_path).exists():
        with open(trainSelection_path, 'rb') as f:
            trainSelection = pickle.load(f)
        with open(validSelection_path, 'rb') as f:
            validSelection = pickle.load(f)   

    else:
        train_Selection = createSelection(context, train_data,"train",tokenizer,max_token_len,q_max_len)
        valid_Selection = createSelection(context, valid_data,"train",tokenizer,max_token_len,q_max_len)
        print('num of train samples:',len(train_Selection))
        print('num of valid samples:',len(valid_Selection))
        
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Current Time =", current_time)
        
        trainSelection = SelectionDataset(train_Selection,"train")
        validSelection = SelectionDataset(valid_Selection,"train") # to get the label and ans_pos
        
        with open(trainSelection_path, 'wb') as f:
            pickle.dump(trainSelection, f)
        with open(validSelection_path, 'wb') as f:
            pickle.dump(validSelection, f)

    
    pretrainloader = DataLoader(trainSelection,batch_size=TRAIN_BATCH_SIZE,shuffle=True)
    prevalidloader = DataLoader(validSelection,batch_size=TRAIN_BATCH_SIZE, shuffle = True)



    # config = BertConfig.from_pretrained(args.pretrained_model_name)
    # model = BertForMultipleChoice(config)
    # model = BertForMultipleChoice.from_pretrained('hfl/chinese-roberta-wwm-ext')
    # config=BertConfig.from_pretrain(pre_train_model)
    model = BertForMultipleChoice.from_pretrained(args.pretrained_model_name)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 1e-2,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    # optimizer = torch.optim.Adam(lr=args.lr, betas=(0.9, 0.98), eps=1e-9, params=optimizer_grouped_parameters)
    optimizer = AdamW(optimizer_grouped_parameters,lr=5e-5,eps=1e-8)
    criterion = torch.nn.CrossEntropyLoss()

    # gradient_accumulation_steps = len(trainSelection) / args.batch_size * args.num_epoch
    gradient_accumulation_steps = 32
    total_steps = len(pretrainloader)// gradient_accumulation_steps * args.num_epoch
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)
    # print(config)
    best_loss = 10000

    print('Start training : ')
    for epoch in range(args.num_epoch):

        start_time = time.time()
        pretrain_acc, pretrain_loss, pred_logit, train_sample_id_list = pretrain(model,pretrainloader, device,optimizer, criterion, scheduler, trainSelection)
        prevalid_acc, prevalid_loss, pred_valid_logit, valid_sample_id_list = prevalid(model,prevalidloader,device, criterion, scheduler, validSelection)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        
        print(f'Epoch:{epoch+1:2d}||time:{epoch_mins}m{epoch_secs:2d}s', 
            f'|| train loss:{pretrain_loss:.4f}',
            f'|| valid loss:{prevalid_loss:.4f}',
            f'|| train acc:{pretrain_acc:.4f}'
            f'|| valid acc:{prevalid_acc:.4f}',
            )

        if prevalid_loss < best_loss:
            best_model = model
            best_loss = prevalid_loss
            best_epoch = epoch+1
            print('Current Best Model!')

            # with open('state_dict(), args.ckpt_dir/'choice_strong.pt') 
   
        
   
        

def pretrain(model, pretrainloader, device, optimizer, criterion, scheduler, trainSelection ):
    model.train()
    loss_total = 0
    acc_total = 0
    logit_list = []
    label_list = []
    sample_id_list = []
    # gradient_accumulation_steps = len(trainSelection) / args.batch_size * args.num_epoch
    gradient_accumulation_steps = 32
    for step, batch in enumerate(tqdm(pretrainloader)):
        input_ids, token_type_ids, attention_mask, q_id, label, sample_id = batch
        # input_ids, token_type_ids, attention_mask, label, q_id, _, sample_id, _ = batch
        input_ids = input_ids.to(device)
        token_type_ids = token_type_ids.to(device)
        attention_mask = attention_mask.to(device)
        label = label.to(device)
        model = model.to(device)
        outputs = model(input_ids = input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids, labels = label)
        loss = outputs.loss
        logit = outputs.logits
        # logit = F.softmax(logit, dim = 1)
        _, logit = torch.max(logit, dim=1)
        # loss = criterion(logit, label)

        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps
            loss_total += loss.item()
        else:
            loss_total += loss.item()


        loss.backward()

        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if (step + 1) % gradient_accumulation_steps == 0:
            # Clip the norm of the gradients to 1.0.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step() # update all parameters
            scheduler.step() # Update learning rate schedule
            optimizer.zero_grad() # initialize the gradient so that it wont repeat accumulate itself(update the params)
            model.zero_grad()
            # print("in step:%s,loss:%s"%(str(step),str(loss)),end = "\r")
        
        logit = logit.tolist()
        label = label.tolist()
        # logit_list.append(logit[0])
        logit_list += logit
        # label_list.append(label)
        label_list += label
        sample_id_list.append(sample_id)

        

    acc,pred_logit = acc_test(logit_list, label_list, device)
        
    # train_acc = (acc/len(trainSelection))*100
    return acc, loss_total/len(pretrainloader) , pred_logit, sample_id_list
        # print(f'Epoch {epoch+0:03} | Acc: {train_acc:.5f}')

def prevalid(model, prevalidloader, device, criterion, scheduler, validSelection):
    loss_total = 0
    acc_total = 0
    logit_list = []
    label_list = []
    sample_id_list = []
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(tqdm(prevalidloader)):
            input_ids, token_type_ids, attention_mask, q_id, label, sample_id = batch
            # input_ids, token_type_ids, attention_mask, label, q_id, _, sample_id, _ = batch
            input_ids = input_ids.to(device)
            token_type_ids = token_type_ids.to(device)
            attention_mask = attention_mask.to(device)
            label = label.to(device)
            model = model.to(device)
            outputs = model(input_ids = input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids, labels = label)
            loss = outputs.loss
            logit = outputs.logits
            # logit = F.softmax(logit, dim = 1)
            loss_total += loss

            # loss = criterion(logit, label)
            # loss_total += loss

            _, logit = torch.max(logit, dim=1)

            logit = logit.tolist()
            label = label.tolist()
            # logit_list.append(logit[0])
            # label_list.append(label)
            logit_list += logit
            label_list += label
            sample_id_list.append(sample_id)

    
    acc_dev, pred_logit = acc_test(logit_list, label_list, device)
    # acc_final = (acc_dev/len(validSelection)) * 100
    return acc_dev, loss_total/len(prevalidloader), pred_logit, sample_id_list
        

class SelectionDataset(Dataset):
    def __init__(self,data,mode):
        assert mode in ["train","test"]
        self.data = data
        self.mode = mode
        
    def __getitem__(self,idx):
        
        encoded_inputs = self.data[idx]['encoded_inputs']
        q_id = self.data[idx]['q_id']
        label = self.data[idx]['label']
        sample_id = self.data[idx]['sample_id']
        input_ids = encoded_inputs['input_ids'] # token tensors
        token_type_ids = encoded_inputs['token_type_ids'] # segement tesnors
        attention_mask = encoded_inputs['attention_mask'] # attention mask tensors
        
        return input_ids,token_type_ids,attention_mask, q_id, label, sample_id
    
    def __len__(self):
        return len(self.data)

     


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./dataset/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/",
    )
    parser.add_argument(
        "--dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/",
    )

    # model
    parser.add_argument("--pretrained_model_name", type = str, default = "hfl/chinese-roberta-wwm-ext")

    # data
    parser.add_argument("--max_token_len", type=int, default=512)
    parser.add_argument("--q_max_len", type=int, default=40)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=1)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--num_epoch", type=int, default=5)

    args = parser.parse_args()
    return args

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def saveCkpt(model,epoch,min_loss):
    ckpt_path = f'./ckpt/choice.pt'
    torch.save(
        {
            'state_dict':model.state_dict(),
            'epoch':epoch
        },
        ckpt_path
    )

def acc_test(preds, targets, device):
    preds = torch.FloatTensor(preds)
    targets = torch.FloatTensor(targets)
    # _, preds = torch.max(preds, dim=1)
    preds = preds.to(device, dtype = torch.int64)
    targets = targets.to(device)
    
    correct_results_sum = 0
    #result = torch.round(y_pred).float()
    correct_results_sum = (preds == targets).sum().float()
    
    return (correct_results_sum / len(targets))*100, preds
    

if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)