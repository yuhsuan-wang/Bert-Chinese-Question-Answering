import json
import torch
from argparse import ArgumentParser, Namespace
import numpy as np
from pathlib import Path
from transformers import BertConfig, BertTokenizerFast, BertForQuestionAnswering, AdamW, get_linear_schedule_with_warmup
from utils import createPairs, QADataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import collections
import time


def main(args):
    PRETRAINED_MODEL_NAME = args.pretrained_model_name
    NUM_EPOCHS = args.num_epoch
    TRAIN_BATCH_SIZE = args.batch_size
    q_max_len  = args.q_max_len
    max_token_len = args.max_token_len

    torch.cuda.set_device(1)
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(torch.cuda.current_device())

    tokenizer = BertTokenizerFast.from_pretrained(PRETRAINED_MODEL_NAME,do_lower_case=True)
    with open(args.data_dir / f"context.json") as f:
        context = json.load(f)
    with open(args.data_dir / f"train.json") as f:
        train_data = json.load(f)

    split = int(len(train_data) * 0.8)
    valid_data = train_data[split:]
    train_data = train_data[:split]



    train_pairs = createPairs(context, train_data, "train", tokenizer, max_token_len, q_max_len)
    valid_pairs = createPairs(context, valid_data, "train", tokenizer, max_token_len, q_max_len)
    print('num of train samples:',len(train_pairs))
        
    trainset = QADataset(train_pairs,"train")
    validset = QADataset(valid_pairs,"train") # to get the label and ans_pos
        



    trainloader = DataLoader(trainset,batch_size=TRAIN_BATCH_SIZE)
    validloader = DataLoader(validset,batch_size=TRAIN_BATCH_SIZE)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = BertConfig.from_pretrained(args.pretrained_model_name  )
    model = BertForQuestionAnswering.from_pretrained(args.pretrained_model_name, 
                            from_tf=bool(".ckpt" in args.pretrained_model_name),
                            config = config)


    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 1e-2,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters,lr=args.lr,eps=1e-8)
    gradient_accumulation_steps = 32
    # # Total number of training steps is number of batches * number of epochs.
    total_steps = len(trainloader)// gradient_accumulation_steps * NUM_EPOCHS
    # # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)

    print('start training....')
    best_loss = float('inf')
    

    for epoch in range(5):

        start_time = time.time()
        train_loss = train(model,train_data, trainset, trainloader,context, device, tokenizer, optimizer,scheduler,gradient_accumulation_steps)
        valid_loss = valid(model,valid_data, validset, validloader, context, device, tokenizer)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(f'Epoch:{epoch+1:2d}||time:{epoch_mins}m{epoch_secs:2d}s', 
            f'|| train loss:{train_loss:.4f}',
            f'|| validation loss:{valid_loss:.4f}'
            )

        if valid_loss < best_loss:
            best_model = model
            best_loss = valid_loss
            best_epoch = epoch+1
            print('Current Best Model!')


def train(model,train_data, trainset, trainloader, contexts, device, tokenizer, optimizer,scheduler,gradient_accumulation_steps):
    total_loss = 0

    # tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_name,do_lower_case=True)
    model.train()
    start_logits_list = []
    end_logits_list = []
    for step,data in enumerate(trainloader):
        input_ids = data[0].to(device) # (batch_size,token_len)
        token_type_ids = data[1].to(device) # (batch_size,token_len)
        attention_mask = data[2].to(device) # (batch_size,token_len)
        start_pos = data[3].to(device) # (batch_size)
        end_pos = data[4].to(device) # (batch_size)
        q_id = data[5] # (batch_size)
        model = model.to(device)


        # bert model contains BCEwithlogitloss and crossentropy 
        outputs = model(input_ids=input_ids,
                       token_type_ids=token_type_ids,
                       attention_mask=attention_mask,
                       start_positions=start_pos,
                       end_positions=end_pos,
                      )
        loss = outputs[0]
        start_logits = outputs[1]
        end_logits = outputs[2]

        total_loss += loss.item()

        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps
        #     total_loss += loss.item()
        # else:
        #     total_loss += loss.item()

        # total_loss += loss.item()

        loss.backward()
        if (step + 1) % gradient_accumulation_steps == 0:
            # Clip the norm of the gradients to 1.0.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step() # update all parameters
            scheduler.step() # Update learning rate schedule
            optimizer.zero_grad() # initialize the gradient so that it wont repeat accumulate itself(update the params)
            model.zero_grad()
            print("in step:%s,loss:%s"%(str(step),str(loss)),end = "\r")

        start_logits = start_logits.tolist()
        end_logits = end_logits.tolist()
        for logit in start_logits:
            start_logits_list.append(logit)

        for logit in end_logits:
            end_logits_list.append(logit)

    predictions = postprocess_qa_predictions(contexts, train_data, trainset, start_logits_list, end_logits_list, tokenizer)
        
        
    return total_loss/len(trainloader)


def valid(model, valid_data, validset, validloader, contexts, device, tokenizer):
    total_loss = 0
    prediction = {}
    start_logits_list = []
    end_logits_list = []

    model.eval()
    with torch.no_grad():
        for i,data in enumerate(validloader):
            input_ids = data[0].to(device) # (batch_size,token_len)
            token_type_ids = data[1].to(device) # (batch_size,token_len)
            attention_mask = data[2].to(device) # (batch_size,token_len)
            start_pos = data[3].to(device) # (batch_size)
            end_pos = data[4].to(device) # (batch_size)
            q_id = data[5] # (batch_size)

            model = model.to(device)

            # bert model contains BCEwithlogitloss and crossentropy 
            outputs = model(input_ids=input_ids,
                           token_type_ids=token_type_ids,
                           attention_mask=attention_mask,
                           start_positions=start_pos,
                           end_positions=end_pos
                          )
            # (total_loss), cls_logits, start_logits, end_logits, (hidden_states), (attentions)
            loss = outputs[0]
            start_logits = outputs[1]
            end_logits = outputs[2]

            total_loss += loss.item()

            start_logits = start_logits.tolist()
            end_logits = end_logits.tolist()
            for logit in start_logits:
                start_logits_list.append(logit)

            for logit in end_logits:
                end_logits_list.append(logit)

        predictions = postprocess_qa_predictions(contexts, valid_data, validset, start_logits_list, end_logits_list, tokenizer)

        return total_loss/len(validloader)

def postprocess_qa_predictions(contexts, dataset, data, start_logits_list, end_logits_list, tokenizer, n_best_size = 10, max_answer_length = 30):
    # Build a map example to its corresponding features.
    example_id_to_index = {k['id']: i for i, k in enumerate(dataset)}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(data):
        features_per_example[example_id_to_index[feature[5]]].append(i)

    # The dictionaries we have to fill.
    predictions = collections.OrderedDict()

    # Let's loop over all the examples!
    for example_index, example in enumerate(tqdm(dataset)):
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_id_to_index[example['id']]]

        min_null_score = None # Only used if squad_v2 is True.
        valid_answers = []
        
        context = contexts[example["relevant"]]
        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            start_logits = start_logits_list[feature_index]
            end_logits = end_logits_list[feature_index]
            # This is what will allow us to map some the positions in our logits to span of texts in the original
            # context.
            offset_mapping = data[feature_index][6]

            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index][0] == -1
                        or offset_mapping[end_index][0] == -1
                    ):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue

                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    valid_answers.append(
                        {
                            "score": start_logits[start_index] + end_logits[end_index],
                            "text": context[start_char: end_char]
                        }
                    )
        
        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
        else:
            # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
            # failure.
            best_answer = {"text": "", "score": 0.0}
        
        # Let's pick our final answer: the best one or the null answer (only for squad_v2)
        predictions[example["id"]] = best_answer["text"]

    return predictions


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


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

    parser.add_argument(
        "--result_path",
        type=Path,
        help="Directory to save the prediction.",
        default="./prediction/",
    )


    # model
    parser.add_argument("--pretrained_model_name", type = str, default = "hfl/chinese-roberta-wwm-ext-large")

    # data
    parser.add_argument("--max_token_len", type=int, default=384)
    parser.add_argument("--q_max_len", type=int, default=40)

    # optimizer
    parser.add_argument("--lr", type=float, default=3e-5)

    # data loader
    parser.add_argument("--batch_size", type=int, default=2)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--num_epoch", type=int, default=5)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)