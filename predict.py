import json
import torch
from argparse import ArgumentParser, Namespace
import random
import numpy as np
from pathlib import Path
from transformers import BertConfig, BertPreTrainedModel, BertForQuestionAnswering, BertForMultipleChoice, BertTokenizerFast
from utils import createSelectionv2, createPairsv2, QADatasetv2
from torch.utils.data import Dataset,DataLoader
import collections
import pickle
from tqdm import tqdm
import torch.nn.functional as F

# SEED = 0
# random.seed(SEED)
# np.random.seed(SEED)
# torch.manual_seed(SEED)
# torch.cuda.manual_seed(SEED)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

CONTEXT = "context"
TRAIN = "train"
MAX_NUM = 2

def main(args):
    NUM_LABELS = 2 # start/end
    q_max_len  = args.q_max_len
    max_token_len = args.max_token_len


    torch.cuda.set_device(1)
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(torch.cuda.current_device())

    tokenizer = BertTokenizerFast.from_pretrained(args.choice_pretrained_model_name,do_lower_case=True)
    with open(args.context_dir) as f:
        context = json.load(f)
    with open(args.data_dir) as f:
        public_data = json.load(f)
    

    public_Selection = createSelectionv2(context, public_data,"test",tokenizer,args.choice_max_token_len,q_max_len)
    print('num of public samples:',len(public_Selection))
        
    publicSelection = SelectionDataset(public_Selection,"test")
    prepublicloader = DataLoader(publicSelection,batch_size = args.choice_batch_size,shuffle=False)

    # config = BertConfig.from_pretrained(args.pretrained_model_name)
    # model = BertForMultipleChoice(config)

    model = BertForMultipleChoice.from_pretrained(args.choice_pretrained_model_name)
    model.load_state_dict(torch.load(args.choice_ckpt_dir))
    model.eval()

    logit_list = []
    sample_id_list = []
    pred_list = []

    with torch.no_grad():
        for step, batch in enumerate(tqdm(prepublicloader, desc = 'Iteration')):
            input_ids, token_type_ids, attention_mask, q_id, sample_id = batch
            input_ids = input_ids.to(device)
            token_type_ids = token_type_ids.to(device)
            attention_mask = attention_mask.to(device)
            model = model.to(device)
            outputs = model(input_ids = input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids)
            logit = outputs.logits

            _, logit = torch.max(logit, dim=1)
            pred = sample_id[logit]

            logit_list.append(logit)
            sample_id_list.append(sample_id)
            pred_list.append(pred)

    
    

    #Question-Answering
    
    public_pairs = createPairsv2(context, public_data, "test", tokenizer, max_token_len, q_max_len, pred_list)
    print('num of public samples:',len(public_pairs))
        
    publicset = QADatasetv2(public_pairs,"test")   
    
    publicloader = DataLoader(publicset,batch_size=args.QA_batch_size, shuffle=False)
    config = BertConfig.from_pretrained(args.pretrained_model_name  )
    model = BertForQuestionAnswering.from_pretrained(args.pretrained_model_name, 
                            from_tf=bool(".ckpt" in args.pretrained_model_name),
                            config = config)
    # model = BertChineseQuestionAnswering(args).to(device)
    model.load_state_dict(torch.load(args.qa_ckpt_dir))

    tokenizer = BertTokenizerFast.from_pretrained(args.pretrained_model_name,do_lower_case=True)  

    prediction = {}
    model.eval()
    start_logits_list = []
    end_logits_list = []
    with torch.no_grad():
        for i,data in enumerate(tqdm(publicloader)):
            input_ids = data[0].to(device) # (batch_size,token_len)
            token_type_ids = data[1].to(device) # (batch_size,token_len)
            attention_mask = data[2].to(device) # (batch_size,token_len)
            q_id = data[3] # (batch_size)

            model = model.to(device)
            # bert model contains BCEwithlogitloss and crossentropy 
            outputs = model(input_ids=input_ids,
                           token_type_ids=token_type_ids,
                           attention_mask=attention_mask,
                          )
            # (total_loss), cls_logits, start_logits, end_logits, (hidden_states), (attentions)
            start_logits = outputs['start_logits']
            end_logits = outputs['end_logits']

            # result = post_processing(q_id, tokenizer,input_ids,token_type_ids,start_logits, end_logits)
            # prediction.update(result)
            start_logits = start_logits.tolist()
            end_logits = end_logits.tolist()
            for logit in start_logits:
                start_logits_list.append(logit)

            for logit in end_logits:
                end_logits_list.append(logit)

        predictions = postprocess_qa_predictions(pred_list, context, public_data, publicset, start_logits_list, end_logits_list, tokenizer)

        with open(args.result_path,'w',encoding='utf8') as f:
            json.dump(predictions,f,ensure_ascii=False)
 


def postprocess_qa_predictions(pred_list, contexts, dataset, data, start_logits_list, end_logits_list, tokenizer, n_best_size = 10, max_answer_length = 45):
    # Build a map example to its corresponding features.
    example_id_to_index = {k['id']: i for i, k in enumerate(dataset)}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(data):
        features_per_example[example_id_to_index[feature[3]]].append(i)

    # The dictionaries we have to fill.
    predictions = collections.OrderedDict()

    # Logging.
    # print(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

    # Let's loop over all the examples!
    for example_index, example in enumerate(tqdm(dataset)):
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_id_to_index[example['id']]]

        min_null_score = None # Only used if squad_v2 is True.
        valid_answers = []
        
        context = contexts[pred_list[example_index]]
        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            start_logits = start_logits_list[feature_index]
            end_logits = end_logits_list[feature_index]
            # This is what will allow us to map some the positions in our logits to span of texts in the original
            # context.
            offset_mapping = data[feature_index][4]

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
        # answer = best_answer["text"] if best_answer["score"] > min_null_score else ""
        # predictions[dataset[example_index]["id"]] = answer

    return predictions


class SelectionDataset(Dataset):
    def __init__(self,data,mode):
        assert mode in ["train","test"]
        self.data = data
        self.mode = mode
        
    def __getitem__(self,idx):
        
        encoded_inputs = self.data[idx]['encoded_inputs']
        q_id = self.data[idx]['q_id']
        sample_id = self.data[idx]['sample_id']
        input_ids = encoded_inputs['input_ids'] # token tensors
        token_type_ids = encoded_inputs['token_type_ids'] # segement tesnors
        attention_mask = encoded_inputs['attention_mask'] # attention mask tensors

        if self.mode == 'train':
            label = self.data[idx]['label']
            
            return input_ids,token_type_ids,attention_mask, q_id, label, sample_id
        else:
            return input_ids,token_type_ids,attention_mask, q_id,sample_id
    
    def __len__(self):
        return len(self.data)



def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--context_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./dataset/",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./dataset/",
    )
    parser.add_argument(
        "--choice_ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/choice.pt",
    )
    parser.add_argument(
        "--qa_ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/qa.pt",
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
        default="./",
    )

    # model
    parser.add_argument("--choice_pretrained_model_name", type = str, default = "hfl/chinese-roberta-wwm-ext")
    parser.add_argument("--pretrained_model_name", type = str, default = "hfl/chinese-roberta-wwm-ext-large")

    # data
    parser.add_argument("--max_token_len", type=int, default=384)
    parser.add_argument("--choice_max_token_len", type=int, default=512)
    parser.add_argument("--q_max_len", type=int, default=40)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--choice_batch_size", type=int, default=1)
    parser.add_argument("--QA_batch_size", type=int, default=4)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--num_epoch", type=int, default=10)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    # args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)