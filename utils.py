import random
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset

# SEED = 0
# random.seed(SEED)
# np.random.seed(SEED)
# torch.manual_seed(SEED)
# torch.cuda.manual_seed(SEED)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

def createSelection(contexts, data,mode,tokenizer,max_token_len,q_max_len):
    selectionPairs = []
    contexts_count = len(contexts)
    
    if mode == 'train':
        for i, content in enumerate(data):
            negative = []
            q_id = content['id']
            positive_id = content['relevant']
            questions = [[content['question']]*5]
            candidate = content['paragraphs']
            candidate.remove(positive_id)
            
            label = 0
            positive = [contexts[positive_id]]
            k = 4
            if len(candidate) < k:
                for i in range(k-len(candidate)):
                    sampling = random.randint(0, contexts_count - 1)
                    while sampling in candidate:
                        sampling = random.randint(0, contexts_count - 1)
                    candidate.append(sampling)

            negative_id = random.sample(candidate, k)
            sample_id = [positive_id] + negative_id
            paragraphs = []
            for i in sample_id:
                paragraphs.append(contexts[i])
        

            questions = sum(questions, [])
            # paragraphs = sum(paragraphs, [])
            tokenized_examples = tokenizer(questions, paragraphs, truncation = True, max_length = max_token_len,  
                                        padding = 'max_length', add_special_tokens = True,
                                        return_tensors='pt', return_token_type_ids=True, return_attention_mask=True,)
            pair = {
                        'encoded_inputs': tokenized_examples,
                        'q_id' : q_id,
                        'label' : label,
                        'sample_id' : sample_id,
                    }

            selectionPairs.append(pair) 
           
    else:
        for i, content in enumerate(data):
            negative = []
            negative_id = []
            q_id = content['id']
            all_id = content['paragraphs']
            context_count = len(all_id)
            questions = [[content['question']]*context_count]            
            
            label = 0
            # positive = [contexts[positive_id]]
            paragraphs = []
            for i in all_id:
                paragraphs.append(contexts[i])

            questions = sum(questions, [])
            # paragraphs = sum(paragraphs, [])
            tokenized_examples = tokenizer(questions, paragraphs, truncation = True, max_length = max_token_len,  
                                        padding = 'max_length', add_special_tokens = True,
                                        return_tensors='pt', return_token_type_ids=True, return_attention_mask=True,)
            pair = {
                        'encoded_inputs': tokenized_examples,
                        'q_id' : q_id,
                        'sample_id' : all_id,
                    }

            selectionPairs.append(pair)

    return selectionPairs

def createPairs(contexts, data, mode, tokenizer, max_token_len, q_max_len, pred = None):
    QA_pairs = []
    for i, content in enumerate(data):
        if mode == 'train':
            context = contexts[content['relevant']]
        else:
            context = contexts[pred[i]]
        questions = content['question']
        q_id = content['id']
        q_text = questions[:q_max_len]
        encoded_inputs = tokenizer.encode_plus(
                text=context,
                text_pair=q_text,
                max_length= 384,  
                add_special_tokens = True,
                truncation='only_first',# will truncate context part
                padding = 'max_length',
                return_tensors='pt',
                return_token_type_ids=True,
                return_attention_mask=True,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                stride = 128
        )

            # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = encoded_inputs['overflow_to_sample_mapping']
        offset_mapping = encoded_inputs['offset_mapping']
        
        # Let's label those examples!
        
        for i, offsets in enumerate(offset_mapping):
            tokenized_examples = {}
            # We will label impossible answers with the index of the CLS token.
            input_ids = encoded_inputs["input_ids"][i]
            token_type_ids = encoded_inputs["token_type_ids"][i]
            attention_mask = encoded_inputs["attention_mask"][i]

            
            tokenized_examples['q_id'] = q_id
            tokenized_examples['input_ids'] = input_ids
            tokenized_examples['token_type_ids'] = token_type_ids
            tokenized_examples['attention_mask'] = attention_mask
            tokenized_examples['offset_mapping'] = offsets
            tokenized_examples['sample_mapping'] = sample_mapping[i]
            # tokenized_examples["start_positions"] = []
            # tokenized_examples["end_positions"] = []

            input_list = input_ids.tolist()
            cls_index = input_list.index(tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = encoded_inputs.sequence_ids(i)
            context_index = 0

            if mode == 'train':
                # One example can give several spans, this is the index of the example containing this span of text.
                sample_index = sample_mapping[i]
                answers = content["answers"][0]
                ans_tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(answers["text"],add_special_tokens=False))
                # If no answers are given, set the cls_index as answer.

                # Start/end character index of the answer in the text.
                start_char = answers["start"]
                end_char = start_char + len(answers["text"])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != 0:
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != 0:
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"] = cls_index
                    tokenized_examples["end_positions"] = cls_index
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"] = token_start_index - 1
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"] = token_end_index + 1
                    
                tokenized_examples["offset_mapping"] = [
                (o if sequence_ids[k] == context_index else torch.tensor([-1,-1]))
                for k, o in enumerate(tokenized_examples["offset_mapping"])
                ]

            QA_pairs.append(tokenized_examples)
        
    return QA_pairs



class QADataset(Dataset):
    def __init__(self,data,mode):
        assert mode in ["train","test"]
        self.data = data
        self.mode = mode
        
    def __getitem__(self,idx):


        if self.mode == 'train':
            q_id = self.data[idx]['q_id']
            input_ids = self.data[idx]['input_ids'].squeeze(0)# token tensors
            token_type_ids = self.data[idx]['token_type_ids'].squeeze(0) # segement tesnors
            attention_mask = self.data[idx]['attention_mask'].squeeze(0)
            offset_mapping = self.data[idx]['offset_mapping']
            sample_mapping = self.data[idx]['sample_mapping'].squeeze(0)
            start_positions = self.data[idx]['start_positions']
            end_positions = self.data[idx]['end_positions']

            return input_ids, token_type_ids, attention_mask, start_positions, end_positions, q_id, offset_mapping, sample_mapping

        else:
            q_id = self.data[idx]['q_id']
            input_ids = self.data[idx]['input_ids'].squeeze(0)
            token_type_ids = self.data[idx]['token_type_ids'].squeeze(0) # segement tesnors
            attention_mask = self.data[idx]['attention_mask'].squeeze(0)
            offset_mapping = self.data[idx]['offset_mapping']
            sample_mapping = self.data[idx]['sample_mapping'].squeeze(0)

            return input_ids, token_type_ids, attention_mask, q_id, offset_mapping, sample_mapping


    def __len__(self):
        return len(self.data)


