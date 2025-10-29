import numpy as np
import pickle
import hashlib
import base64
import copy
import random

import torch
from torch.utils.data import Dataset
from szx82.tokenizer.tokenizer import Tokenizer as Tokenizer
from szx82.data.forecast import Forecast, VirtualProfitTokenizer
import szx82.core.tools as hp

"""Dataset BERT
    [BertForMaskedLM](https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertForMaskedLM)

    [A Complete Guide to BERT with Code](https://medium.com/data-science/a-complete-guide-to-bert-with-code-9f87602e4a11)

    The input for NSP consists of the first and second segments (denoted A and B) separated by a [SEP] token with a second [SEP] token at the end. BERT actually expects at least one [SEP] token per input sequence to denote the end of the sequence, regardless of whether NSP is being performed or not.

    NSP forms a classification problem, where the output corresponds to IsNext when segment A logically follows segment B, and NotNext when it does not.

    ## input_ids: indices of input sequence tokens in the vocabulary,

    ## attention_mask: Mask to avoid performing attention on padding token indices. Mask values selected in [0, 1]:
        1 for tokens that are not masked,
        0 for tokens that are masked.
    Optional.
    We use it to mask/unmask the second sentence.

    ## token_type_ids: Segment token indices to indicate first and second portions of the inputs. Indices are selected in [0, 1]: 
        0 corresponds to a sentence A token, 
        1 corresponds to a sentence B token.
    Optional.

    Example:
    [CLS] HuggingFace is based in NYC [SEP] Where is HuggingFace based? [SEP]
    token_type_ids == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    ## position_ids: indices of positions of each input sequence tokens in the position embeddings. Selected in the range 
    [0, config.max_position_embeddings - 1]

    Optional, if not set then absolute position is used ().
    Here, I do not know why the absolute positioning could be wrong.

"""

class Data:
    STYLES = CLASS_LABELS, PRE_TREN, MASKED, SIMILAR, NONE\
        = ('class_labels', 'pre_tren', 'masked', 'similar', 'none')
    MASK_FREQ = 0.15
    NOT_MASKED = -100
    def __init__(
                self,
                data_file_or_object, # data file string or Tokenizer object
                prediction_file,
                ModelClass,
                data_slice=(None, None),
                data_split={'train_val': 0.75, 'val_test': 0.5},
                ):

        self.seq_len = None
        self.bert_dataset = ModelClass.DATASET

        if isinstance(data_file_or_object, str):
            with open(data_file_or_object, "rb") as f:
                self.tokenizer = pickle.load(f)
        else:
            self.tokenizer = data_file_or_object
        
        with open(prediction_file, "rb") as f:
            prediction = pickle.load(f)
        self.vp_tokenizer = VirtualProfitTokenizer(prediction)
        
        self.pred_list = sorted(list(self.vp_tokenizer.token_map.keys()))
        self.tokenizer.add_to_vocab(self.pred_list)
        predictions_tokenized = self.vp_tokenizer.vp_tokens
        self.num_labels = self.vp_tokenizer.num_labels()
        
        segments = self.tokenizer.tokens_plus.reshape(
                        -1, self.tokenizer.data_object.get_segment_size(), 2)
        segments = segments[slice(*hp.slice(data_slice, segments))]

        data = [] # associate segment with prediction:
        for i in range(10, len(segments)):
            segment = segments[i]
            sgm_index = int(segment[-1][-1])
            if not sgm_index in predictions_tokenized:
                break
            pred_item = predictions_tokenized[int(segment[-1][-1])]
            prediction = pred_item[0]
            virtual_profit = pred_item[1]
            virtual_profit_token = pred_item[2]
            segment = segments[i]
            segment = segment.T[:-1].T  
            data.append((
                np.array(segment).reshape(-1), prediction, 
                virtual_profit, virtual_profit_token, sgm_index))

        def set_bert_input_data(data): # sentences
            if self.PRE_TREN in self.bert_dataset:
                return self.pre_input(data)
            
        test_data = data[
            -int(len(data) * data_split['val_test']):
            ]
        self.test_data = set_bert_input_data(test_data) 

        test_val_data = data[:-len(test_data)]
        test_val_data = set_bert_input_data(test_val_data) 
        random.shuffle(test_val_data)
        
        self.train_data = test_val_data[
            :int(len(test_val_data) * data_split['train_val'])]
        self.val_data = test_val_data[len(self.train_data):]

        self.train_dataset = Dataset(self.train_data, self)
        self.val_dataset = Dataset(self.val_data, self)
        self.test_dataset = Dataset(self.test_data, self)

        # validation: are train, val, test data separated?
        train_idxs = set([_['admin']['candle_index'] for _ in self.train_data])
        val_idxs = set([_['admin']['candle_index'] for _ in self.val_data])
        test_idxs = set([_['admin']['candle_index'] for _ in self.test_data])
        assert not train_idxs & val_idxs, \
            'Train data and valuation data overlap!'
        assert not val_idxs & test_idxs, \
            'Valuation data and test data overlap!'
        assert not test_idxs & train_idxs, \
            'Test data and train data overlap!'
        
        self.vocab_hash = base64.b64encode(
            hashlib.sha1(
                repr(tuple(sorted(self.tokenizer.vocab))).encode()).digest()
            ).decode()

        # lighten Data object
        self.tokenizer.tokens_plus = None 

    def _mask(self, sequence):
        sequence = copy.deepcopy(sequence)
        not_masked = self.NOT_MASKED
        mask_idx = self.tokenizer.convert_tokens_to_ids(self.tokenizer.MASK)
        masked_ids = [not_masked] * len(sequence)

        for i in range(0, len(sequence)): # dont mask CLS, SEP,
            if i in [0]:
                continue
            if random.uniform(0, 1) < self.MASK_FREQ:
                masked_ids[i] = sequence[i]
                sequence[i] = mask_idx
        return sequence, masked_ids

    def pre_input_old(self, data):
        bert_input_data = []
        ns_label_is_true = True

        categories = {}
        for i in range(0, len(data), 1): 
            data_item = data[i]
            if not (virtual := data_item[3]) in categories:
                categories[virtual] = []
            categories[virtual].append(data_item) 

        # first sentence is feature sequence, second one is another feature 
        # sequence of the same or different category 
        for i in range(0, len(data), 1):
            data_item = data[i]
            direction = data_item[1] # gambling direction
            virtual_profit = data_item[2] # if you can see the future
            pred_token = data_item[3]
            candle_idx = data_item[-1]
            next_sentence_label = 0

            if not ns_label_is_true:
                pred = pred_token
                while pred == pred_token:
                    pred = random.choice(self.pred_list)
                pred_token = pred # next is false
                next_sentence_label = 1
            ns_label_is_true = not ns_label_is_true            
            
            first_stc = [self.tokenizer.CLS]
            first_stc.extend(data_item[0])
            first_stc.append(self.tokenizer.SEP)

            while True:
                di = categories[pred_token][
                            random.randrange(0, len(categories[pred_token]))]
                if not di[-1] == data_item[-1]:
                    break

            snd_stc = di[0].tolist()
            snd_stc.append(self.tokenizer.SEP)
            input_tokens = first_stc + snd_stc

            if self.seq_len is None:
                self.seq_len = len(input_tokens)          
            
            padding_len = self.seq_len - len(input_tokens)
            attention_mask = [1] * len(input_tokens) + [0] * padding_len
            token_type_ids = [0] * len(first_stc) + [1] * len(snd_stc) \
                                                        + [1] * padding_len
 
            input_ids = self.tokenizer.convert_tokens_to_ids(
                input_tokens + [self.tokenizer.PAD] * padding_len, unk=False) 
            input_ids,  labels = self._mask(input_ids)

            input = {
                'input': {
                    'input_ids': torch.LongTensor(input_ids),
                    'attention_mask': torch.LongTensor(attention_mask),
                    'token_type_ids': torch.LongTensor(token_type_ids),
                    'labels': torch.LongTensor(labels),
                    'next_sentence_label': torch.LongTensor(
                                                    [next_sentence_label]),   
                },
                'admin': { 
                    'candle_index': torch.LongTensor([candle_idx]),
                    'pred_token': torch.LongTensor([
                            self.tokenizer.convert_tokens_to_ids(pred_token)]),
                    'virtual': torch.FloatTensor([virtual_profit]),
                    'direction': torch.LongTensor([direction])
                }
            }
            bert_input_data.append(input)

        return bert_input_data

    def pre_input(self, data):
        bert_input_data = []
        categories = {}
        for i in range(0, len(data), 1): 
            data_item = data[i]
            if not (virtual := data_item[3]) in categories:
                categories[virtual] = []
            categories[virtual].append(data_item) 

        # first sentence is feature sequence, second one is another feature 
        # sequence of the same or different category 
        for i in range(0, len(data), 1):
            data_item = data[i]
            direction = data_item[1] # gambling direction
            virtual_profit = data_item[2] # if you can see the future
            candle_idx = data_item[-1]

            # 0 indicates sequence snd is a continuation of sequence fst:
            next_sentence_label = i % 2
            pred_token = data_item[3]
            if next_sentence_label: # next_sentence_label == 1
                pred = pred_token # sentence discontinuation
                while pred == pred_token: 
                    pred = random.choice(self.pred_list)
                pred_token = pred # next is false
                next_sentence_label = 1 

            first_stc = [self.tokenizer.CLS]
            first_stc.extend(data_item[0])
            first_stc.append(self.tokenizer.SEP)

            snd_stc_case = (i % 4) < 2 
            if snd_stc_case:
                while True:
                    di = categories[pred_token][
                                random.randrange(0, len(categories[pred_token]))]
                    if not di[-1] == data_item[-1]:
                        break
                snd_stc = di[0].tolist()
            else:
                snd_stc = [pred_token]

            snd_stc.append(self.tokenizer.SEP)
            input_tokens = first_stc + snd_stc

            if self.seq_len is None:
                self.seq_len = len(input_tokens)          
            
            padding_len = self.seq_len - len(input_tokens)
            attention_mask = [1] * len(input_tokens) + [0] * padding_len
            token_type_ids = [0] * len(first_stc) + [1] * len(snd_stc) \
                                                        + [1] * padding_len
 
            input_ids = self.tokenizer.convert_tokens_to_ids(
                input_tokens + [self.tokenizer.PAD] * padding_len, unk=False) 
            input_ids,  labels = self._mask(input_ids)

            input = {
                'input': {
                    'input_ids': torch.LongTensor(input_ids),
                    'attention_mask': torch.LongTensor(attention_mask),
                    'token_type_ids': torch.LongTensor(token_type_ids),
                    'labels': torch.LongTensor(labels),
                    'next_sentence_label': torch.LongTensor(
                                                    [next_sentence_label]),   
                },
                'admin': { 
                    'candle_index': torch.LongTensor([candle_idx]),
                    'pred_token': torch.LongTensor([
                            self.tokenizer.convert_tokens_to_ids(pred_token)]),
                    'virtual': torch.FloatTensor([virtual_profit]),
                    'direction': torch.LongTensor([direction])
                }
            }
            bert_input_data.append(input)

        return bert_input_data

    def bert_input_item(self, data, prediction):
        segments = data[0][0], data[1][0]
        candle_idx = data[0][-1], data[1][-1]
        predictions = prediction[0], prediction[1]
        
        def next_sentence_label(predictions): 
            label = predictions[0] == predictions[1]
            if random.randrange(0, 2):
                label = not label
            return torch.LongTensor(int(label))
        
        tokens = [self.tokenizer.CLS]
        tokens.extend(segments[0])
        tokens.append(self.tokenizer.SEP)

        first_stc = copy.deepcopy(tokens)
        first_stc_len = len(tokens)

        tokens.extend(segments[1])
        tokens.append(self.tokenizer.SEP)

        snd_stc_len = len(tokens) - first_stc_len

        if self.seq_len is None:
            self.seq_len = len(tokens)
        attention_mask = [1] * len(tokens)
        # attention_mask[first_stc_len - 1] = 1
        # attention_mask[first_stc_len + snd_stc_len - 1] = 1
        
        token_type_ids = [0] * first_stc_len + [1] * snd_stc_len
        token_ids = self.tokenizer.convert_tokens_to_ids(
                                                tokens, unk=False) 
        token_ids,  masked_ids = self._mask(token_ids)

        token_ids = torch.LongTensor(token_ids)
        attention_mask = torch.LongTensor(attention_mask)
        token_type_ids = torch.LongTensor(token_type_ids)
        masked_ids = torch.LongTensor(masked_ids)
        style_tren = torch.LongTensor(self.STYLES.index(self.PRE_TREN))
        class_labels = torch.LongTensor(
                                    [self.vp_tokenizer.label(prediction[0])])
        
        retval = None
        # BertForPreTraining:
        if self.PRE_TREN in self.bert_dataset: 
            retval = {
                'input': {
                    'input_ids': token_ids,
                    'attention_mask': attention_mask,
                    'token_type_ids': token_type_ids,
                    'labels': masked_ids,
                    'next_sentence_label': next_sentence_label(predictions),   
                },
                'target': {'target': masked_ids}, 
                'admin': {
                    'style': style_tren,
                }
            }
        # BertForMaskedLM:
        elif self.MASKED in self.bert_dataset: 
            retval = {
                'input': {
                    'input_ids': token_ids,
                    'attention_mask': attention_mask,
                    'token_type_ids': token_type_ids,
                    'labels': masked_ids,
                },
                'target': {'target': masked_ids}, 
                'admin': {
                    'style': style_tren,
                }
            }
        # BertForSequenceClassification:            
        elif self.CLASS_LABELS in self.bert_dataset: 
            input_ids = torch.LongTensor(self.tokenizer.convert_tokens_to_ids(
                first_stc, unk=False))

            retval = {
                'input': {
                    'input_ids': input_ids,
                    'attention_mask': torch.LongTensor(
                                    [1] * first_stc_len), 
                    'token_type_ids': torch.LongTensor(
                                    [0] * (first_stc_len)),
                    'labels': class_labels,
                },
                'target': {'target': class_labels},
                'admin': {
                    'style': style_tren,
                }
            }
        
        if retval is None:
            retval = {
                'input': {
                    'input_ids': token_ids,
                    'attention_mask': attention_mask,
                    'token_type_ids': token_type_ids,
                },
                'target': {'target': predictions},
                'admin': {
                    'style': torch.LongTensor(self.STYLES.index(self.NONE)),
                }
            }

        retval['admin']['prediction'] = class_labels
        retval['admin']['timestamp'] = [int(candle_idx[0]), int(candle_idx[1])]

        return retval
    
    def __len__(self):
        return len(self.train_data) + len(self.val_data)  + len(self.test_data)

class Dataset(Dataset):
    def __init__(
                self,
                data,
                data_object, 
    ):
        super().__init__()
        self.data = data
        self.data_object = data_object
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        # print(self.data[index])
        return self.data[index]

def test():
    from szx82.models.bert.pre.model import MODEL
    from torch.utils.data import DataLoader

    tokenizer_file = r'tokenizer_piecewise_short;mean_len-15;seg_size-10;idx_step-1;level-4;'
    prediction_file = r'forecast;profit_min-8.0;panic_threshold-2.0;data_window-30;' 

    data_object = Data(
        tokenizer_file, 
        prediction_file,
        ModelClass=MODEL,
        data_slice=(0, 100000),
        data_split={'train_val': 0.75, 'val_test': 0.25},
        )
    
    dataset = data_object.train_dataset
    train_dataloader=DataLoader(
                dataset,
                batch_size=1, 
                shuffle=False, 
                pin_memory=False,
                drop_last=True,
                                )
    
    for idx, (batch) in enumerate(train_dataloader):
        print(f'''
batch['input']['input_ids']:
{batch['input']['input_ids']}
{dataset.data_object.tokenizer.convert_ids_to_tokens(batch['input']['input_ids'][0].tolist())}
batch['input']['token_type_ids']:
{batch['input']['token_type_ids']}
batch[0]['attention_mask']:
{batch['input']['attention_mask']}
batch[0]['token_type_ids']:
{batch['input']['token_type_ids']}
{batch['input']['labels']}
{batch['input']['next_sentence_label']}
''')
        break

def main():
    test()
# python -m szx82.models.bert_rv.dataset
if __name__ == '__main__':
    main()