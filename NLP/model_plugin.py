from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import EvalPrediction
from sklearn.utils import shuffle
import numpy as np
import torch
import random
import os


SEED_VALUE = 42

sentiment_id_to_str = ['1', '-1', '0']  # pos: 0, neg: 1, neu: 2로 변환
sentiment_str_to_id = {sentiment_id_to_str[i]: i for i in range(len(sentiment_id_to_str))}


class klue_Dataset(torch.utils.data.Dataset):
    """
    Input: 정규표현식, 개수가 적은 속성 제거 등으로 전처리된 데이터셋
    Ouput: 1차원 텐서(__getitem__) / 샘플의 수(__len__)
    """
    def __init__(self, dataset, label):
        self.dataset = dataset # {'input_ids': ~, 'token_type_ids': ~, 'attention_mask': ~, 'entity_ids' : ~}
        self.label = label

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.dataset.items()}
        item['label'] = torch.tensor(self.label[idx])
        return item

    def __len__(self):
        return len(self.label)


def set_seed(seed):
    """
    seed value를 고정하는 함수
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def extract_annotation_keys(jsonl_data):
    """
    JSONL 데이터에서 annotation 키 목록을 추출하는 함수
    """
    annotation_keys = []

    for json_data in jsonl_data:
        if "annotation" in json_data:
            annotation_keys.extend([item[0] for item in json_data["annotation"]])
    
    unique_annotation_keys = list(set(annotation_keys))

    return unique_annotation_keys


def define_datasets(data, main_category, category_with_original_aspects): # train, validation, test별로 들어옴 !
    """
    속성 & 감성 관련 데이터 및 레이블을 정의하는 함수
    """
    ASP_datas = [[] for i in range(len(category_with_original_aspects))]
    ASP_labels = [[] for i in range(len(category_with_original_aspects))]

    SEN_data = []
    SEN_labels = []

    for i, pair in enumerate(category_with_original_aspects):
        for datas in data:
            review = datas['raw_text']
            annotations = datas['annotation']
            check_point = False
            
            ASP_datas[i].append(review)
            
            for annotation in annotations:
                entity_property = f'{main_category}#' + annotation[0]
                sentiment = annotation[1]

                if entity_property == pair:
                    check_point = True
                    
            if check_point:
                ASP_labels[i].append(1)
                SEN_data.append(review + " " + pair)
                SEN_labels.append(sentiment_str_to_id[sentiment])
            
            else:
                ASP_labels[i].append(0)
                
        ASP_datas[i], ASP_labels[i] = shuffle(ASP_datas[i], ASP_labels[i], random_state = SEED_VALUE)
        
    SEN_data, SEN_labels = shuffle(SEN_data, SEN_labels, random_state = SEED_VALUE)
    
    return ASP_datas, ASP_labels, SEN_data, SEN_labels


def reshape_to_1d(val, Datas, labels, tokenizer): # train, validation, test별로 들어옴 !
    """
    Class를 이용해 1차원 텐서로 변경하는 함수
    """
    if val == 'aspect':
        klue_sets = []

        for i in range(len(Datas)):
            tok_sentence = tokenizer(Datas[i], return_tensors="pt", padding='max_length' \
                            , truncation=True, max_length=256, add_special_tokens=True)  
            
            klue_sets.append(klue_Dataset(tok_sentence, labels[i]))
        
        return klue_sets
    
    elif val == 'sentiment':
        sen_tok_sentence = tokenizer(Datas, return_tensors="pt", padding='max_length' \
                            , truncation=True,max_length=256, add_special_tokens=True)  
        
        SEN_klue_sets = klue_Dataset(sen_tok_sentence, labels)

        return SEN_klue_sets
    

def compute_metrics(val, p: EvalPrediction):
    """
    Input:
      val이 aspect라면    average = 'binary',  (이진 분류)
      val이 sentiment라면 average = 'weighted' (다중클래스 분류)
    Output:
      평가지표 점수
    """
    if val == 'aspect':
        average = 'binary'
    elif val == 'sentiment':
        average = 'weighted'
    
    labels = p.label_ids
    preds = p.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average=average)
    acc = accuracy_score(labels, preds)

    return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}


def show_test_evaluation(val, infers, infer_labels, category_with_original_aspects=None):
    """
    속성/감성 모델의 테스트 평가지표 결과를 출력하는 함수
    """
    if val == 'aspect':
        length = len(category_with_original_aspects)
        average = 'binary'
    elif val == 'sentiment':
        length = 1
        average = 'weighted'
    
    for x in range(0, length):
        print(x, "th Test.....")
        labelss = []
        for i in infer_labels[x]:
            for j in i:
                labelss.append(j)

        precision, recall, f1, _ = precision_recall_fscore_support(labelss, infers[x], average=average)
        acc = accuracy_score(labelss, infers[x])

        print("Accuracy: ", acc)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1-score:", f1)
