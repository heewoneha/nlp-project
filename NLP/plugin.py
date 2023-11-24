import os
import torch
import shutil
import json


def delete_folder(folder_path):
    """
    Best 모델을 선정한 뒤 폴더 및 파일 제거
    """
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    else:
        print('not exists: ', folder_path)


def check_exists_cuda(device):
    """
    cuda 정보를 출력하는 함수
    """
    print(f'========{device}========')
    print('Device:', torch.cuda.device)
    print('Count of using GPUs:', torch.cuda.device_count())   
    print('Current cuda device:', torch.cuda.current_device())
    print('====================')


def load_jsonl(file_name):
    """
    JSONL 파일을 읽어들이는 함수
    """
    json_list = []

    with open(file_name, encoding="utf-8") as f:
        for l in f.readlines():
            json_list.append(json.loads(l))
    
    return json_list
