import json
import os

def read_json_file(fp):
    with open(fp, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def load_json_files(dir):
    json_data_list = []

    for filename in os.listdir(dir):
        if filename.endswith('.json'):
            fp = os.path.join(dir, filename)
            json_data = read_json_file(fp)
            json_data_list.extend(json_data)
    
    return json_data_list
