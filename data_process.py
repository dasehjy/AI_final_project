import torch
from torch.utils.data import DataLoader, random_split
import pandas as pd
import os
from PIL import Image
from transformers import AutoFeatureExtractor
from transformers import BertTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

tokenizer = BertTokenizer.from_pretrained('./pretrained_model/bert-base-uncased')
feature_extractor = AutoFeatureExtractor.from_pretrained("./pretrained_model/resnet-50")

# 处理训练集数据
def get_train_data_list(data_folder_path = './data'):
    train_data_list = []
    train_label_path = './train.txt'
    
    # 简化标签
    train_label = pd.read_csv(train_label_path)
    tag_mapping = {
        'negative': 0,
        'neutral': 1,
        'positive': 2,
    }

    # 结合图片、文本
    for idx, (guid, tag) in enumerate(train_label.values):
        data_dict = {}
        data_dict['guid'] = int(guid)
        data_dict['tag'] = tag_mapping[tag]
        file_path_image = os.path.join(data_folder_path, f'{int(guid)}.jpg')
        file_path_text = os.path.join(data_folder_path, f'{int(guid)}.txt')
        image = Image.open(file_path_image)
        text = None
        with open(file_path_text, 'rb') as f:
            text = f.readline().strip()
        data_dict['image'], data_dict['text'] = image, text
        train_data_list.append(data_dict)
        
    return train_data_list

# 处理测试集数据
def get_test_data_list():
    test_data_list = []
    test_label_path = 'test_without_label.txt'
    data_folder_path = './data'
    
    test_label = pd.read_csv(test_label_path)

    for guid, tag in test_label.values:
        data_dict = {}
        data_dict['guid'] = int(guid)
        data_dict['tag'] = None
        file_path_image = os.path.join(data_folder_path, f'{int(guid)}.jpg')
        file_path_text = os.path.join(data_folder_path, f'{int(guid)}.txt')
        image = Image.open(file_path_image)
        text = None
        with open(file_path_text, 'rb') as f:
            text = f.readline().strip()
        data_dict['image'], data_dict['text'] = image, text
        test_data_list.append(data_dict)

    return test_data_list

# 解码
def text_decode(text: bytes):
    try:
        decode = text.decode(encoding='utf-8')
    except:
        try: 
            decode = text.decode(encoding='GBK')
        except:
            decode = text.decode(encoding='gb18030')
    return decode

# 对整个数据列表进行解码
def data_preprocess(data_list):
    for data in data_list:
        data['text'] = text_decode(data['text'])

    return data_list
# 返回guid, 标签，处理后的图片，解码后的文本
def collate_fn(data_list):
    guid = [data['guid'] for data in data_list]
    tag = [data['tag'] for data in data_list]
    image = [data['image'] for data in data_list]
    image = feature_extractor(image, return_tensors="pt")
    text = [data['text'] for data in data_list]
    text = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=100)

    return guid, None if tag[0] == None else torch.LongTensor(tag), image, text

# 划分数据集并创建训练集、验证集的DataLoader
def get_train_data_loader() -> (DataLoader, DataLoader):
    train_data_list= get_train_data_list()
    train_data_list = data_preprocess(train_data_list)

    train_data_length = int(len(train_data_list) * 0.75)
    valid_data_length = len(train_data_list) - train_data_length
    train_dataset, valid_dataset = random_split(dataset=train_data_list, lengths = [train_data_length, valid_data_length])

    train_data_loader = DataLoader(
        dataset=train_dataset,
        collate_fn=collate_fn,
        batch_size=16,
        shuffle=True,
        drop_last=False,
    )

    valid_data_loader = DataLoader(
        dataset=valid_dataset,
        collate_fn=collate_fn,
        batch_size=16,
        shuffle=True,
        drop_last=False,
    )

    return train_data_loader, valid_data_loader

# 创建测试集DataLoader
def get_test_data_loader() -> (DataLoader):
    test_data_list = get_test_data_list()
    test_data_list = data_preprocess(test_data_list)

    test_data_loader = DataLoader(
        dataset=test_data_list,
        collate_fn=collate_fn,
        batch_size=16,
        shuffle=False,
        drop_last=False,
    )

    return test_data_loader
    
# 模型评测标准
def calc_metrics(target, pred):
    accuracy = accuracy_score(target, pred)
    precision_w = precision_score(target, pred, average='weighted')
    recall_w = recall_score(target, pred, average='weighted')
    f1_w = f1_score(target, pred, average='weighted')
    precision = precision_score(target, pred, average='macro')
    recall = recall_score(target, pred, average='macro')
    f1 = f1_score(target, pred, average='macro')
    return accuracy, precision_w, recall_w, f1_w, precision, recall, f1