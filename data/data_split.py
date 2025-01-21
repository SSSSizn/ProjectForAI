import os
import pandas as pd
import shutil
from sklearn.utils import shuffle
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from torchvision import transforms
from PIL import Image
import torch

# 划分训练集和验证集，为了保证实验可复现，设置统一的随机种子
def train_val_split(data, val_size=0.2, random_state=42):
    data = shuffle(data, random_state=random_state)
    split_index = int(len(data) * (1 - val_size))
    train = data[:split_index].reset_index(drop=True)
    val = data[split_index:].reset_index(drop=True)
    return train, val

# 复制图像和文本数据到新文件夹中，src_dir为源目录，dest_dir为新目录
def copy_files(ids, img_src_dir, txt_src_dir, img_dest_dir, txt_dest_dir):
    
    os.makedirs(img_dest_dir, exist_ok=True)
    os.makedirs(txt_dest_dir, exist_ok=True)
    
    for data_id in ids:
        img_path = os.path.join(img_src_dir, f'{data_id}.jpg')
        img_dest = os.path.join(img_dest_dir, f'{data_id}.jpg')
        text_path = os.path.join(txt_src_dir, f'{data_id}.txt')
        text_dest = os.path.join(txt_dest_dir, f'{data_id}.txt')
        
        if os.path.exists(img_path) and os.path.exists(text_path):
            shutil.copy(img_path, img_dest)
            shutil.copy(text_path, text_dest)
        else:
            print(f"警告: {img_path} 或 {text_path} 不存在，跳过复制.")


class MultimodalDataset(Dataset):
    # 初始化数据集，paths为文件路径，labels为标签，tokenizer用于文本编码的预训练，img_transform用于图像的转换操作
    def __init__(self, text_paths, img_paths, labels, tokenizer, img_transform):
        self.text_paths = text_paths
        self.img_paths = img_paths
        self.labels = labels
        self.tokenizer = tokenizer
        self.img_transform = img_transform
        self.label_map = {
            "positive": 0,
            "negative": 1,
            "neutral": 2
        }
    
    # 返回数据集的大小
    def __len__(self):
        return len(self.text_paths)
    
    # 获取索引为idx的数据样本，并进行数据处理，包括维度和格式的修改
    def __getitem__(self, idx):
        vocab_size = 30522  
        with open(self.text_paths[idx], 'r', encoding='utf-8', errors='surrogateescape') as file:
            text = file.read()

        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=512, return_tensors="pt")
        input_ids = encoding['input_ids'].squeeze(0)  

        max_index = input_ids.max().item()
        if max_index >= vocab_size:
            print(f"Warning: Max input index {max_index} exceeds vocab size {vocab_size}.")

        attention_mask = encoding['attention_mask'].squeeze(0)

        img = Image.open(self.img_paths[idx]).convert('RGB')
        img = self.img_transform(img)  
        
        label = self.labels[idx]
        label = self.label_map.get(label, -1)  
        label = torch.tensor(label)  

        return input_ids, attention_mask, img, label

# 准备数据集，对训练集、验证集、测试集进行划分和复制
def prepare_data():
    
    test_labels = pd.read_csv('./data/test_without_label.txt', encoding='utf-8')
    test_ids = test_labels['guid'].copy().values
    copy_files(test_ids, './data/original_data/', './data/original_data/', './data/test/img/', './data/test/text/')

    train_labels = pd.read_csv('./data/train.txt', encoding='utf-8')
    train_data, val_data = train_val_split(train_labels)

    train_data.sort_values(by="guid", inplace=True, ascending=True)
    val_data.sort_values(by="guid", inplace=True, ascending=True)
    train_data.to_csv('./data/train/train_data.csv', index=False)
    val_data.to_csv('./data/val/val_data.csv', index=False)

    train_ids = train_data['guid'].copy().values
    copy_files(train_ids, './data/original_data/', './data/original_data/', './data/train/img/', './data/train/text/')

    val_ids = val_data['guid'].copy().values
    copy_files(val_ids, './data/original_data/', './data/original_data/', './data/val/img/', './data/val/text/')

    test_ids = test_labels['guid'].copy().values
    copy_files(test_ids, './data/original_data/', './data/original_data/', './data/test/img/', './data/test/text/')

    train_text_paths = [os.path.join('./data/train/text', f'{train_id}.txt') for train_id in train_ids]
    train_img_paths = [os.path.join('./data/train/img', f'{train_id}.jpg') for train_id in train_ids]
    val_text_paths = [os.path.join('./data/val/text', f'{val_id}.txt') for val_id in val_ids]
    val_img_paths = [os.path.join('./data/val/img', f'{val_id}.jpg') for val_id in val_ids]
    test_text_paths = [os.path.join('./data/test/text', f'{test_id}.txt') for test_id in test_ids]
    test_img_paths = [os.path.join('./data/test/img', f'{test_id}.jpg') for test_id in test_ids]

    train_labels = train_data['tag'].values  
    val_labels = val_data['tag'].values  
    test_labels = test_labels['tag'].values  
    
    BERT_PATH = 'bert'
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    img_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = MultimodalDataset(train_text_paths, train_img_paths, train_labels, tokenizer, img_transform)
    val_dataset = MultimodalDataset(val_text_paths, val_img_paths, val_labels, tokenizer, img_transform)
    test_dataset = MultimodalDataset(test_text_paths, test_img_paths, test_labels, tokenizer, img_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return train_loader, val_loader, test_loader

