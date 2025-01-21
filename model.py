import torch
import torch.nn as nn
from torch.optim import Adam
import sys
import os
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import classification_report
sys.path.append(os.path.join(os.getcwd(), 'data'))

from data_split import prepare_data

# 定义BERT模型用于处理文本信息
class BERTModel(nn.Module):
    def __init__(self, embed_size, hidden_size, num_layers, num_heads, vocab_size=30522, dropout=0.1, num_classes=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = nn.Parameter(torch.randn(1, 512, embed_size))  
        self.encoder_layers = nn.TransformerEncoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            dim_feedforward=hidden_size,
            dropout=dropout,
            batch_first=True  
        )
        
        self.encoder = nn.TransformerEncoder(
            self.encoder_layers,
            num_layers=num_layers
        )
        self.fc = nn.Linear(embed_size, num_classes)

    def forward(self, x, attention_mask):
        emb = self.embedding(x) + self.positional_encoding[:, :x.size(1), :]
        attention_mask = attention_mask.bool()
        output = self.encoder(emb, src_key_padding_mask=attention_mask)
        cls_token_output = output[:, 0, :]  
        logits = self.fc(cls_token_output)  
        return logits

# 定义AlexNet模型用于处理图像信息
class AlexNetModel(nn.Module):
    def __init__(self, num_classes=3):
        super(AlexNetModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# 多模态模型
class MultiModalModel(nn.Module):
    def __init__(self, text_model, img_model, fusion_dim = 6, num_classes = 3):
        super(MultiModalModel, self).__init__()
        self.text_model = text_model
        self.img_model = img_model
        self.fc_fusion = nn.Linear(fusion_dim, num_classes)

    def forward(self, img, text, attention_mask):
        img_features = self.img_model(img)  
        img_features = img_features.view(img_features.size(0), -1)  
        
        text_features = self.text_model(text, attention_mask)
        
        combined_features = torch.cat([img_features, text_features], dim=-1)
        
        output = self.fc_fusion(combined_features)
        return output

#消融实验：仅使用图像信息
class ImageModel(nn.Module):
    def __init__(self, text_model, img_model, fusion_dim=3, num_classes=3):
        super(ImageModel, self).__init__()
        self.text_model = text_model
        self.img_model = img_model
        self.fc_fusion = nn.Linear(fusion_dim, num_classes)  

    def forward(self, img, text, attention_mask):
        img_features = self.img_model(img)  
        img_features = img_features.view(img_features.size(0), -1)  
        
        output = self.fc_fusion(img_features)
        return output

#消融实验：仅使用文本信息
class TextModel(nn.Module):
    def __init__(self, text_model, img_model, fusion_dim=3, num_classes=3):
        super(TextModel, self).__init__()
        self.text_model = text_model
        self.img_model = img_model
        self.fc_fusion = nn.Linear(fusion_dim, num_classes)
        
    def forward(self, img, text, attention_mask):
        text_features = self.text_model(text, attention_mask)

        output = self.fc_fusion(text_features)
        return output

# 准备数据集
train_loader, val_loader, test_loader = prepare_data()

# 设置训练参数
args = {
    'model': 'agg',            # 模型类型
    'lr': 1e-5,                # 学习率
    'batch_size': 64,          # 批量大小
    'epochs': 20,              # 训练轮次
    'embed_size': 256,         # 嵌入维度
    'hidden_size': 64,         # 隐藏层维度
    'num_layers': 2,           # Transformer 层数
    'num_heads': 4,            # Attention heads 数量
}

# 检查是否有可用的 GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")  # 打印当前使用的 GPU 名称
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.")

# 分别初始化BERT模型和ALEXNET模型
text_model = BERTModel(vocab_size=30522, embed_size=args['embed_size'], hidden_size=args['hidden_size'], num_layers=args['num_layers'], num_heads=args['num_heads'])
print("Text model initialized successfully.")

img_model = AlexNetModel(num_classes=3)
print("Image model initialized successfully.")

# 多模态模型初始化
model = MultiModalModel(text_model=text_model, img_model=img_model).to(device)
only_text_model = TextModel(text_model=text_model, img_model=img_model).to(device)
only_image_model = ImageModel(text_model=text_model, img_model=img_model).to(device)
print("Models initialized successfully.")

# 加权损失函数
class_counts = [1910, 954, 336]  # 训练集中的每个类别的样本数量
total_samples = sum(class_counts) 
class_weights = [total_samples / count for count in class_counts]
class_weights = torch.tensor(class_weights).float().to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = Adam(model.parameters(), lr=args['lr'])

# 多模态模型
best_val_f1 = 0.0  
best_train_f1 = 0.0 

for epoch in range(args['epochs']):
    print(f"Starting Epoch {epoch+1}/{args['epochs']}...")
    
    # 训练
    model.train()
    running_loss = 0.0
    all_train_labels = []
    all_train_preds = []

    for batch_idx, batch in enumerate(train_loader):
        text_inputs, attention_masks, img_inputs, labels = batch
        
        text_inputs = text_inputs.to(device)
        attention_masks = attention_masks.to(device)
        img_inputs = img_inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        
        outputs = model(img_inputs, text_inputs, attention_masks)  
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)

        all_train_labels.extend(labels.cpu().numpy())
        all_train_preds.extend(predicted.cpu().numpy())

        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{args['epochs']}, Batch {batch_idx+1}/{len(train_loader)}: "
                  f"Loss: {loss.item():.4f}")

    print(f"Epoch {epoch+1}/{args['epochs']} - Average Training Loss: {running_loss / len(train_loader):.4f}")
    train_f1 = f1_score(all_train_labels, all_train_preds, average='weighted')  # 计算加权的 F1-score
    train_precision = precision_score(all_train_labels, all_train_preds, average='weighted')
    train_recall = recall_score(all_train_labels, all_train_preds, average='weighted')
    print(f"Epoch {epoch+1}/{args['epochs']} - Training F1-score: {train_f1:.4f}")
    print(f"Epoch {epoch+1}/{args['epochs']} - Training Precision: {train_precision:.4f}")
    print(f"Epoch {epoch+1}/{args['epochs']} - Training Recall: {train_recall:.4f}")
    
    # 验证
    only_text_model.eval()
    all_val_labels = []
    all_val_preds = []
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            text_inputs, attention_masks, img_inputs, labels = batch
            text_inputs = text_inputs.to(device)
            attention_masks = attention_masks.to(device)
            img_inputs = img_inputs.to(device)
            labels = labels.to(device)

            outputs = model(img_inputs, text_inputs, attention_masks) # 使用仅文本模型

            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            all_val_labels.extend(labels.cpu().numpy())
            all_val_preds.extend(predicted.cpu().numpy())

        print(f"Epoch {epoch+1}/{args['epochs']} - Validation Loss: {val_loss / len(val_loader):.4f}")
        val_f1 = f1_score(all_val_labels, all_val_preds, average='weighted')  # 计算加权的 F1-score
        val_precision = precision_score(all_val_labels, all_val_preds, average='weighted')
        val_recall = recall_score(all_val_labels, all_val_preds, average='weighted')
        print(f"Epoch {epoch+1}/{args['epochs']} - Validation F1-score: {val_f1:.4f}")
        print(f"Epoch {epoch+1}/{args['epochs']} - Validation Precision: {val_precision:.4f}")
        print(f"Epoch {epoch+1}/{args['epochs']} - Validation Recall: {val_recall:.4f}")
    
    # 储存最佳模型
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), 'val_best_model.pth')
        print(f"Saved best model with validation F1-score: {best_val_f1:.4f}")

# 加载最佳模型进行测试
model.load_state_dict(torch.load('val_best_model.pth'))
model.eval()

with torch.no_grad():
    predictions = []
    for batch in test_loader:
        text_inputs, attention_masks, img_inputs, _ = batch
        text_inputs = text_inputs.to(device)
        attention_masks = attention_masks.to(device)
        img_inputs = img_inputs.to(device)

        outputs = model(img_inputs, text_inputs, attention_masks)

        _, predicted = torch.max(outputs, 1)
        predictions.extend(predicted.tolist())
        
with open('val_predictions.txt', 'w') as f:
    for pred in predictions:
        f.write(str(pred) + '\n')

# 消融实验：只使用文本信息

best_val_f1 = 0.0  
best_train_f1 = 0.0  

for epoch in range(args['epochs']):
    print(f"Starting Epoch {epoch+1}/{args['epochs']}...")
    
    # 训练
    only_text_model.train()
    running_loss = 0.0
    all_train_labels = []
    all_train_preds = []

    for batch_idx, batch in enumerate(train_loader):
        text_inputs, attention_masks, img_inputs, labels = batch
        
        text_inputs = text_inputs.to(device)
        attention_masks = attention_masks.to(device)
        img_inputs = img_inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        
        outputs = only_text_model(img_inputs, text_inputs, attention_masks)  
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)

        all_train_labels.extend(labels.cpu().numpy())
        all_train_preds.extend(predicted.cpu().numpy())

        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{args['epochs']}, Batch {batch_idx+1}/{len(train_loader)}: "
                  f"Loss: {loss.item():.4f}")

    print(f"Epoch {epoch+1}/{args['epochs']} - Average Training Loss: {running_loss / len(train_loader):.4f}")
    train_f1 = f1_score(all_train_labels, all_train_preds, average='weighted')  
    train_precision = precision_score(all_train_labels, all_train_preds, average='weighted')
    train_recall = recall_score(all_train_labels, all_train_preds, average='weighted')
    print(f"Epoch {epoch+1}/{args['epochs']} - Training F1-score: {train_f1:.4f}")
    print(f"Epoch {epoch+1}/{args['epochs']} - Training Precision: {train_precision:.4f}")
    print(f"Epoch {epoch+1}/{args['epochs']} - Training Recall: {train_recall:.4f}")

    # 验证
    only_text_model.eval()
    all_val_labels = []
    all_val_preds = []
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            text_inputs, attention_masks, img_inputs, labels = batch
            text_inputs = text_inputs.to(device)
            attention_masks = attention_masks.to(device)
            img_inputs = img_inputs.to(device)
            labels = labels.to(device)

            outputs = only_text_model(img_inputs, text_inputs, attention_masks)  

            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            all_val_labels.extend(labels.cpu().numpy())
            all_val_preds.extend(predicted.cpu().numpy())

        print(f"Epoch {epoch+1}/{args['epochs']} - Validation Loss: {val_loss / len(val_loader):.4f}")
        val_f1 = f1_score(all_val_labels, all_val_preds, average='weighted') 
        val_precision = precision_score(all_val_labels, all_val_preds, average='weighted')
        val_recall = recall_score(all_val_labels, all_val_preds, average='weighted')
        print(f"Epoch {epoch+1}/{args['epochs']} - Validation F1-score: {val_f1:.4f}")
        print(f"Epoch {epoch+1}/{args['epochs']} - Validation Precision: {val_precision:.4f}")
        print(f"Epoch {epoch+1}/{args['epochs']} - Validation Recall: {val_recall:.4f}")

# 消融实验：只使用图像信息

best_val_f1 = 0.0  
best_train_f1 = 0.0  

for epoch in range(args['epochs']):
    print(f"Starting Epoch {epoch+1}/{args['epochs']}...")
    
    # 训练
    only_image_model.train()
    running_loss = 0.0
    all_train_labels = []
    all_train_preds = []

    for batch_idx, batch in enumerate(train_loader):
        text_inputs, attention_masks, img_inputs, labels = batch
        
        text_inputs = text_inputs.to(device)
        attention_masks = attention_masks.to(device)
        img_inputs = img_inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        
        outputs = only_image_model(img_inputs, text_inputs, attention_masks)
    
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)

        all_train_labels.extend(labels.cpu().numpy())
        all_train_preds.extend(predicted.cpu().numpy())

        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{args['epochs']}, Batch {batch_idx+1}/{len(train_loader)}: "
                  f"Loss: {loss.item():.4f}")

        print(f"Epoch {epoch+1}/{args['epochs']} - Validation Loss: {val_loss / len(val_loader):.4f}")
        val_f1 = f1_score(all_val_labels, all_val_preds, average='weighted')  
        val_precision = precision_score(all_val_labels, all_val_preds, average='weighted')
        val_recall = recall_score(all_val_labels, all_val_preds, average='weighted')
        print(f"Epoch {epoch+1}/{args['epochs']} - Validation F1-score: {val_f1:.4f}")
        print(f"Epoch {epoch+1}/{args['epochs']} - Validation Precision: {val_precision:.4f}")
        print(f"Epoch {epoch+1}/{args['epochs']} - Validation Recall: {val_recall:.4f}")
    
    # 验证
    only_image_model.eval()
    all_val_labels = []
    all_val_preds = []
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            text_inputs, attention_masks, img_inputs, labels = batch
            text_inputs = text_inputs.to(device)
            attention_masks = attention_masks.to(device)
            img_inputs = img_inputs.to(device)
            labels = labels.to(device)

            outputs = only_image_model(img_inputs, text_inputs, attention_masks)

            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            all_val_labels.extend(labels.cpu().numpy())
            all_val_preds.extend(predicted.cpu().numpy())

        print(f"Epoch {epoch+1}/{args['epochs']} - Validation Loss: {val_loss / len(val_loader):.4f}")
        val_f1 = f1_score(all_val_labels, all_val_preds, average='weighted')  
        val_precision = precision_score(all_val_labels, all_val_preds, average='weighted')
        val_recall = recall_score(all_val_labels, all_val_preds, average='weighted')
        print(f"Epoch {epoch+1}/{args['epochs']} - Validation F1-score: {val_f1:.4f}")
        print(f"Epoch {epoch+1}/{args['epochs']} - Validation Precision: {val_precision:.4f}")
        print(f"Epoch {epoch+1}/{args['epochs']} - Validation Recall: {val_recall:.4f}")