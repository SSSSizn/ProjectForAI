# 实验五：多模态情感分析

### 环境配置

在命令行中输入下面的命令即可完成环境配置：

```shell
pip install -r requirements.txt
```



### 代码运行
运行python文件
```sh
python model.py 
```
model.py中包含多模态模型以及融合模型，直接运行可训练三个模型并查看效果，已设置最佳超参数



### 文件内容

```
.
|-- README.md
|-- __pycache__
|-- model.py  # 完整模型以及训练、验证、测试过程代码
|-- data
|   |-- original_data  # 原始数据集
|   |-- data_split.py  # 数据处理，训练集、验证计划分等
|   |-- test  # 测试数据集
|   |-- test_without_label.txt  
|   |-- train  # 训练数据集
|   |-- train.txt 
|   `-- val  # 验证数据集
|-- model.ipynb  # 与model.py内容相同，已包含完整训练结果
|-- test_predictions.txt  # 预测文件
|-- requirements.txt
|-- 实验报告.pdf
|-- val_best_model.pth  # 最佳模型
.
```
