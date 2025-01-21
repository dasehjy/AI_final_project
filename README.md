# 实验五：多模态情感分析

给定配对的文本和图像，预测对应的情感标签，使用多模态完成三分类任务：positive, neutral, negative。使用bert预训练模型对文本进行特征抽取，resnet-50对图像进行特征抽取，将两者融合后完成模型。

## 下载方式

可输入以下命令获取项目代码：

```
git clone https://github.com/dasehjy/AI_final_project
```

在目录下创建`pretrained_model`文件夹，下载bert-base-uncased预训练模型：

```
git lfs install
git clone https://hf-mirror.com/google-bert/bert-base-uncased
```

下载resnet-50模型：

```
git clone https://hf-mirror.com/microsoft/resnet-50
```

或者可以直接在https://hf-mirror.com中直接下载预训练模型文件。

## 文件结构

```
|── AI_final_project  #项目根目录
│   ├── data                   #数据集文件夹（需自行导入）
│   ├── data_process.py        #数据预处理的程序
│   ├── test_without_label.txt #无标签的测试集
│   ├── train.txt              #有标签的训练集
│   ├── main.py                #主程序
│   ├── model.py               #存放所有模型的程序
│   ├── output.txt             #输出文件
│   ├── pretrained_model       #预训练模型的存放目录（需自行创建并下载模型）
│   │   ├── bert-base-uncased      #bert预训练模型
│   │   └── resnet-50              #resnet50预训练模型       
│   ├── p5.ipynb               #保存实验过程的notebook文件
│   ├── README.md              #README
|   └──requirements.txt        #项目依赖
```

## Requirements

```sh
ipykernel==6.29.5
ipython==8.29.0
ipywidgets==8.1.5
numpy==2.1.3
pandas==2.2.3
pillow==11.0.0
scikit-learn==1.6.1
torch==2.5.1
transformers==4.48.0
```

## 执行流程

```sh
usage: main.py [--lr LR] [--weight_decay WEIGHT_DECAY] [--epoch EPOCH] [--model MODEL]

optional arguments:
  --lr LR               set the learning rate, default 2e-5
  --weight_decay WEIGHT_DECAY
                        set weight decay, default 1e-4
  --epoch EPOCH         set train epochs, default 10
  --model MODEL         set the type of model, default AttentionCatModel
```

可选参数列表如下：

1. lr：学习率，默认为2e-5
2. weight_decay：权重衰减，默认为1e-4
3. epoch：模型训练迭代轮数，默认10轮
4. model：使用的模型，可选的模型包括：CatModel，AttentionAddModel ，AttentionCatModel，默认为AttentionCatModel

运行完成后结果保存至当前目录下的`result1.txt`

## 参考

预训练模型选择：[Hugging Face – The AI community building the future.](https://huggingface.co/)