## Transformer的tensorflow2实现
---

## 目录
1. [所需环境 Environment](#所需环境)
2. [模型结构 Structure](#模型结构)
3. [注意事项 Cautions](#注意事项)
4. [位置编码 Positional Encoding](#位置编码)
5. [训练步骤 How2train](#训练步骤)
6. [参考资料 Reference](#参考资料)

## 所需环境
1. numpy==1.19.5
2. tensorflow-gpu==2.5.1  
3. tensorflow-datasets==4.4.0  

## 模型结构
Transformer
![image]()

## 注意事项
1. 该tf2版本的Transformer将mask计算过程整合于模型的推理过程
2. 修改MultiHeadAttention中的通道拆分、合并方式
3. 修改padding_mask方法，实测提升效果不明显
4. 使用tensorflow_datasets数据集，无配置文件
5. 更改数据源，可将该翻译模型用于QA问答

## 位置编码
以序列长度max_seq_len=32，嵌入维度embedding_size=512为例，位置编码特征：  
![image]()  
通过颜色深浅能明细反映三角波属性

## 训练步骤
运行pt_en_train.py即可开始训练。  
将pe葡萄牙语作为输入源，en英语作为输出源 

## 参考资料
1. https://arxiv.org/pdf/1706.03762.pdf  
2. https://blog.csdn.net/qq_44766883/article/details/112008655

