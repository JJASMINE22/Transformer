## Transformer的TF实现
---

1. [所需环境 Environment](#所需环境)
2. [注意事项 Cautions](#注意事项)
3. [训练步骤 How2train](#训练步骤)

1.所需环境
numpy==1.19.5
tensorflow-gpu==2.5.1  
tensorflow-datasets==4.4.0  

2.注意事项
该手写Transformer将mask的提取步骤整合于模型前向传递过程
修改部分多头注意力机制拆分合并规则

3.运行train.py即可开始训练。
将pe葡萄牙语作为输入源，en英语作为输出源 

