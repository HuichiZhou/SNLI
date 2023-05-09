# 以下是针对SNLI数据集创建的ESIM模型
[参考文献] Enhanced LSTM for Natural Language Inference (https://arxiv.org/pdf/1609.06038v3.pdf)

本框架主要使用了BiLSTM或TreeLSTM循环神经网络模型对文本对进行建模，  
<div align=center>
  <img src="./pic/esim.png"> 
  <div class="caption">ESIM框架的示意图。</div>
</div>   


```python
torch==1.5.1
```

  
## 在命令行终端执行文件
```python
python fetch_data.py
python preprocess_snli.py
python train_snli.py
```
