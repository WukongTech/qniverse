# 基于深度学习的时序方法用于金融数据预测


### 数据

A股数据：[百度网盘](https://pan.baidu.com/s/1vJivEwGEmHPJ_xbSodLSDQ) 提取码: hu8b
将数据放入data文件夹下，然后运行gendata.ipynb生成qlib格式的数据

### 训练
```
python train.py --config_file configs/config_wftnet.yaml
```

修改各自的配置文件即可
