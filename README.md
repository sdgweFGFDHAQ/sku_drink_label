# 数据集的准备
## 1. python脚本  
/home/DI/zhouzx/code/sku_drink_label/down_drink_sku_data.py: 下载数据集，人工清洗后，用于模型的训练和测试
```python
# 下载数据集所需列
columns = ["id", "name", "appcode", "channeltype_new", "category1_new", "state",
           "city", "brand_name", "series_name", "sku_name",
           "sku_code", "drink_label"]
# 当前清洗后的30w数据集列为：
col1 = ['id', 'name', 'storetype'] # storetype即category1_new
col2 = ['plant_clean', 'fruit_vegetable_clean', 'protein_clean', 'flavored_clean', 'tea_clean',
        'carbonated_clean', 'coffee_clean', 'water_clean', 'special_uses_clean']
```
/home/DI/zhouzx/code/sku_drink_label/down_predict_sku_data.py: 下载数据集，渠道源(appcode)为高德、腾讯、百度，用于预测打标  
```python
# 下载数据集所需列
columns=["id", "name", "appcode", "category1_new", "state", "city",
         "predict_category", "drink_label"]
```
## 2. 数据来源  
数据库：standard_db  
关联表：di_store_sku_drink_label、
di_sku、di_store_classify_dedupe、
di_store_dedupe_labeling  
其中di_store_sku_drink_label表需要通过di_store_sku和di_store_brand表维护(人工打标)

# 预训练
```
# 统一输入输出的标准格式
# 训练集输入
columns = ['id', 'name', 'storetype', 'plant_clean', 'fruit_vegetable_clean', 'protein_clean', 'flavored_clean',
           'tea_clean', 'carbonated_clean', 'coffee_clean', 'water_clean', 'special_uses_clean']
# 预测集输入
columns = ['id', 'name', 'storetype']
# 预测集输出
columns = ['id', 'name', 'storetype', '植物饮料', '果蔬汁类及其饮料', '蛋白饮料', '风味饮料', '茶（类）饮料',
           '碳酸饮料', '咖啡（类）饮料', '包装饮用水', '特殊用途饮料']
```
## 1.原型网络
```
文件保存路径前缀：/home/DI/zhouzx/code/sku_drink_label
/prototypical_network/prototypical.py 原型网络训练模型  
/prototypical_network/prototypical_predict.py 对待测数据进行预测
训练集:/datasets/fs_sku_drink_data_train.csv
验证集：/datasets/fs_sku_drink_data_valid.csv
测试集：/datasets/fs_sku_drink_data_test.csv
预测集：/datasets/di_sku_log_drink_labeling_zzx.csv
```

## 2.匹配网络
...
## 3.孪生网络
```
xx
训练集:/datasets/fs_sku_drink_data_train.csv
验证集：/datasets/fs_sku_drink_data_valid.csv
测试集：/datasets/fs_sku_drink_data_test.csv
预测集：
```
## 4.PU-learning
...
# 集成
1. python脚本
```
/home/DI/zhouzx/code/sku_drink_label/voting.py：投票法  
输出：/home/DI/zhouzx/code/sku_drink_label/datasets/di_store_drink_label_predict.csv
```

# 数据入库
1、海豚脚本
/home/DI/zhouzx/code/sku_drink_label/di_store_drink_label_predict.py
2、clickhouse
/home/DI/zhouzx/code/sku_drink_label/di_store_drink_label_predict_CK.py
```clickhouse
-- clickhouse 创建表 删除表
	-- 分布式表
	CREATE TABLE di_store_drink_label_predict ON CLUSTER xw_clickhouse
	(   id String,
	    name String,
	    storetype String,
	    pred_plant int,
	    pred_juice int,
	    pred_protein int,
	    pred_flavored int,
	    pred_tea int,
	    pred_carbonated int,
	    pred_coffee int,
	    pred_water int,
	    pred_special int
	)
	ENGINE = Distributed('xw_clickhouse',
	 'ai_db',
	 'di_store_drink_label_predict_local',
	 rand())
	COMMENT '品类预测';
	
	CREATE TABLE di_store_drink_label_predict_local ON CLUSTER xw_clickhouse
	(   id String,
	    name String,
	    storetype String,
	    pred_plant int,
	    pred_juice int,
	    pred_protein int,
	    pred_flavored int,
	    pred_tea int,
	    pred_carbonated int,
	    pred_coffee int,
	    pred_water int,
	    pred_special int
	)
	ENGINE = MergeTree
	PRIMARY key id
	COMMENT '品类预测';
	
	--删除数据
	TRUNCATE TABLE di_store_drink_label_predict_local ON CLUSTER xw_clickhouse
	
	DROP table di_store_drink_label_predict ON CLUSTER xw_clickhouse;
	
	DROP table di_store_drink_label_predict_local ON CLUSTER xw_clickhouse;
```