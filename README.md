# 数据集的准备
~~down_di_sku_labeling_data.py 查询待品类打标的数据~~  
1. 代码合并到prototypical.py 查询待品类打标的数据  
数据库：standard_db
关联表：di_store_sku_drink_label、
di_sku、di_store_classify_dedupe、
di_store_dedupe_labeling
其中di_store_sku_drink_label表需要通过di_store_sku和di_store_brand表维护

# 预训练
## 原型网络
prototypical.py 原型网络训练模型
prototypical_predict.py 对待测数据进行预测
## textCNN
...
## PU-learning
...
# 集成
voting.py
 
# 数据入库
1、海豚脚本
di_store_drink_label_predict.py
2、clickhouse
di_store_drink_label_predict_CK.py
```
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