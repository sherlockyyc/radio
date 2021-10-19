
# 项目框架

```
├── adv_method			## 存放各类攻击方法
│   ├── FGSM.py
│   ├── __init__.py
├── checpoint			## 存放防御模型训练出来的模型
├── data_loader			## 数据加载器
│   ├── __init__.py
│   ├── MNIST.py
├── defender			## 存放各种防御方法
├── figure			    ## 存放生成的各种图片（混淆矩阵等）
├── log			        ## 存放攻击过程的日志
├── metric			    ## 存放各种评测标准（准确率）
├── model_loader		## 存放各种分类模型
├── parameters		    ## 存放各种运行参数（其配合scirpts文件进行使用）
├── scripts		        ## 存放各种脚本文件（其配合parameters文件进行使用）
├── tmp_parameter		## 存放一些中间参数
├── user				## 攻击主程序
│   ├── attacker.py
│   ├── __init__.py
├── utils.py			## 功能函数
├── attack.py			## 攻击脚本
├── config.py			## 攻击的通用参数
├── defend.py			## 防御脚本
├── defend_config.py    ## 防御的通用参数
└── README.md			

```

# 攻击篇

## 如何运行一个攻击模型

### 1. yaml参数文件创建
在`parameters/attack`文件夹下创建一个`yaml`文件，并进行以下各种配置

1. 对使用的GPU进行配置`config.GPU`
- `use_gpu`： True表示采用GPU，True表示使用
- `device_id`：所采用的GPU的设备号，列表

2. 对攻击中进行通用设置`config.CONFIG`
- `dataset_name`： 攻击时所采用的数据集名称，注意：该名称对应于`data_loader`下的数据加载类，且存在一个数据config参数与其对应，：如`Rml2016_10a`
- `model_name`：进行攻击的目标模型，注意：该名称对应于`model_loader`下的模型类，且存在一个模型config参数与其对应，如：`VTCNN2`
- `criterion_name`：攻击时所采用的损失函数名称，注意：该名称只能是`torch.nn`中已存在的损失函数，如：`CrossEntropyLoss`
- `metrics`：所采用的各种评价标准，列表，注意：列表中的值只能是metric文件夹中所实现的评价标准，如：`['accuracy']`
- `attack_name`：攻击模型，注意：该名称对应于`adv_method`下的攻击类，且存在一个攻击config参数与其对应，如：`NAM`

3. 数据集参数设置--其名称对应于`config.CONFIG.dataset_name`，参数值对应于具体的参数，如
```
Rml2016_10a: {
      dirname: "/home/yuzhen/wireless/RML2016.10a",  # 数据集文件路径
      prop: 0.5,                     # 所占的比例
  },
```

4. 目标模型参数设置--其名称对应于`config.CONFIG.model_name`，参数值对应于该模型所对应的加载模型函数`load_model`的参数，如：
```
VTCNN2: {
      filepath: '/home/yuzhen/wireless/model/VTCNN2/VTCNN2_Epoch85.pkl' # 预训练模型存储位置
  },
```

5. 攻击方法参数设置--其名称对应于`config.CONFIG.attack_name`，参数值对应于具体的参数，如：
```
PGD: {
      eps: 1.0e-4,                 # 控制大小的参数
      epoch: 20,                  # 迭代次数
      is_target: False,           # 控制攻击方式，目标攻击、无目标攻击
      target: 3,                  # 目标攻击的目标
  },
```

6. 整体的攻击类型设置，可选白盒攻击、黑盒攻击和shift攻击
```
Switch_Method: {
      method: 'White_Attack',        # 可选['Black_Attack', 'White_Attack', 'Shifting_Attack']
  }
```

### 2. 通用参数的查看
查看攻击通用参数文件`config.py`中是否存在需要修改的参数（一般没啥要修改）


### 3. 书写脚本文件
在`scripts/attack`文件夹下创建一个`sh`文件, 加载上面的参数文件，并进行攻击，如：
```
python attack.py --config parameters/attack/pgd_vtcnn2_white_attack.yaml --vGPU 0
```

### 4. 运行该脚本文件
```
sh scripts/attack/***.sh
```


## 如何添加一个攻击方法

1. 在`adv_method`文件夹下，创建新攻击方法文件，如`attack.py`
2. 在`adv_method/__init__.py`中添加该文件的索引，如：
```
from adv_method.attack import *
```
3. 在攻击文件中进行攻击模型类`Attacker`的书写，其要继承`BaseMethod`父类，并写出`attack(x,y)` 攻击方法，该方法返回固定的结果 `x_adv, pertubation, logits, pred`
4. 在使用该攻击方法时，修改`config.CONFIG.attack_name`为该攻击类的名称`Attacker`，并写出其要使用的参数：
```
Attacker: {

}
```




# 防御篇

## 如何运行一个防御模型

### 1. yaml参数文件创建
在`parameters/defend`文件夹下创建一个`yaml`文件，并进行以下各种配置

1. 对使用的GPU进行配置`config.GPU`
- `use_gpu`： True表示采用GPU，True表示使用
- `device_id`：所采用的GPU的设备号，列表

2. 对攻击中进行通用设置`config.CONFIG`
- `dataset_name`： 攻击时所采用的数据集名称，注意：该名称对应于`data_loader`下的数据加载类，且存在一个数据config参数与其对应，：如`Rml2016_10a`
- `model_name`：进行攻击的目标模型，注意：该名称对应于`model_loader`下的模型类，且存在一个模型config参数与其对应，如：`VTCNN2`
- `criterion_name`：攻击时所采用的损失函数名称，注意：该名称只能是`torch.nn`中已存在的损失函数，如：`CrossEntropyLoss`
- `optimizer_name`：攻击时所采用的优化器名称，注意：该名称只能是`torch.optim`中已存在的损失函数，如：`Adam`
- `metrics`：所采用的各种评价标准，列表，注意：列表中的值只能是metric文件夹中所实现的评价标准，如：`['accuracy']`
- `adjust_lr`：是否在训练过程中使用自动变化学习率，`True` 表示自动变化学习率，并用`config.LrAdjust`进行自适应学习率的配置
- `load_method`：是否在训练开始之前加载预训练的模型，`True`表示加载，并用`config.LoadModel`选择要加载的预训练模型
- `defender_name`：攻击模型，注意：该名称对应于`defender`下的防御类，且存在一个攻击config参数与其对应，如：`FGSM_Adv_Trainer`

3. 数据集参数设置--其名称对应于`config.CONFIG.dataset_name`，参数值对应于具体的参数，如
```
Rml2016_10a: {
      dirname: "/home/yuzhen/wireless/RML2016.10a",  # 数据集文件路径
      prop: 0.5,                     # 所占的比例
  },
```

4. 目标模型参数设置--其名称对应于`config.CONFIG.model_name`，参数值对应于模型类的参数，如：
```
VTCNN2: {
      output_dim: 11,
  },
```

5. 防御方法参数设置--其名称对应于`config.CONFIG.defender_name`，参数值对应于具体的参数，如：
```
FGSM_Adv_Trainer: {
      clean_sigma: 0.5,
      adv_sigma: 0.5,
      alp_coef: 0.5
  }
```

6. 优化器设置，其名称对应于`config.CONFIG.optimizer_name`：
```
Adam: {
      lr: 0.003,                  # 学习率
      weight_decay: 1.0e-3,        # 权重衰减
  },
```


7. 训练过程参数设置, 包括训练epcoh数目和batch_size：
```
ARG: {
      epoch: 1200,         # 训练epoch
      batch_size: 1024,    # 训练集batch_size
  },
```


### 2. 通用参数的查看
查看攻击通用参数文件`defend_config.py`中是否存在需要修改的参数（一般没啥要修改）


### 3. 书写脚本文件
在`scripts/defend`文件夹下创建一个`sh`文件, 加载上面的参数文件，并进行攻击，如：
```
python defend.py --config parameters/defend/fgsm_adv_train.yaml --vGPU 0
```

### 4. 运行该脚本文件
```
sh scripts/defend/***.sh
```


## 如何添加一个防御方法

1. 在`defender`文件夹下，创建新攻击方法文件，如`defender.py`
2. 在`defender/__init__.py`中添加该文件的索引，如：
```
from defender.defender import *
```
3. 在防御文件中进行防御模型类`Defender`的书写，其要继承`BaseTrainer`父类，并写出`_train_epoch(epoch)` 训练函数，该方法返回训练过程的`loss`和评测结果`metric`，将其放入到`log`字典中进行输出
4. 在使用该攻击方法时，修改`config.CONFIG.defender_name`为该攻击类的名称`Defender`，并写出其要使用的参数：
```
Defender: {

}
```