

## 项目框架

```
├── advMethod			##各类攻击方法
│   ├── FGSM.py
│   ├── __init__.py
├── attack.py			##攻击脚本
├── config.py			##各类配置
├── dataLoader			##数据加载
│   ├── __init__.py
│   ├── MNIST.py
├── modelLoader			##模型加载
│   ├── __init__.py
│   ├── lenet.py
├── README.md
├── user				##攻击主程序
│   ├── attacker.py
│   ├── __init__.py
└── utils.py			##功能函数

```



## Config

在开始攻击前，设置GPU、数据集、攻击模型、损失函数、攻击方法、攻击手段

### 1) GPU

在`config.GPU`中进行设置`

- `use_gpu`： True表示采用GPU
- `device_id`：所采用的GPU的设备号



### 2) 数据集

以Mnist为例

**选择框架存在的数据集**

1. 在`config.CONFIG['dataset_name']`中设置数据集名称

**添加框架之外的数据集**

1. 在`dataLoader`文件夹中写好数据加载函数，`MnistTrainSet、MnistTestSet`
2. 在`config`中添加数据集参数（数据加载函数所需要的参数）

```
self.Mnist = dict(dirname='',)  ##注意config中的数据集名称要与dataLoader的数据类名称相同
```

3. 在`config.CONFIG['dataset_name']`中设置所选的数据集名称



### 3) 攻击模型

以LeNet为例

1. 在`modelLoader`文件夹中添加模型类LeNet，并写好模型加载函数loadLeNet，注意函数中模型名称一致
2. 在`config`中添加模型类参数（loadLeNet所需要的参数）

```
self.LeNet = dict(filepath='',)  ##注意config中的模型类名称要与modelLoader的模型类名称相同
```

3. 在`config.CONFIG['model_name']`中设置所选的攻击模型名称



### 4) 损失函数

**选择`torch.nn`中的损失函数**

1. 在`config.CONFIG['criterion_name']`中设置所选的损失函数名称CrossEntropyLoss



### 5) 攻击方法

以FGSM为例

**选择已有的攻击方法**

1. 在`config.CONFIG['attack_name']`中设置攻击方法名称

**自己手动添加攻击方法**

1. 在`advMethod`文件夹中添加文件`FGSM.py`，在`__init__.py`中导入该文件`from advMethod.FGSM import *`
2. 在`FGSM.py`文件中，写好攻击方法，要包含方法`attack`，`_attackWithNoTarget`，`_attackWithTarget`，在进行攻击时仅调用`attack`方法，后面两个表示无目标攻击和吴彪攻击，返回一批对抗样本和对抗扰动。`attack`方法返回一批对抗样本、对抗扰动、攻击后的标签。
3. 在`config`中添加攻击类参数（`attack`方法所需要的参数）

```
self.FGSM = dict(eps=0.2,)  ##注意config中的攻击类名称要与advMethod的攻击类名称相同
```

4. 在`config.CONFIG['attack_name']`中设置所选的攻击模型名称

## Start Attack

### 1) 设置攻击手段

在`attack.py`中进行设置

**攻击单个图片**

1. 在`attack.py`文件的开始攻击部分，选择图片及对应的标签，采用`attacker.attackOneImage(x,y)`来进行单图片攻击，返回值 对抗样本x_adv，对抗扰动pertubation，攻击后的标签nowLabel
2. 注意：此处输入样本`x` 为三维，`y` 为list

**攻击一组图片**

1. 在`attack.py`文件的开始攻击部分，选择一组图片及对应的标签，采用`attacker.attackOnce(x,y)`来进行攻击，返回值 一组对抗样本x_advs，一组对抗扰动pertubations，一组攻击后的标签nowLabels
2. 注意：此处输入样本`x` 为四维，`y` 为list

**攻击一个数据集**

1. 在`attack.py`文件的开始攻击部分，选择要攻击的数据集，采用`attacker.attackSet(TestLoader)`来进行攻击，返回值 攻击后的模型准确率acc
2. 注意：`TestLoader`的输出为`(x,y)`

### 2) 运行

```
python attack.py
```



## 攻击效果

### LeNet

| 攻击方法\攻击后的准确率,平均扰动值 | MNSIT           | MNIST            | 备注                                 |
| ---------------------------------- | --------------- | ---------------- | ------------------------------------ |
| 未攻击                             | 98%             | 99.19%           |                                      |
| FGSM(eps=0.1,noTarget)             | 65.74%---0.0375 | 74.52%---0.0368  |                                      |
| FGSM(eps=0.2,noTarget)             | 10.97%---0.075  | 21.28%---0.0736  |                                      |
| BIM(eps=0.01, epoch=10, noTarget)  | 56.87%---0.03   | 55.42%---0.0269  |                                      |
| BIM(eps=0.03,epoch=10, noTarget)   | 1.2%---0.08     | 0.02%---0.069    |                                      |
| DeepFool(max_iter = 50)            |                 | 48.77%---0.00978 | 需要一个图片一个图片的迭代，速度很慢 |

**发现**

对于模型准确率较低的模型A，模型准确率较高的模型B

FGSM对A更容易攻击，及攻击后的准确率更低

BIM对B更容易攻击