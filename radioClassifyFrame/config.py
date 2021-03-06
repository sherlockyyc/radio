import datetime
import time

class Config(object):
    def __init__(self):
        self.ENV = 'default'            # 当前的环境参数
        self.Introduce = 'Not at the moment'    #对 此次实验的描述
        self.VERSION = 1                # 当前版本


        #------------------------------------------------GPU配置
        self.GPU = dict(
            use_gpu = True,             # 是否使用GPU，True表示使用
            device_id = [2],            # 所使用的GPU设备号，type=list
        )


        self.CONFIG = dict(
            dataset_name = 'Rml2016_10a',     # 所选择的数据集的名称
            model_name = 'VTCNN2',       # 攻击模型的名称
            criterion_name = 'CrossEntropyLoss',       # 失函数的名称
            optimizer_name = 'Adam',     # 优化器的名称（torch.nn中）
            metrics = ['accuary'],        # 评价标准的名称（metric文件夹中）
            adjust_lr = True,               # 是否自动的变化学习率
            load_model = True,              # 是否加载预训练模型（测试、迁移）
        )

        #------------------------------------------------训练参数设置
        self.ARG = dict(
            epoch = 1200,         # 训练epoch
            batch_size = 1024,    # 训练集batch_size
        )

        #------------------------------------------------损失函数选择
        
        
        
        #------------------------------------------------网络模型
        self.VTCNN2 = dict(
            output_dim = 11,
        )
        self.Based_LSTM = dict(
            output_dim = 11,
        )
        self.Based_VGG = dict(
            output_dim = 11,
        )
        self.Based_ResNet = dict(
            output_dim = 11,
        )
        self.Based_GRU = dict(
            output_dim = 11,
        )
        self.Based_Transformer_Baseline = dict(
            output_dim = 11,
        )
        self.CLDNN = dict(
            output_dim = 11,
        )

        #------------------------------------------------优化器
        self.Adam = dict(
            lr = 0.003,                  # 学习率
            weight_decay = 5e-3,        # 权重衰减
        )


        #------------------------------------------------数据集
        #--------------------------------数据集参数
        self.Rml2016_10a = dict(
            dirname = "/home/yuzhen/wireless/RML2016.10a",  # 数据集文件路径
            prop = 0.5,                     # 所占的比例
        )

        
        #------------------------------------------------学习率变化
        self.LrAdjust = dict(
            lr_step = 10,                   # 学习率变化的间隔
            lr_decay = 0.5,                 # 学习率变化的幅度
            increase_bottom = 5,            # 退火前学习率增加的上界
            increase_amp = 1.1,             # 学习率增加的幅度
            warm_lr = 0.0005,               # 当学习率小于1e-8时，恢复学习率为warm_lr
        )


        #------------------------------------------------模型加载
        self.LoadModel = dict(
            filename = '/home/yuzhen/wireless/model/VTCNN2/VTCNN2_Epoch85.pkl',     #加载模型的位置，与上面模型要对应
            base_epoch = 0,           # 预训练的基础epoch
        )


        #------------------------------------------------checkpoint
        self.Checkpoint = dict(
            checkpoint_dir = './checkpoint/{}_{}_V{}'.format(
                self.CONFIG['dataset_name'], self.CONFIG['model_name'],
                self.VERSION),                          # checkpoint 所在的文件夹
            checkpoint_file_format = self.CONFIG['model_name']+'_Epoch{}.pkl',     #模型文件名称格式，分别表示模型名称、Epoch
            model_best = 'model_best.ptk',            #最好的模型名称，暂时未用到
            log_file = '{}_{}.log'.format(
                self.CONFIG['model_name'], time.strftime("%m-%d_%H-%M")
            ),                         #log文件名称
            save_period = 5,                            #模型的存储间隔
        )


    def log_output(self):
        log = {}
        log['ENV'] = self.ENV
        log['Introduce'] = self.Introduce
        log['CONFIG'] = self.CONFIG
        for name,value in self.CONFIG.items():
            if type(value) is str and hasattr(self,value):
                log[value] = getattr(self,value)
            else:
                log[name] = value
        for name,value in self.ARG.items():
            log[name] = value
        log['LoadModel'] = self.LoadModel
        return log
