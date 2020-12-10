import time
class Config(object):
    def __init__(self):
        self.ENV = 'default'            #当前的环境参数
        self.Introduce = 'Not at the moment'    #对此次实验的描述
        self.VERSION = 1                #当前的版本


        ##################################################GPU配置
        self.GPU = dict(
            use_gpu = True,             #是否使用GPU，True表示使用
            device_id = [0],            #所使用的GPU设备号，type=list
        )


        self.CONFIG = dict(
            dataset_name = 'Rml2016_10a',     #所选择的数据集的名称
            model_name = 'VTCNN2',       #攻击模型的名称
            criterion_name = 'CrossEntropyLoss',       #损失函数的名称
            metrics = ['accuracy'],        # 评价标准的名称（metric文件夹中）
            attack_name = 'Shifting_MI_FGSM',       #设定攻击方法的名称
        )



        #################################################模型选择
        ##########################模型参数
        self.VTCNN2 = dict(
            filepath = '/home/baiding/Study/research/radio/log/VTCNN2/model/VTCNN2_Epoch85.pkl'
        )
        self.Based_GRU = dict(
            filepath = '/home/baiding/Study/research/radio/log/Based_GRU/model/Based_GRU_Epoch1260.pkl'
        )
        self.Based_LSTM = dict(
            filepath = ''
        )
        self.Based_VGG = dict(
            filepath = '/home/baiding/Study/research/radio/log/Based_VGG/modelBased_VGG_Epoch1160.pkl'
        )
        self.Based_ResNet = dict(
            filepath = '/home/baiding/Study/research/radio/log/Based_ResNet/model/Based_ResNet_Epoch1160.pkl'
        )
        self.CLDNN = dict(
            filepath = '/home/baiding/Study/research/radio/log/CLDNN_GRU3/model/CLDNN_Epoch1160.pkl'
        )
        # self.VTCNN2 = dict(
        #     filepath = '/home/yuzhen/wireless/model/VTCNN2/VTCNN2_Epoch85.pkl'
        # )
        # self.Based_GRU = dict(
        #     filepath = '/home/yuzhen/wireless/model/Based_GRU/Based_GRU_Epoch1260.pkl'
        # )
        # self.Based_LSTM = dict(
        #     filepath = ''
        # )
        # self.Based_VGG = dict(
        #     filepath = '/home/yuzhen/wireless/model/Based_VGG/Based_VGG_Epoch1160.pkl'
        # )
        # self.Based_ResNet = dict(
        #     filepath = '/home/yuzhen/wireless/model/Based_ResNet/Based_ResNet_Epoch1160.pkl'
        # )
        # self.CLDNN = dict(
        #     filepath = '/home/yuzhen/wireless/model/CLDNN_GRU3/CLDNN_Epoch1160.pkl'
        # )




        #################################################损失函数选择



        #################################################数据集
        ##########################数据集参数
        self.Mnist = dict(
            dirname = '/home/baiding/Desktop/Study/Deep/datasets/MNIST/raw',            #MNIST数据集存放的文件夹
            is_vector = False,         #False表示得到784维向量数据，True表示得到28*28的图片数据
        )
        self.Rml2016_10a = dict(
            dirname = "/home/baiding/Study/research/radio/RML2016.10a",  # 数据集文件路径
            prop = 0.8,                     # 所占的比例
        )
        # self.Rml2016_10a = dict(
        #     dirname = "/home/yuzhen/wireless/RML2016.10a",  # 数据集文件路径
        #     prop = 0.8,                     # 所占的比例
        # )



        #################################################攻击方法
        ##########################FGSM方法
        self.FGSM = dict(
            eps = 3e-3,                  #FGSM的控制大小的参数
            is_target = False,           #控制攻击方式，目标攻击、无目标攻击
            target = 3,               #目标攻击的目标
        )
        ##########################BIM方法
        self.BIM = dict(
            eps = 1e-5,                  #BIM的控制大小的参数
            epoch = 10,                 #BIM的迭代次数
            is_target = False,           #控制攻击方式，目标攻击、无目标攻击
            target = 3,               #目标攻击的目标
        )
        ##########################DeepFool方法
        self.DeepFool = dict(
            max_iter = 10,              #最大寻找次数
        )
        ## PGD
        self.PGD = dict(
            eps = 1e-4,                 # 控制大小的参数
            epoch = 30,                  # 迭代次数
            is_target = False,           # 控制攻击方式，目标攻击、无目标攻击
            target = 3,                  # 目标攻击的目标
        )
        ## MI-FGSM
        self.MI_FGSM = dict(
            eps = 13*1e-4,                # 控制大小的参数
            epoch = 30,                 # 迭代次数
            is_target = False,          # 控制攻击方式，目标攻击、无目标攻击
            target = 3,                 # 目标攻击的目标
            mu = 1,                     # momentum参数
        )
        ## NI-FGSM
        self.NI_FGSM = dict(
            eps = 1e-4,                # 控制大小的参数
            epoch = 30,                 # 迭代次数
            is_target = False,          # 控制攻击方式，目标攻击、无目标攻击
            target = 3,                 # 目标攻击的目标
            mu = 1,                     # momentum参数
        )
        ## Shifting_MI-FGSM
        self.Shifting_MI_FGSM = dict(
            eps = 13*1e-4,                # 控制大小的参数
            epoch = 30,                 # 迭代次数
            is_target = False,          # 控制攻击方式，目标攻击、无目标攻击
            target = 3,                 # 目标攻击的目标
            mu = 1,                     # momentum参数
        )

        #################################################log
        self.Checkpoint = dict(
            log_dir = './log',          #log所在的文件夹
            log_filename = '{}_{}_{}_{}_V{}_{}.log'.format(
                self.CONFIG['dataset_name'], self.CONFIG['model_name'],
                self.CONFIG['criterion_name'], self.CONFIG['attack_name'],
                self.VERSION, time.strftime("%m-%d %H:%M")),              #log文件名称
        )
        ## 针对attacker的特定函数
        self.Switch_Method = dict(
            method = 'Shifting_Attack',        # 可选['Black_Attack', 'White_Attack', 'Shifting_Attack']
        )
        self.Black_Attack = dict(
            threat_model = 'VTCNN2',
            black_model = 'Based_GRU',
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
        log['Switch_Method'] = self.Switch_Method['method']
        if self.Switch_Method['method'] != 'White_Attack':
            log.update(self.Black_Attack)
        return log
        
    

    
    