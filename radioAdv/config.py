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
            dataset_name = 'Rml2016_10a',       #所选择的数据集的名称
            model_name = 'VTCNN2',              #白盒攻击模型的名称
            criterion_name = 'CrossEntropyLoss',       #损失函数的名称
            metrics = ['accuracy'],        # 评价标准的名称（metric文件夹中）
            attack_name = 'FGSM',       #设定攻击方法的名称
        )



        #################################################模型选择
        ##########################模型参数
        self.VTCNN2 = dict(
            # filepath = '/home/yuzhen/wireless/model/VTCNN2/VTCNN2_Attack.pkl'
            filepath = './checkpoint/Rml2016_10a_VTCNN2_V1/VTCNN2_Epoch130.pkl'
        )
        # self.VTCNN2 = dict(
        #     filepath = '/home/yuzhen/wireless/model/VTCNN2/Adv_Train_VTCNN2_Epoch230.pkl'
        # )
        self.Based_GRU = dict(
            filepath = '/home/yuzhen/wireless/model/Based_GRU/Based_GRU_Attack.pkl'
        )
        # self.Based_GRU = dict(
        #     filepath = '/home/yuzhen/wireless/model/Based_GRU/Adv_Train_Based_GRU_Epoch170.pkl'
        # )
        self.Based_LSTM = dict(
            filepath = '/home/yuzhen/wireless/model/Based_LSTM/Based_LSTM_Attack.pkl'
        )
        self.Based_VGG = dict(
            filepath = '/home/yuzhen/wireless/model/Based_VGG/Based_VGG_Attack.pkl'
        )
        self.Based_ResNet = dict(
            filepath = '/home/yuzhen/wireless/model/Based_ResNet/Based_ResNet_Attack.pkl'
        )
        self.CLDNN = dict(
            filepath = '/home/yuzhen/wireless/model/CLDNN/CLDNN_Attack.pkl'
        )




        #################################################损失函数选择



        #################################################数据集
        ##########################数据集参数
        self.Mnist = dict(
            dirname = '/home/baiding/Desktop/Study/Deep/datasets/MNIST/raw',            #MNIST数据集存放的文件夹
            is_vector = False,         #False表示得到784维向量数据，True表示得到28*28的图片数据
        )
        # self.Rml2016_10a = dict(
        #     dirname = "/home/baiding/Study/research/radio/RML2016.10a",  # 数据集文件路径
        #     prop = 0.5,                     # 所占的比例
        # )
        self.Rml2016_10a = dict(
            dirname = "/home/yuzhen/wireless/RML2016.10a",  # 数据集文件路径
            prop = 0.5,                     # 所占的比例
        )



        #################################################攻击方法
        ##########################FGSM方法
        self.FGSM = dict(
            eps = 10*1e-4,                  #FGSM的控制大小的参数
            is_target = False,           #控制攻击方式，目标攻击、无目标攻击
            target = 3,               #目标攻击的目标
        )
        ##########################BIM方法
        self.BIM = dict(
            eps = 1e-4,                  #BIM的控制大小的参数
            epoch = 20,                 #BIM的迭代次数
            is_target = False,           #控制攻击方式，目标攻击、无目标攻击
            target = 3,               #目标攻击的目标
        )
        ##########################DeepFool方法
        self.DeepFool = dict(
            max_iter = 5,              #最大寻找次数
            eps = 0.0001
        )
        ## PGD
        self.PGD = dict(
            eps = 1e-4,                 # 控制大小的参数
            epoch = 20,                  # 迭代次数
            is_target = False,           # 控制攻击方式，目标攻击、无目标攻击
            target = 3,                  # 目标攻击的目标
        )
        ## MI-FGSM
        self.MI_FGSM = dict(
            eps = 1e-4,                # 控制大小的参数
            epoch = 30,                 # 迭代次数
            is_target = False,          # 控制攻击方式，目标攻击、无目标攻击
            target = 3,                 # 目标攻击的目标
            mu = 1,                     # momentum参数
        )
        ## NI-FGSM
        self.NI_FGSM = dict(
            eps = 1e-4,                # 控制大小的参数
            epoch = 20,                 # 迭代次数
            is_target = False,          # 控制攻击方式，目标攻击、无目标攻击
            target = 3,                 # 目标攻击的目标
            mu = 1,                     # momentum参数
        )
        # CW
        self.CW = dict(
            binary_search_steps=10, 
            n_iters=200, 
            c=1e-4, 
            kappa=0, 
            lr=0.0001, 
            is_target=False, 
            target=0,
            eps = 0.001
        )
        self.Jamming = dict(
            mean = 0.003,
            std = 1.0
        )
        self.NAM = dict(
            eps = 1e-4,                # 控制大小的参数
            epoch = 20,                 # 迭代次数
            is_target = False,          # 控制攻击方式，目标攻击、无目标攻击
            target = 3,                 # 目标攻击的目标
            beta1 = 0.9,
            beta2 = 0.999,                    
        )
        

        self.PIM_FGSM = dict(
            eps = 20*1e-4,                # 控制大小的参数
            is_target = False,          # 控制攻击方式，目标攻击、无目标攻击
            target = 3,                 # 目标攻击的目标
            mu = 1,                     # momentum参数
            shift = 8,                 # 在两边扩充noise, 20 + noise + 20
            sample_num = 128,             # 采样点
        )
        self.PIM_PGD = dict(
            eps = 1e-4,                # 控制大小的参数
            epoch = 20,                 # 迭代次数
            is_target = False,          # 控制攻击方式，目标攻击、无目标攻击
            target = 3,                 # 目标攻击的目标
            mu = 1,                     # momentum参数
            shift = 8,                 # 在两边扩充noise, 20 + noise + 20
            sample_num = 128,             # 采样点
        )
        self.PIM_MIM = dict(
            eps = 1e-4,                # 控制大小的参数
            epoch = 20,                 # 迭代次数
            is_target = False,          # 控制攻击方式，目标攻击、无目标攻击
            target = 3,                 # 目标攻击的目标
            mu = 1,                     # momentum参数
            shift = 8,                 # 在两边扩充noise, 20 + noise + 20
            sample_num = 128,             # 采样点
        )
        self.PIM_DeepFool = dict(
            max_iter = 35,              #最大寻找次数
            is_target = False,          # 控制攻击方式，目标攻击、无目标攻击
            target = 3,                 # 目标攻击的目标
            shift = 8,                 # 在两边扩充noise, 20 + noise + 20
            sample_num = 128,             # 采样点
            eps = 0.002
        )
        self.PIM_CW = dict(
            n_iters=10, 
            c=1e-4, 
            kappa=0, 
            lr=0.0001, 
            is_target = False,          # 控制攻击方式，目标攻击、无目标攻击
            target = 3,                 # 目标攻击的目标
            mu = 1,                     # momentum参数
            shift = 8,                 # 在两边扩充noise, 20 + noise + 20
            sample_num = 128,             # 采样点
            eps = 0.002,
        )
        self.PIM_NAM = dict(
            eps = 1e-4,                # 控制大小的参数
            epoch = 20,                 # 迭代次数
            is_target = False,          # 控制攻击方式，目标攻击、无目标攻击
            target = 3,                 # 目标攻击的目标
            beta1 = 0.9,
            beta2 = 0.999,   
            shift = 8,                 # 在两边扩充noise, 20 + noise + 20
            sample_num = 128,             # 采样点
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
            method = 'White_Attack',        # 可选['Black_Attack', 'White_Attack', 'Shifting_Attack']
        )
        self.Black_Attack = dict(
            threat_model = 'VTCNN2',
            black_model = 'Based_GRU',
            is_uap = False,
            eps = 0.002,
        )
        self.Shifting_Attack = dict(
            load_parameter = False,         # 是否加载预攻击的扰动
            parameter_path = './tmp_parameter/vtcnn2_pim_nam_0020_s.p',   #
            is_save_parameter = True,
            shift_k = 64,
            is_uap = False,
            eps = 0.002,
            is_sim = False,
            save_k = 4,
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
        if self.Switch_Method['method'] == 'Black_Attack':
            log.update(self.Black_Attack)
        if self.Switch_Method['method'] == 'Shifting_Attack':
            log.update(self.Black_Attack)
            log.update(self.Shifting_Attack)
        return log

    def load_parameter(self, parameter):
        """[加载parameter内的参数]

        Args:
            parameter ([dict]): [由yaml文件导出的字典，包含有各个属性]
        """
        for key, value in parameter.items():
            if hasattr(self, key):
                if type(value) is dict:
                    orig_config = getattr(self, key)
                    if orig_config.keys() == value.keys():
                        setattr(self, key, value)
                    else:
                        redundant_key = value.keys() - orig_config.keys()
                        if redundant_key:
                            msg = "there are many redundant keys in config file, e.g.:  " + str(redundant_key)
                            assert None, msg
                        
                        lack_key = orig_config.keys() - value.keys()
                        if lack_key:
                            msg = "there are many lack keys in config file, e.g.:  " + str(lack_key)
                            assert None, msg
                else:
                    setattr(self, key, value)
            else:
                setattr(self, key, value)
        
        # 更新受影响的参数(不太重要的参数)
        self.Checkpoint['log_filename'] =  '{}_{}_V{}_{}.log'.format(self.CONFIG['model_name'], self.CONFIG['attack_name'], self.VERSION, time.strftime("%m-%d_%H-%M"))              #log文件名称

        return None
        
    

    
    
