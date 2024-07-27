from os import path
from numpy import array
from mne import channels
from matplotlib.colors import TABLEAU_COLORS


# import yaml

class IndexConfig:
    """
    各功能在ui界面、功能上的索引
    """
    SeiD_ESA_ui_idx = 1
    AD_ui_idx = 2
    SD_ui_idx = 3
    HFO_ui_idx = 4
    VD_ui_idx = 5

    SeiD_ESA_idx = SeiD_ESA_ui_idx - 1
    AD_idx = AD_ui_idx - 1
    SD_idx = SD_ui_idx - 1
    HFO_idx = HFO_ui_idx - 1
    VD_idx = VD_ui_idx - 1


class ModelConfig:
    SeiD_ESA_model = ['', 'DSMN-ESS', 'R3DClassifier']
    AD_model = ['', 'Resnet34_AE_BCELoss', 'Resnet34_SkipAE_BCELoss', 'Resnet34_MemAE_BCELoss',
                'Resnet34_VAE_BCELoss', 'SENet18_AE_BCELoss', 'SENet18_SkipAE_BCELoss',
                'SENet18_MemAE_BCELoss', 'SENet18_VAE_BCELoss', 'VGG16_AE_BCELoss', 'VGG16_SkipAE_BCELoss',
                'VGG16_MemAE_BCELoss', 'VGG16_VAE_BCELoss', 'DenseNet121_AE_BCELoss',
                'DenseNet121_SkipAE_BCELoss', 'DenseNet121_MemAE_BCELoss', 'DenseNet121_VAE_BCELoss']
    SD_model = ['', 'Template Matching', 'Unet+ResNet34']
    HFO_model = ['', 'MFCNN']
    VD_model = ['', 'yolov5l_3dResnet']

    SeiD_ESA_model_des = 'This model requires the <INPUT> .EDF FILE:\n' \
                         'sfreq: 1000Hz\n' \
                         '21 channels, which can be selected in Select Signals Form per button ESA/SeiD (21 channels)\n' \
                         'preprocessing: None\n' \
                         'time span: 4s'
    AD_model_des = 'This model requires the <INPUT> .EDF FILE:\n' \
                   'sfreq: 1000Hz\n' \
                   '19 channels, which can be selected in Select Signals Form per button AD/SD (19 channels)\n' \
                   'preprocessing: None\n' \
                   'time span: 11s'
    SD_model_des = 'This model requires the <INPUT> .EDF FILE:\n' \
                   'sfreq: 500Hz\n' \
                   '19 channels, which can be selected in Select Signals Form per button AD/SD (19 channels)\n' \
                   'preprocessing: None\n' \
                   'time span: >=0.3s(Template) | multiple of 30s(Semantics)'
    HFO_model_des = 'This model requires the <INPUT> .EDF FILE:\n' \
                    'sfreq: 1000Hz\n' \
                    '19 channels, which can be selected in Select Signals Form per button AD/SD (19 channels)\n' \
                    'preprocessing: None\n' \
                    'time span: >=1s'
    VD_model_des = 'This model requires the <INPUT> .MP4 FILE:\n' \
                   'frame per second: 20'

    @staticmethod
    def get_des(model):
        if model in ModelConfig.SeiD_ESA_model:
            return ModelConfig.SeiD_ESA_model_des
        elif model in ModelConfig.AD_model:
            return ModelConfig.AD_model_des
        elif model in ModelConfig.SD_model:
            return ModelConfig.SD_model_des
        elif model in ModelConfig.HFO_model:
            return ModelConfig.HFO_model_des
        elif model == 'VD':
            return ModelConfig.VD_model_des


class ChannelConfig:
    xpos = [-0.0293387312092767, -0.0768097954926838, -0.051775707348028, -0.0949421285668141, -0.0683372810321719,
            -0.0768097954926838, -0.051775707348028,
            4.18445162392412E-18, -0.0293387312092767, 0.0293387312092767, 0.051775707348028, 0.0768097954926838,
            0.0683372810321719, 0.0949421285668141,
            0.051775707348028, 0.0768097954926838, 0.0293387312092767, 4.18445162392412E-18, 0]

    ypos = [0.0902953300444008, 0.0558055829928292, 0.0639376737816708, 0, 0, -0.0558055829928292, -0.0639376737816708,
            -0.0683372810321719,
            -0.0902953300444008, -0.0902953300444008, -0.0639376737816708, -0.0558055829928292, 0, 0,
            0.0639376737816708,
            0.0558055829928292, 0.0902953300444008, 0.0683372810321719, 0]

    zpos = [-0.00331545218673759, -0.00331545218673759, 0.0475, -0.00331545218673759, 0.0659925451936047,
            -0.00331545218673759, 0.0475, 0.0659925451936047,
            -0.00331545218673759, -0.00331545218673759, 0.0475, -0.00331545218673759, 0.0659925451936047,
            -0.00331545218673759, 0.0475, -0.00331545218673759,
            -0.00331545218673759, 0.0659925451936047, 0.095]

    freq_bands = [(0, 4), (4, 8), (8, 12), (12, 30), (30, 45)]

    colors = list(TABLEAU_COLORS.values())  # 获取颜色列表

    # color_dict = {'Fp1': '#DC143C', 'Fp2': '#C71585', 'F3': '#DDA0DD', 'F4': '#BA55D3', 'C3': '#9932CC',
    #               'C4': '#6A5ACD', 'P3': '#4169E1', 'P4': '#00BFFF', 'O1': '#008B8B', 'O2': '#3CB371', 'F7': '#556B2F',
    #               'F8': '#808000', 'T3': '#FFD700', 'T4': '#DAA520', 'T5': '#D2B48C', 'T6': '#D2691E', 'A1': '#FFFF00',
    #               'A2': '#8B4513', 'Fz': '#FF7F50', 'Cz': '#CD5C5C', 'Pz': '#FFB6C1'}

    # test_edf_channels = ['Fp1', 'F7', 'T3', 'T5', 'O1', 'F3', 'C3', 'P3', 'A1', 'Fz', 'Cz', 'Fp2', 'F8', 'T4',
    #                      'T6', 'O2', 'F4', 'C4', 'P4', 'A2', 'Fpz', 'Pz', 'ECG', 'X2', 'X3', 'X4', 'X5', 'X6',
    #                      'X7', 'X8', 'X9', 'X10', 'X11', 'X12', 'X13', 'X14', 'X15', 'X16', 'X17', 'X18',
    #                      'DC1', 'DC2', 'DC3', 'DC4', 'OSAT', 'PR']  # 46

    def __init__(self):
        # self.prefix = None
        self.topomap_channels = ['Fp1', 'F7', 'F3', 'T3', 'C3', 'T5', 'P3', 'Pz', 'O1', 'O2', 'P4', 'T6', 'C4', 'T4',
                                 'F4', 'F8', 'Fp2', 'Fz', 'Cz']  # 19
        self.SeiD_ESA_channels = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7',
                                  'F8', 'T3', 'T4', 'T5', 'T6', 'A1', 'A2', 'Fz', 'Cz', 'Pz']  # 21 + A1、A2
        self.AD_SD_channels = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4',
                               'T5', 'T6', 'Fz', 'Cz', 'Pz']  # 19

        # pre, suf = self.get_pre_suf()
        # if pre is not None:
        #     self.prefix = pre
        #
        # self.topomap_channels = self.set_pre_suf(self.topomap_channels, pre, suf)
        # self.ESA_channels = self.set_pre_suf(self.ESA_channels, pre, suf)
        # self.AD_SD_channels = self.set_pre_suf(self.AD_SD_channels, pre, suf)

    def get_montage(self):
        position = {}
        for i in range(19):
            ch_name = self.topomap_channels[i]
            pos = [ChannelConfig.xpos[i], ChannelConfig.ypos[i], ChannelConfig.zpos[i]]
            position[ch_name] = array(pos)
        montage = channels.make_dig_montage(ch_pos=position)
        return montage

    # @staticmethod
    # def get_pre_suf():
    #     with open('config.yaml', 'r') as config_file:
    #         config = yaml.safe_load(config_file)
    #         if config is None:
    #             prefix = None
    #             suffix = None
    #         else:
    #             if 'prefix' not in config:
    #                 prefix = None
    #             else:
    #                 prefix = config.get('prefix', '')
    #             if 'suffix' not in config:
    #                 suffix = None
    #             else:
    #                 suffix = config.get('suffix', '')
    #     return prefix, suffix
    #
    # @staticmethod
    # def set_pre_suf(chns, pre=None, suf=None):
    #     if pre is not None:
    #         chns = [pre + chn for chn in chns]
    #     if suf is not None:
    #         chns = [chn + suf for chn in chns]
    #     return chns


class AddressConfig:
    # offline_analyse
    # Python写路径字符串用/或\\均可(Python的路径处理库会根据操作系统自动适应正确的路径分隔符)
    # relative_path -> absolute_path

    # yaml_adr = "./config.yaml"
    log_folder_path = './utils/diary'

    @staticmethod
    def get_esa_adr(name):
        hashtable = {
            'fm': "./ESA/offline_analyse/feature_map/fm.png",
            'feature': "./ESA/offline_analyse/feature/feature.png",
            'STFT': "./ESA/offline_analyse/STFT/",
            'res': "./ESA/offline_analyse/result/res.png",
            'cp': "./ESA/A3D-EEG_epoch-19.pth.tar"
        }
        return path.abspath(hashtable.get(name, ''))

    @staticmethod
    def get_ad_adr(name, model_name=None):
        hashtable = {
            'idx': f"./AD/offline_analyse/index/{model_name}.html",
            'topo': "./AD/offline_analyse/topomap/topo.png",
            'res': "./AD/offline_analyse/result/res.png",
        }
        return path.abspath(hashtable.get(name, ''))

    @staticmethod
    def get_sd_adr(name):
        hashtable = {
            'idx': "./SD/offline_analyse/index/idx.html",
            'fam': "./SD/offline_analyse/family/fam.png",
            'res': "./SD/offline_analyse/result/res.png",
            'mat': "./SD/mat",
            'npz': "./SD/npz",
        }
        return path.abspath(hashtable.get(name, ''))

    @staticmethod
    def get_seid_adr(name):
        hashtable = {
            'fm': "./SeiD/offline_analyse/feature_map/fm.png",
            'feature': "./SeiD/offline_analyse/feature/feature.png",
            'STFT': "./SeiD/offline_analyse/STFT/",
            'res': "./SeiD/offline_analyse/result/res.png",
            'cp': "./SeiD/0.15-EEG_epoch-19.pth.tar"
        }
        return path.abspath(hashtable.get(name, ''))

    @staticmethod
    def get_hfo_adr(name):
        hashtable = {
            'cp': "./HFO/model_weights.pth"
        }
        return path.abspath(hashtable.get(name, ''))

    @staticmethod
    def get_vd_adr(name):
        hashtable = {
            'cp1': "./VD/weights/yolov5l_best.pt",
            'cp2': "./VD/weights/3d_Resnet_best.pth"
        }
        return path.abspath(hashtable.get(name, ''))

    @staticmethod
    def get_icon_adr(name):
        icon_adr = f'./icon/{name}.png'
        return path.abspath(icon_adr)
