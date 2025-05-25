from enum import Enum
from os import path
from matplotlib.colors import TABLEAU_COLORS


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
    HFO_model = ['', 'MKCNN']
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


class ChannelEnum(Enum):
    """
    .name 获取枚举成员的名字，通过 .value 获取枚举成员的具体值
    """
    TPM = ['Fp1', 'F7', 'F3', 'T3', 'C3', 'T5', 'P3', 'Pz', 'O1', 'O2', 'P4', 'T6', 'C4', 'T4',
           'F4', 'F8', 'Fp2', 'Fz', 'Cz']  # 19
    CH21 = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7',
            'F8', 'T3', 'T4', 'T5', 'T6', 'A1', 'A2', 'Fz', 'Cz', 'Pz']  # 21 + A1、A2
    CH19 = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4',
            'T5', 'T6', 'Fz', 'Cz', 'Pz']  # 19
    COLLECTED = ['Fp1', 'F7', 'F3', 'T3', 'C3', 'T5', 'P3', 'O1', 'Fp2', 'F8', 'F4', 'T4', 'C4', 'T6', 'P4',
                 'O2', 'Fz', 'Cz', 'Pz', 'A1', 'A2']  # 21


class PSDEnum(Enum):
    COLORS = list(TABLEAU_COLORS.values())  # 获取颜色列表
    FREQ_LABELS = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
    FREQ_BANDS = [(0, 4), (4, 8), (8, 12), (12, 30), (30, 45)]


class MontageEnum(Enum):
    XPOS = [-0.0293387312092767, -0.0768097954926838, -0.051775707348028, -0.0949421285668141, -0.0683372810321719,
            -0.0768097954926838, -0.051775707348028,
            4.18445162392412E-18, -0.0293387312092767, 0.0293387312092767, 0.051775707348028, 0.0768097954926838,
            0.0683372810321719, 0.0949421285668141,
            0.051775707348028, 0.0768097954926838, 0.0293387312092767, 4.18445162392412E-18, 0]

    YPOS = [0.0902953300444008, 0.0558055829928292, 0.0639376737816708, 0, 0, -0.0558055829928292, -0.0639376737816708,
            -0.0683372810321719,
            -0.0902953300444008, -0.0902953300444008, -0.0639376737816708, -0.0558055829928292, 0, 0,
            0.0639376737816708,
            0.0558055829928292, 0.0902953300444008, 0.0683372810321719, 0]

    ZPOS = [-0.00331545218673759, -0.00331545218673759, 0.0475, -0.00331545218673759, 0.0659925451936047,
            -0.00331545218673759, 0.0475, 0.0659925451936047,
            -0.00331545218673759, -0.00331545218673759, 0.0475, -0.00331545218673759, 0.0659925451936047,
            -0.00331545218673759, 0.0475, -0.00331545218673759,
            -0.00331545218673759, 0.0659925451936047, 0.095]

    BRAIN_REGIONS = {
        "中央区": ["C3", "C4", "Cz"],
        "枕区": ["T5", "T6", "O1", "O2"],
        "顶区": ["P3", "P4", "Pz"],
        "颞区": ["F7", "F8", "T3", "T4"],
        "额区": ["Fp1", "Fp2", "F3", "F4", "Fz"]
    }


class ThemeColorConfig:
    theme = "dark"

    class ThemeColorEnum(Enum):
        DARK_UI_SS = """
                         QWidget#Form {
                             background-color: qlineargradient(x0:0, y0:0, x1:1, y1:1,
                                                               stop:0 rgb(20, 32, 44),
                                                               stop:1 rgb(37, 85, 117));
                         }
                     """
        DARK_EEG_BG = '#79a4c8'
        DARK_EAI_BG = '#19232d'
        DARK_SRD_THEME = {
            "background": "#204660",  # 黑色背景
            "title_color": "white",  # 标题颜色
            "axis_color": "white",  # 轴线和刻度颜色
            "line_color": "#29f1ff",  # 数据线颜色
            "highlight_color": (70, 130, 180, 100),  # 区域高亮颜色 (RGBA)
            "annotation_color": "yellow",  # 注释文字颜色
            "layout_border_color": "#204660"
        }

        LIGHT_UI_SS = """
                          QWidget#Form {
                              background-color: qlineargradient(x0:0, y0:0, x1:1, y1:1,
                                                                stop:0 rgb(255, 255, 255),
                                                                stop:1 rgb(240, 240, 240));
                          }
                      """
        LIGHT_EEG_BG = '#fffefffe'
        LIGHT_EAI_BG = '#fafafa'
        LIGHT_SRD_THEME = {
            "background": "white",  # 白色背景
            "title_color": "black",  # 标题颜色
            "axis_color": "black",  # 轴线和刻度颜色
            "line_color": "royalblue",  # 数据线颜色
            "highlight_color": (135, 206, 250, 50),  # 区域高亮颜色 (RGBA)
            "annotation_color": "red",  # 注释文字颜色
            "layout_border_color": "white"
        }

    @classmethod
    def get_ui_ss(cls):
        return cls.ThemeColorEnum.DARK_UI_SS.value if cls.theme == "dark" else cls.ThemeColorEnum.LIGHT_UI_SS.value

    @classmethod
    def get_eeg_bg(cls):
        return cls.ThemeColorEnum.DARK_EEG_BG.value if cls.theme == "dark" else cls.ThemeColorEnum.LIGHT_EEG_BG.value

    @classmethod
    def get_eai_bg(cls):
        return cls.ThemeColorEnum.DARK_EAI_BG.value if cls.theme == "dark" else cls.ThemeColorEnum.LIGHT_EAI_BG.value

    @classmethod
    def get_srd_theme(cls):
        return cls.ThemeColorEnum.DARK_SRD_THEME.value if cls.theme == "dark" else cls.ThemeColorEnum.LIGHT_SRD_THEME.value

    @classmethod
    def get_txt_color(cls):
        return '#FFFFFF' if cls.theme == "dark" else '#000000'


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
            'idx': f"./AD/offline_analyse/index/{model_name}_{ThemeColorConfig.theme}.html",
            'topo': "./AD/offline_analyse/topomap/topo.png",
            'res': "./AD/offline_analyse/result/res.png",
        }
        return path.abspath(hashtable.get(name, ''))

    @staticmethod
    def get_sd_adr(name):
        hashtable = {
            'idx': f"./SD/offline_analyse/index/idx_{ThemeColorConfig.theme}.html",
            'fam': f"./SD/offline_analyse/family/fam_{ThemeColorConfig.theme}.png",
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
    def get_icon_adr(name, category=None):
        icon_adr = f'./icon/{name}.ico' if category == 'icon' else f'./icon/{name}.png'
        return path.abspath(icon_adr)
