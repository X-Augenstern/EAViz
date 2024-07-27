class MyConfig:
    leakrelu = True
    Kfold = 1
    epoch = 300
    learning_rate = 0.001
    wd = 1e-4
    momentum = 0.9
    lr_decay_rate = 0.1
    lr_decay_epochs = [50, 100, 150, 200]
    logs = True
    with_last_relu = False
    dataset = 'HDU'
    testbatch_size = 1
    use_feat = False
    sample = 'ch10_1s'
    batch_size = 50
    patience = 100

    # save_root = './result_' + sample + self.model + '_' + dataset
    savelog = 'logs'
    saveweight = 'weights'

    if dataset == 'HDU':
        # 原有数据
        traintxt = './data_split/train.txt'
        valtxt = './data_split/val.txt'
        testtxt = './data_split/test.txt'

        # 补充数据
        add_traintxt = './Norm_Data_split/train.txt'
        add_valtxt = './Norm_Data_split/val.txt'
        # add_testtxt = './NOrm_Data_split/test.txt'

        # 总体数据
        test_abnorm = './data_split/test_abnorm.txt'

        # 单类分析
        test_Eye = './sigle_type/Eye.txt'
        test_Eye_Front = './sigle_type/Eye_Front.txt'
        test_Eye_Temp = './sigle_type/Eye_Temp.txt'
        test_Eye_Muscle = './sigle_type/Eye_Muscle.txt'
        test_Front = './sigle_type/Front.txt'
        test_Front_Temp = './sigle_type/Front_Temp.txt'
        test_Chew = './sigle_type/Chew.txt'
        test_Temp = './sigle_type/Temp.txt'
        test_Unknow = './sigle_type/Unknow.txt'

        mean = [-0.00786801526801073, -0.00278565112623452, -0.05482746121969513,
                -0.027030028693491265, -0.09992397242957872, -0.05103782905744617,
                0.023541660959840466, -0.06301732366665798, -0.02007156729695995,
                0.0167145799331698]
        std = [141.00469540729193, 140.11667254034768, 140.078467098774,
               140.71135024918598, 140.38892508374005, 145.18014397337947,
               142.27858667910175, 140.63237143922595, 142.12897209932004,
               139.74905713374832]

    def __init__(self):
        self.model = 'AE'
        self.loss = None
        self.testweight = None

    def get_test_weight(self):
        dataset = 'HDU'
        if self.model == 'AE':
            self.testweight = r'.\AD\result_ch10_1sAE_HDU1\weights\save_124.pth'
            self.loss = ['Recon_Loss']
            # loss = ['Recon_Loss', 'Corr_Loss']

        elif self.model == 'SkipAE':
            if dataset == 'CHB':
                self.testweight = '/home/fang_yuan/PycharmProjects/long/AE/result_SkipAE_CHB0/weights/save_153.pth'
            elif dataset == 'HDU':
                self.testweight = r'.\AD\result_ch10_1sSkipAE_HDU1\weights\save_110.pth'
            self.loss = ['Recon_Loss']

        elif self.model == 'MemAE':
            if dataset == 'CHB':
                self.testweight = '/home/fang_yuan/PycharmProjects/long/AE/result_MemAE_CHB0/weights/save_219.pth'
            elif dataset == 'HDU':
                self.testweight = r'.\AD\result_ch10_1sMemAE_HDU0\weights\save_242.pth'
            self.loss = ['Recon_Loss', 'Entropy_Loss', 'Align_Loss']

        elif self.model == 'VAE':
            if dataset == 'HDU':
                self.testweight = r'.\AD\result_ch10_1sVAE_HDU3\weights\save_259.pth'
                self.loss = ['Recon_Loss', 'KL_Loss']
