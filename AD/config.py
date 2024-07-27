class config:
    mean = [-0.019813645741049712, -0.08832978649691134, -0.17852094982207156, -0.141283147662929,
            -0.164364199798768, -0.10493702302254725, 0.0069039257850445224, 0.053706128833827776,
            -0.07108375609375886, -0.036934718124703704]
    std = [51.59277790417314, 55.41723123794839, 71.50782941925381, 67.58345657953764, 55.94475895317833,
           52.58134875167906, 28.773163340067427, 28.172769340175684, 25.5494790918186, 24.69622808553754]
    name_loss = 'BCELoss'
    num_class = 6
    num_workers = 2
    batch_size = 50

    epoch = 1000
    patience = 80
    learning_rate = 0.005
    wd = 1e-4
    momentum = 0.9
    lr_decay_rate = 0.1
    lr_decay_epochs = [20, 35, 45]

    savelog = 'logs'
    saveweight = 'weights'
    # save_root = './result' + self.model + self.name_loss

    train_txt = './datasplit/train.txt'
    val_txt = './datasplit/val.txt'
    test_txt = './datasplit/test.txt'

    def __init__(self):
        self.model = 'DenseNet121'
        self.loss = ['BCELoss']
        self.test_weight = None

    def get_test_weight(self):
        if self.model == 'Resnet34':
            if self.loss == 'BCELoss':
                self.test_weight = r'.\AD\resultResnet34BCELoss0\weights\save_9.pth'
            else:
                self.test_weight = r'.\AD\resultResnet34MSMLoss1\weights\save_8.pth'
        elif self.model == 'VGG16':
            if self.loss == 'BCELoss':
                self.test_weight = r'.\AD\resultVGG16BCELoss5\weights\save_12.pth'
            else:
                self.test_weight = r'.\AD\resultVGG16MSMLoss0\weights\save_7.pth'
        elif self.model == 'SENet18':
            if self.loss == 'BCELoss':
                self.test_weight = r'.\AD\resultSENet18BCELoss0\weights\save_85.pth'
            else:
                self.test_weight = r'.\AD\resultSENet18MSMLoss0\weights\save_25.pth'
        elif self.model == 'DenseNet121':
            if self.loss == 'BCELoss':
                self.test_weight = r'.\AD\resultDenseNet121BCELoss0\weights\save_21.pth'
            else:
                self.test_weight = r'.\AD\resultDenseNet121MSMLoss0\weights\save_22.pth'
