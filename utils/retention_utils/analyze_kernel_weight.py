# ESC

# import torch
# from ESA.A3D_model import R3DClassifier
# import matplotlib.pyplot as plt
# import numpy as np
#
# model = R3DClassifier(num_classes=7, layer_sizes=(2, 2, 2, 2))
# checkpoint = torch.load('A3D-EEG_epoch-1.pth.tar', map_location=lambda storage, loc: storage)
# model.load_state_dict(checkpoint['state_dict'])
# print(model)
#
# weights_keys = model.state_dict().keys()  # 获取模型中所有具有可训练参数的层的名称
# for key in weights_keys:
#     if "num_batches_tracked" in key:
#         continue
#     # [kernel_number 卷积核个数/输出特征矩阵的深度, kernel_channel 卷积核深度/输入特征矩阵的深度, kernel_height卷积核高度, kernel_width]
#     # [64,1,3,7,7]
#     weight_t = model.state_dict()[key].numpy()  # 获取各层的参数信息
#
#     # # 读取第一个卷积核的信息
#     # k = weight_t[0, :, :, :, :]
#
#     weight_mean = weight_t.mean()
#     weight_std = weight_t.std(ddof=1)
#     weight_min = weight_t.min()
#     weight_max = weight_t.max()
#     print("mean is {}, std is {}, min is {}, max is {}.".format(weight_mean, weight_std, weight_min, weight_max))
#
#     # plot hist image
#     plt.close()
#     weight_vec = np.reshape(weight_t, [-1])  # 将卷积核权重reshape至一维
#     plt.hist(weight_vec, bins=50)  # min-max均分为50等份
#     plt.title(key)
#     plt.show()
