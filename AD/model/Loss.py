from torch.nn import MultiLabelSoftMarginLoss


# class classLoss1:
#     def __init__(self):
#         self.loss = BCEWithLogitsLoss(weight=None, reduction='mean', pos_weight=None)
#
#     def __call__(self, pred, label):
#         BCEloss = self.loss(pred, label)
#         loss_dict = dict(BCELoss=BCEloss, Total_Loss=BCEloss)
#         return loss_dict
#
#
# class classLoss2:
#     def __init__(self):
#         self.loss = MultiLabelMarginLoss(reduction='mean')
#
#     def __call__(self, pred, label):
#         MuLLoss = self.loss(pred, label.long())
#         loss_dict = dict(MuLloss=MuLLoss, Total_Loss=MuLLoss)
#         return loss_dict


class classLoss:
    def __init__(self):
        self.loss = MultiLabelSoftMarginLoss(weight=None, reduction='mean')

    def __call__(self, pred, label):
        MSMLoss = self.loss(pred, label)
        loss_dict = dict(MSMLoss=MSMLoss, Total_Loss=MSMLoss)
        return loss_dict


'''
# 二进制交叉熵损失
binary_crossentropy_loss = tf.keras.losses.BinaryCrossentropy()
binary_loss = binary_crossentropy_loss(true_labels, predicted_probs)
print("Binary Cross-Entropy Loss:", binary_loss.numpy())

# 交叉熵损失
categorical_crossentropy_loss = tf.keras.losses.CategoricalCrossentropy()
categorical_loss = categorical_crossentropy_loss(true_labels, predicted_probs)
print("Categorical Cross-Entropy Loss:", categorical_loss.numpy())

# 焦点损失
focal_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
gamma = 2.0  # 调节焦点损失的重要性
alpha = 0.25  # 调节难易样本的权重
focal_loss_value = tf.reduce_mean(
    alpha * tf.pow(1.0 - predicted_probs, gamma) * categorical_crossentropy_loss(true_labels, predicted_probs)
)
print("Focal Loss:", focal_loss_value.numpy())

# Dice 损失
epsilon = 1e-7
dice_loss = 1 - (2 * tf.reduce_sum(predicted_probs * true_labels) + epsilon) / (tf.reduce_sum(predicted_probs) + tf.reduce_sum(true_labels) + epsilon)
print("Dice Loss:", dice_loss.numpy())

# 平均绝对误差损失
mean_absolute_error_loss = tf.keras.losses.MeanAbsoluteError()
mae_loss = mean_absolute_error_loss(true_labels, predicted_probs)
print("Mean Absolute Error Loss:", mae_loss.numpy())

# 平均平方误差损失
mean_squared_error_loss = tf.keras.losses.MeanSquaredError()
mse_loss = mean_squared_error_loss(true_labels, predicted_probs)
print("Mean Squared Error Loss:", mse_loss.numpy())
'''

# if __name__ == '__main__':
#     var = torch.rand(8, 10, 1000)
#
#     print(var.shape)
