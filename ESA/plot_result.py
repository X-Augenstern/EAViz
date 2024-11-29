from numpy import float32, concatenate, array
from torch.nn.functional import softmax
import matplotlib.pyplot as plt
from io import BytesIO
from utils.config import ThemeColorConfig


def plot_esa_res(data, res_signal):
    # 1
    # class_label = ['BECT', 'CAE', 'CSWS', 'EIEE', 'FS', 'Normal', 'WEST']
    # 19
    class_label = ['BECT', 'CAE', 'CSWS', 'EIEE', 'Else', 'FS', 'Normal', 'WEST']
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_facecolor(ThemeColorConfig.get_eai_bg())  # 坐标区域背景
    _plot_probs([data], class_label)

    # idx = np.argmax(input.cpu().data.numpy())  # data:(1,7) 获取最大概率值索引

    plt.tight_layout()
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=300)
    buffer.seek(0)
    res_signal.emit(buffer.getvalue())
    plt.close()


def plot_seid_res(data1, data2, res_signal):
    class_label = ['EIEE', 'WEST', 'CAE', 'FS+', 'BECT', 'CSWS', 'interictal', 'seizure']
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_facecolor(ThemeColorConfig.get_eai_bg())  # 坐标区域背景
    _plot_probs([data1, data2], class_label, 5)

    plt.tight_layout()
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=300)
    buffer.seek(0)
    res_signal.emit(buffer.getvalue())
    plt.close()


def _plot_probs(data, class_labels, v_idx=None):
    all_probs = array([])
    colors = plt.colormaps['tab10'](range(len(class_labels)))

    for i in range(len(data)):
        all_probs = concatenate((all_probs, softmax(data[i], dim=1).detach().numpy()[0]))  # softmax

    plt.bar(class_labels, all_probs * 100, align='center', color=colors, alpha=0.8)  # 直接将class_label作为x
    result = [("%.2f" % i) for i in all_probs * 100]
    for a, b in zip(class_labels, float32(result)):
        # x:a y:b+2 text:b
        plt.text(a, b + 2, b, ha='center', va='bottom', fontproperties="Arial", fontsize=10, fontweight='bold')

    # 设置分隔线
    if v_idx is not None:
        xticks_positions = plt.xticks()[0]
        mid = (xticks_positions[v_idx] + xticks_positions[v_idx + 1]) / 2
        plt.axvline(mid, color='grey', linestyle='--', linewidth=2)

    plt.ylabel('Probability(%)', fontproperties="Microsoft YaHei", fontsize=12, fontweight='bold')
    # plt.title('概率分布图', fontproperties="Microsoft YaHei", loc='left', fontsize=12, fontweight='bold')
    plt.xticks(fontproperties="Arial", fontsize=11, fontweight='bold')
    plt.yticks(fontproperties="Arial", fontsize=12, fontweight='bold')
    plt.ylim([0, 100])
