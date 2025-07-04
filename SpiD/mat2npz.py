from os import path, listdir
from scipy.io import loadmat
from numpy import arange, load, savez_compressed

duration = 15000  # 每个样本的长度
sample_rate = 500  # 采样频率
stage_dict = {'NREM1': 1, 'NREM2': 1, 'NREM3': 1, 'WAKE': 0, 'REM': 2}
fillnum = 5  # 在编码数字前自动补到5位，用0填充 (00001 00002 ...)


def check_spike_num(root):
    """
    查看当前npz文件夹下有多少个spike_label
    """
    cnt = 0
    for file in listdir(root):
        file_p = path.join(root, file)  # 将文件名与目录路径拼接为完整的文件路径
        spike_label = load(file_p, allow_pickle=True)['spike_label']  # 加载每个文件，并用字典解包获取spike_label的值
        cnt += len(spike_label)
    print(cnt)


def load_file_dict(mat_path):
    """
    遍历给定路径下所有.mat文件，返回{.mat相对路径 : data = .mat绝对路径, label = .txt绝对路径}
    """
    file_dict = {}
    txt_subname = '_annotations.txt'
    for file in listdir(mat_path):
        if file.endswith('.mat'):
            # print(dict(data=path.join(mat_path, file), label=path.join(mat_path, file[:-4]) + txt_subname))
            file_dict[file] = dict(data=path.join(mat_path, file), label=path.join(mat_path, file[:-4]) + txt_subname)
    return file_dict


def load_data_label(file_dict, key):
    """
    读取.mat以及get_labeled_dict()
    """
    file = file_dict[key]  # 获取对应的value
    mat = loadmat(file['data'])['data']  # 获取.mat中key为'data'对应的value (19,15000) 数据格式[1, 1000000] -> [1000000]
    label_file = file['label']  # .txt绝对地址
    with open(label_file, 'r') as f:
        txt_list = f.readlines()[1:]  # 读取.txt所有行并存储在list中，只取第二行及之后的
    labeled_dict, num = get_labeled_dict(txt_list)
    return mat, labeled_dict


def get_labeled_dict(txt_list):
    """
    去除没有棘波标签的txt行并构建Spike - dict: {NREM1: [Spike1, Spike2, ...]}
    """
    labeled_dict = {}
    for i in range(len(txt_list)):
        key = txt_list[i].strip().split(',')[-1]  # NREM1 / Spike
        if key in stage_dict.keys():
            if i + 1 != len(txt_list) and txt_list[i + 1].strip().split(',')[-1] in stage_dict.keys():
                continue
            else:
                cur = txt_list[i].strip()  # 去除两端空格
                labeled_dict[cur] = []  # 将下一label不为 NREM1 / NREM2 / NREM3 / WAKE / REM 的行作为key
        elif key == 'Spike':
            labeled_dict[cur].append(txt_list[i].strip())  # 将label为 Spike 的行添加进 NREM1 列表
    num = 0
    for v in labeled_dict.values():
        num += len(v)
    print(f'Total {num} rows are labeled!')
    assert check_num(txt_list) == num  # 检查Spike行数量是否与txt内数量一致
    return labeled_dict, num


def check_num(txt_list):
    num = 0
    for line in txt_list:
        if line.strip().split(',')[-1] == 'Spike':
            num += 1
    return num


def check_spike_range(spikelist, stage_start, stage_end):
    key_point = arange(stage_start, stage_end + 1, duration)[1:]  # [15000]
    cur = 0
    for spike in spikelist:
        spikesplit = spike.split(',')
        spike_s = int(float(spikesplit[0]) * sample_rate)  # Spike_onset
        spike_e = spike_s + int(float(spikesplit[1]) * sample_rate)  # Spike_end
        if spike_s >= key_point[cur]:
            cur += 1
        if cur == len(key_point) - 1:
            break
        if spike_s < key_point[cur] and spike_e > key_point[cur + 1]:
            print(spike)


def match_spike_range_to_30s_sample(spikelist, stage_s, stage_e):
    totalnum = (stage_e - stage_s) // duration  # 当前stage有几个30s
    cur = 0
    matched_dict = {}
    for i in range(totalnum):
        matched_dict[i] = []
        while cur < len(spikelist):
            cur_spike = spikelist[cur]  # 当前spike
            cur_split = cur_spike.split(',')
            cur_s = int(float(cur_split[0]) * sample_rate)  # Spike_onset
            cur_e = cur_s + int(float(cur_split[1]) * sample_rate)  # Spike_end
            if cur_s >= stage_s + (i + 1) * duration:  # Spike的开始时间 > stage的结束时间
                break
            cur_s = cur_s - i * duration - stage_s  # 相对开始时间
            cur_e = cur_e - i * duration - stage_s  # 相对结束时间
            matched_dict[i].append([cur_s, cur_e])  # {第i个30s: [Spike 的开始时间和结束时间]}
            cur += 1
    return matched_dict


def mat2npz(mat_path, npz_path):
    # check_spike_num(npz_path)
    file_dict = load_file_dict(mat_path)
    cnt = 0
    for file in file_dict.keys():  # .mat相对路径
        mat, labeled_dict = load_data_label(file_dict, file)
        patient_name = file[:-4]  # 'temp' 除去扩展名后的文件名 -> batch?
        num = 0
        for k, v in labeled_dict.items():
            stage_split = k.split(',')  # stage: [onset, dur, stage]
            stage_s = int(float(stage_split[0]) * sample_rate)  # stage_onset * 500
            stage_e = stage_s + int(float(stage_split[1]) * sample_rate)  # stage_end * 500
            stage_cla = stage_split[-1]  # stage
            check_spike_range(v, stage_s, stage_e)
            matched_dict = match_spike_range_to_30s_sample(v, stage_s, stage_e)
            for p in matched_dict.values():
                cnt += len(p)
            labeled_mat = mat[:, stage_s:stage_e]  # 获取到当前stage的数据
            assert (stage_e - stage_s) // duration == (stage_e - stage_s) / duration
            for matched_k, matched_v in matched_dict.items():  # 当前stage的第i个30s 及对应的spike
                num += 1
                # .npz 通常用于存储压缩的 np 数组
                save_path = path.join(npz_path, patient_name + '_' + str(num).zfill(fillnum) + '.npz')
                if not matched_v:
                    continue
                else:
                    savez_compressed(
                        save_path,
                        data=labeled_mat[:, matched_k * duration:(matched_k + 1) * duration],  # 获取当前stage第i个30s的数据
                        stage_label=stage_dict[stage_cla],  # 1
                        spike_label=matched_v  # [[onset1,dur1], [onset2,dur2], ...] len = 15 （会被转化为np数组压缩到npz中）
                    )  # 几个key就会在npz中生成对应的几个npy文件
    print(cnt)

# if __name__ == '__main__':
# path = "./mat"
# save_root = "./npzdata"
# duration = 15000  # 每个样本的长度
# stage_dict = {'NREM1': 1, 'NREM2': 1, 'NREM3': 1, 'WAKE': 0, 'REM': 2}
# sample_rate = 500  # 采样频率
# fillnum = 5  # 在编码数字前自动补到5位，用0填充
# check_spike_num(save_root)
# file_dict = Load_File_Dict(path)
# cnt = 0
# for file in file_dict.keys():
#     mat, label = Load_Data_Label(file_dict, file)
#     patient_name = file[:-4]
#     num = 0
#     for k, v in label.items():
#         stage_split = k.split(',')
#         stage_s = int(float(stage_split[0]) * sample_rate)
#         stage_e = stage_s + int(float(stage_split[1]) * sample_rate)
#         stage_cla = stage_split[-1]
#         check_spike_range(v, stage_s, stage_e)
#         matched_dict = match_spike_range_to_30s_sample(v, stage_s, stage_e)
#         for k, v in matched_dict.items():
#             cnt += len(v)
#         labeled_mat = mat[:, stage_s:stage_e]
#         assert (stage_e - stage_s) // duration == (stage_e - stage_s) / duration
#         for matched_k, matched_v in matched_dict.items():
#             num += 1
#             save_path = os.path.join(save_root, patient_name + '_' + str(num).zfill(fillnum) + '.npz')
#             if not matched_v:
#                 continue
#             else:
#                 np.savez_compressed(
#                     save_path,
#                     data=labeled_mat[:, matched_k*duration:(matched_k+1) * duration],
#                     stage_label=stage_dict[stage_cla],
#                     spike_label=matched_v
#                 )
# print(cnt)
