def get_label_data(corr_windows):
    # 调整时间窗口 创建一个空列表，用于存储不重叠的时间窗
    merged_windows = []
    # 初始化起始时间和终止时间
    start = corr_windows[0][0]
    end = corr_windows[0][1]
    # 遍历每个时间窗
    for i in range(1, len(corr_windows)):
        curr_start = corr_windows[i][0]
        curr_end = corr_windows[i][1]
        # 模板匹配事件重叠处理
        # 如果时间窗重叠，则将其与已经添加到列表中的时间窗合并
        if curr_start <= end:
            end = max(curr_end, end)
        # 如果时间窗不重叠，则将其添加到列表中，并更新起始时间和终止时间
        else:
            merged_windows.append((start, end))
            start = curr_start
            end = curr_end
    # 添加最后一个时间窗
    merged_windows.append((start, end))
    return merged_windows

# 输出保存的相关窗口个数
# print(len(get_label_data()))
# print(get_label_data())
