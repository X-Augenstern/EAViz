# from moviepy.editor import VideoFileClip


def cut_video(input_path, output_folder, start=0, dur=3):
    # 加载视频剪辑
    video = VideoFileClip(input_path)
    video_duration = video.duration

    clip_num = 0
    start_time = start
    end_time = start + dur

    while end_time <= video_duration:
        print(f"正在处理第{clip_num + 1}段视频...")
        # 剪辑视频
        clipped_video = video.subclip(start_time, end_time)

        # 保存剪辑后的视频
        output_path = f"{output_folder}/clip_{clip_num}.mp4"
        clipped_video.write_videofile(output_path, codec="libx264")

        clip_num += 1
        start_time += dur
        end_time += dur

    # 关闭视频剪辑
    video.close()


def resample():
    # 指定目标帧率
    target_fps = 20

    # 加载视频并修改帧率
    clip = VideoFileClip(input_video)
    clip_with_new_fps = clip.set_fps(target_fps)

    # 保存到文件
    clip_with_new_fps.write_videofile(output_video, codec="libx264", audio_codec="aac")


# 调用函数进行视频剪辑
# input_file = "../test_data/test_video.mp4"  # 输入视频文件名
# output_folder = "../test_data/clips"  # 输出视频文件名
# cut_video(input_file, output_folder)

# 调用函数改变视频帧率
input_video = "F:\\Python\\EAViz\\test_data\\VD\\Interictal.mp4"
output_video = "F:\\Python\\EAViz\\test_data\\VD\\new.mp4"
resample()
