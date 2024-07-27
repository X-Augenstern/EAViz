from moviepy.editor import VideoFileClip


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


# 调用函数进行视频剪辑
input_file = "../test_data/test_video.mp4"  # 输入视频文件名
output_folder = "../test_data/clips"  # 输出视频文件名
cut_video(input_file, output_folder)
