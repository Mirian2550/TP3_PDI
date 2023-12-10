import multiprocessing


from video_process.video import VideoProcessor


def _process_video(video):
    video_processor = VideoProcessor(video)
    processed_frames = video_processor.process_video()

if __name__ == '__main__':
    videos = ["data/tirada_3.mp4"]
    with multiprocessing.Pool() as pool:
        pool.map(_process_video, videos)
