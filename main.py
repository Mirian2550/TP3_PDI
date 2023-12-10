import multiprocessing

from bounding_box_Drawer.boundingbox import BoundingBoxDrawer
from dice_detector.dice import DiceDetector
from numbe_recognizer.number import NumberRecognizer
from video_generator.video_generator import VideoGenerator
from video_process.video import VideoProcessor


def _process_video(video):
    video_processor = VideoProcessor(video)
    processed_frames = video_processor.process_video()
if __name__ == '__main__':
    # 1 humbral esta bien 1071951
    videos = ["data/tirada_4.mp4"]

    with multiprocessing.Pool() as pool:
        pool.map(_process_video, videos)
