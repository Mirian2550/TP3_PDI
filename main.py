import multiprocessing

from bounding_box_Drawer.boundingbox import BoundingBoxDrawer
from dice_detector.dice import DiceDetector
from numbe_recognizer.number import NumberRecognizer
from video_generator.video_generator import VideoGenerator
from video_process.video import VideoProcessor


def _process_video(video):
    video_processor = VideoProcessor(video)
    processed_frames = video_processor.process_video()
    """
    dice_detector = DiceDetector()
    dice_positions = dice_detector.detect_dice(processed_frames)

    number_recognizer = NumberRecognizer()
    recognized_numbers = number_recognizer.recognize_numbers(processed_frames, dice_positions)

    bounding_box_drawer = BoundingBoxDrawer()
    annotated_frames = bounding_box_drawer.draw_boxes(processed_frames, dice_positions, recognized_numbers)

    video_generator = VideoGenerator()
    output_video = video_generator.generate_video(annotated_frames, video)
    # output_video.show()
    """
if __name__ == '__main__':
    # 1 humbral esta bien 1071951
    videos = ["data/tirada_2.mp4"]

    with multiprocessing.Pool() as pool:
        pool.map(_process_video, videos)
