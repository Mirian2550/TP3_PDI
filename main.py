import os
import multiprocessing

from video_process.video import VideoProcessor

# Función para procesar un video individualmente
def _process_video(video):
    video_name = os.path.splitext(os.path.basename(video))[0]
    output_path = os.path.join("output", f"{video_name}_resultado.mp4")

    video_processor = VideoProcessor(video_path=video, output_path=output_path)
    video_processor.process_video()


if __name__ == '__main__':
    video_folder = "data"
    # Crear una lista de rutas de videos en la carpeta especificada
    videos = [os.path.join(video_folder, file) for file in os.listdir(video_folder) if file.endswith(".mp4")]
    
    # Utilizar multiprocessing para procesar los videos en paralelo
    with multiprocessing.Pool() as pool:
        # Mapear la función _process_video a cada video en la lista
        pool.map(_process_video, videos)