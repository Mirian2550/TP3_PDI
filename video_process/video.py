import cv2
import numpy as np


def _filtro_sobel(imagen_entrada: np.ndarray, tamano_kernel: int = 3) -> np.ndarray:
    """
    Aplica el filtro Sobel en las direcciones x e y a la imagen dada.

    Parameters:
    - imagen_entrada (np.ndarray): La imagen a la cual se aplicará el filtro Sobel.
    - tamano_kernel (int, optional): Tamaño del kernel para el filtro Sobel. Por defecto es 3.

    Returns:
    - np.ndarray: La imagen resultante después de aplicar el filtro Sobel.
    """
    sobel_x = cv2.Sobel(imagen_entrada, cv2.CV_64F, 1, 0, ksize=tamano_kernel)
    sobel_y = cv2.Sobel(imagen_entrada, cv2.CV_64F, 0, 1, ksize=tamano_kernel)

    imagen_gradiente = cv2.convertScaleAbs(np.sqrt(sobel_x**2 + sobel_y**2))

    return imagen_gradiente


class VideoProcessor:
    def __init__(self, video):
        self.video = video

    def process_video(self):
        cap = cv2.VideoCapture(self.video)
        if not cap.isOpened():
            print("Error al abrir el video.")
            return None

        processed_frames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Aplicar un filtro: Convertir a escala de grises
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            imagen_suavizada = cv2.GaussianBlur(gray_frame, (15, 15), 0)

            imagen_gradiente = _filtro_sobel(imagen_suavizada)

            imagen_filtrada_umbral = cv2.GaussianBlur(imagen_gradiente, (19, 19), 0)

            imagen_gradiente_abs = cv2.convertScaleAbs(imagen_filtrada_umbral)

            umbral_superior = int(np.percentile(imagen_gradiente_abs, 90))

            _, imagen_umbralizada = cv2.threshold(imagen_gradiente_abs, umbral_superior, 1, cv2.THRESH_BINARY)

            imagen_expandida = cv2.morphologyEx(imagen_umbralizada, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))

            contornos, _ = cv2.findContours(imagen_expandida, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for i, contorno in enumerate(contornos):
                area_contorno = cv2.contourArea(contorno)
                if area_contorno >= 1000:
                    cv2.drawContours(imagen_umbralizada, [contorno], -1, 255, thickness=cv2.FILLED)

            # Agregar el fotograma procesado a la lista
            processed_frames.append(imagen_umbralizada)

            cv2.imshow(self.video, imagen_umbralizada)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        return processed_frames
