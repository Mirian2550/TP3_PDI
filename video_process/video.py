import cv2
import numpy as np

class VideoProcessor:
    def __init__(self, video):
        self.video = video

    def process_video(self):
        cap = cv2.VideoCapture(self.video)
        if not cap.isOpened():
            print("Error al abrir el video.")
            return None

        previous_frame = None
        stopping_threshold = 1  # Ajusta este valor según sea necesario

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Eliminar canal verde
            frame_sin_verde = frame.copy()
            frame_sin_verde[:, :, 1] = 0

            # Segmentar dados utilizando canal rojo
            canal_rojo = frame[:, :, 2]
            _, dados_segmentados = cv2.threshold(canal_rojo, 200, 255, cv2.THRESH_BINARY)

            # Contar puntos utilizando canal azul
            canal_azul = frame[:, :, 0]
            _, puntos_contados = cv2.threshold(canal_azul, 200, 255, cv2.THRESH_BINARY)

            # Mostrar las imágenes procesadas
            cv2.imshow("Frame sin verde", frame_sin_verde)
            cv2.imshow("Dados Segmentados", dados_segmentados)
            cv2.imshow("Puntos Contados", puntos_contados)

            if previous_frame is not None:
                # Calcular la diferencia entre el fotograma actual y el anterior
                diff = cv2.absdiff(frame, previous_frame)
                diff_sum = np.sum(diff)
                print(diff_sum)
                # Si la diferencia es menor que el umbral, mostrar el fotograma y detener el bucle
                if diff_sum < stopping_threshold:
                    print("Dado detenido. Mostrando el fotograma.")
                    cv2.imshow("Fotograma Detenido", frame)
                    cv2.waitKey(0)  # Esperar hasta que se presione una tecla
                    break

            previous_frame = frame.copy()

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()