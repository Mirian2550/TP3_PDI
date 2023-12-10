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

                # Convertir la diferencia a escala de grises
                diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

                # Aplicar umbralización a la diferencia
                _, diff_thresholded = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY)

                # Calcular la suma de los píxeles umbralizados
                diff_sum = np.sum(diff_thresholded)
                print(diff_sum)

                # Si la suma es menor que el umbral, mostrar el fotograma y detener el bucle
                # video 4 suma 510
                # video 3
                # video 1 suma 255
                # video 2 suma 510
                if diff_sum <= 600 and diff_sum > 100:  # Puedes ajustar este valor según sea necesario
                    print("Dado detenido. Mostrando el fotograma.")
                    cv2.imshow("Fotograma Detenido", frame)
                    print('suma', diff_sum)
                    cv2.waitKey(0)  # Esperar hasta que se presione una tecla
                    #break

            previous_frame = frame.copy()

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


