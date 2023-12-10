import cv2
import numpy as np

def detectar_contornos(frame):
    # Convertir a escala de grises
    frame_gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Aplicar umbralización
    _, umbralizado = cv2.threshold(frame_gris, 200, 255, cv2.THRESH_BINARY)

    # Encontrar contornos en la imagen umbralizada
    contours, _ = cv2.findContours(umbralizado, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filtrar contornos por área
    contours_filtrados = [cnt for cnt in contours if 1 < cv2.contourArea(cnt) < 500000000]

    # Dibujar contornos en la imagen original
    frame_contornos = frame.copy()
    cv2.drawContours(frame_contornos, contours_filtrados, -1, (255, 0, 0), 2)

    # Aplicar operación de apertura
    kernel_apertura = np.ones((5, 5), np.uint8)
    frame_contornos = cv2.morphologyEx(frame_contornos, cv2.MORPH_OPEN, kernel_apertura)

    # Aplicar operación de cierre
    kernel_cierre = np.ones((10, 10), np.uint8)
    frame_contornos = cv2.morphologyEx(frame_contornos, cv2.MORPH_CLOSE, kernel_cierre)

    return frame_contornos


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
                diff_sum = np.sum(diff_thresholded)
                if diff_sum <= 600 and diff_sum > 100:  # Puedes ajustar este valor según sea necesario
                    print("Dado detenido. Mostrando el fotograma.")
                    """
                    cv2.imshow("Fotograma Detenido", frame)
                    print('suma', diff_sum)
                    cv2.waitKey(0)  # Esperar hasta que se presione una tecla
                    #break
                    """

                    frame_con_contornos = detectar_contornos(frame)

                    # Mostrar el frame con contornos
                    cv2.imshow("Contornos de Dados", frame_con_contornos)

                    cv2.waitKey(0)

            previous_frame = frame.copy()

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


