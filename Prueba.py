import cv2
import numpy as np
import matplotlib.pyplot as plt

def leer_video(direccion):
    cap = cv2.VideoCapture(direccion)

    if not cap.isOpened():
        raise ValueError("No se pudo abrir el archivo de video")

    ret, frame = cap.read()

    if not ret:
        raise ValueError("No se pudo leer el primer fotograma del video")

    height, width, _ = frame.shape

    video_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        video_frames.append(frame)

    cap.release()

    return {
        'width': width,
        'height': height,
        'frames': video_frames
    }

def imshow(img: np.ndarray, title = None, color_img: bool = False, blocking: bool = True) -> None:
    """
    Muestra una imagen utilizando Matplotlib.

    Parameters:
    - img (np.ndarray): La imagen que se mostrará.
    - title (str, optional): El título de la imagen. Por defecto es None.
    - color_img (bool, optional): Indica si la imagen es a color. Por defecto es False.
    - blocking (bool, optional): Indica si la ejecución del programa se bloquea hasta que se cierra la ventana de la imagen. Por defecto es True.
    """
    plt.figure()
    if color_img:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.show(block=blocking)

def umbralizar_percentil(imagen, percentil=70):
    umbral = np.percentile(imagen, percentil)
    _, imagen_binaria = cv2.threshold(imagen, umbral, 255, cv2.THRESH_BINARY)
    return imagen_binaria

def clausura(imagen_binaria, radio_kernel=4):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radio_kernel, 2 * radio_kernel))
    resultado_clausura = cv2.morphologyEx(imagen_binaria, cv2.MORPH_CLOSE, kernel)
    return resultado_clausura

def eliminar_objetos_borde(imagen_binaria):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(imagen_binaria)

    borde_highlight = np.zeros_like(imagen_binaria)

    for i in range(1, num_labels):
        if (stats[i, cv2.CC_STAT_LEFT] == 0 or
            stats[i, cv2.CC_STAT_TOP] == 0 or
            stats[i, cv2.CC_STAT_HEIGHT] + stats[i, cv2.CC_STAT_TOP] == imagen_binaria.shape[0] or
            stats[i, cv2.CC_STAT_WIDTH] + stats[i, cv2.CC_STAT_LEFT] == imagen_binaria.shape[1]):

            borde_highlight[labels == i] = 255

    imagen_sin_borde = imagen_binaria.copy()
    imagen_sin_borde[borde_highlight == 255] = 0

    return imagen_sin_borde

def diferencia(imagen1, imagen2):
    # Resta los pixeles entre dos imágenes en escala de grises
    diferencia = imagen2.astype(int) - imagen1.astype(int)

    # Calcula el valor absoluto de la diferencia
    diferencia_abs = np.abs(diferencia).astype(np.uint8)

    # Umbraliza la diferencia
    _, diferencia_binaria = cv2.threshold(diferencia_abs, 10, 255, cv2.THRESH_BINARY_INV)

    # Aplica la máscara para resaltar las zonas de diferencia
    zonas_resaltadas = cv2.bitwise_and(imagen2, imagen2, mask=diferencia_binaria)

    return zonas_resaltadas





direccion_video = "data/tirada_3.mp4"
video_data = leer_video(direccion_video)

ruta_video_salida = "resultado.mp4"

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_salida = cv2.VideoWriter(ruta_video_salida, fourcc, 20, (video_data['width'], video_data['height']))

for n in range(1, len(video_data['frames'])):
    framer = video_data['frames'][n-1][:, :, 2]
    frame = video_data['frames'][n]

    azul = frame[:, :, 0]
    rojo = frame[:, :, 2]

    zonas_resaltadas = diferencia(framer, rojo)

    rojo_binario = umbralizar_percentil(zonas_resaltadas)

    rojo_apertura = clausura(rojo_binario)

    rojo_sin_borde = eliminar_objetos_borde(rojo_apertura)

    contornos, _ = cv2.findContours(rojo_sin_borde, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contornos_filtrados = [contorno for contorno in contornos if 3000 < cv2.contourArea(contorno) < 7000]

    imagen_contornos_filtrados = frame.copy()
    cv2.drawContours(imagen_contornos_filtrados, contornos_filtrados, -1, (0, 0, 0), 5)

    video_salida.write(imagen_contornos_filtrados)



video_salida.release()