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
    diferencia = imagen2.astype(int) - imagen1.astype(int)
    diferencia_abs = np.abs(diferencia).astype(np.uint8)

    _, diferencia_binaria = cv2.threshold(diferencia_abs, 10, 255, cv2.THRESH_BINARY_INV)
    zonas_resaltadas = cv2.bitwise_and(imagen2, imagen2, mask=diferencia_binaria)

    return zonas_resaltadas

def seleccion_contornos(imagen):
    contornos, _ = cv2.findContours(imagen, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contornos_seleccionados = [
        contorno for contorno in contornos
        if 3000 < cv2.contourArea(contorno) < 8000
        and 0.6 < (4 * np.pi * cv2.contourArea(contorno) / (cv2.arcLength(contorno, True) ** 2))
    ]

    return contornos_seleccionados

def subimagen(imagen, contorno):
    mascara_contorno = np.zeros_like(imagen)
    cv2.drawContours(mascara_contorno, [contorno], -1, 255, thickness=cv2.FILLED)

    subimagen_contorno = np.zeros_like(imagen)
    subimagen_contorno[mascara_contorno == 255] = imagen[mascara_contorno == 255]

    x, y, w, h = cv2.boundingRect(contorno)
    subimagen_contorno_reducida = subimagen_contorno[y:y+h, x:x+w]

    return subimagen_contorno_reducida

def contar_puntos(contorno, imagen):
    subimagen_azul_reducida = subimagen(imagen, contorno)
    
    subimagen_bin = umbralizar_percentil(subimagen_azul_reducida, 95)
    
    contornos_subimagen_bin, _ = cv2.findContours(subimagen_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    contornos_filtrados_subimagen_bin = [
        contorno_subimagen for contorno_subimagen in contornos_subimagen_bin
        if cv2.arcLength(contorno_subimagen, True) != 0 
        and 0.8 < (4 * np.pi * cv2.contourArea(contorno_subimagen) / (cv2.arcLength(contorno_subimagen, True) ** 2))
        and cv2.contourArea(contorno_subimagen) > 50
    ]
    
    return contorno, len(contornos_filtrados_subimagen_bin)

def cambio_dado(dados, imagen):
    nuevos_dados = []

    for contorno, puntos_anteriores in dados:
        _, puntos_actuales = contar_puntos(contorno, imagen)

        if puntos_anteriores == puntos_actuales:
            nuevos_dados.append((contorno, puntos_actuales))

    return nuevos_dados

def contorno_similar(contorno1, contorno2, distancia_umbral=30):
    M1 = cv2.moments(contorno1)
    M2 = cv2.moments(contorno2)

    cx1 = int(M1['m10'] / M1['m00']) if M1['m00'] != 0 else 0
    cy1 = int(M1['m01'] / M1['m00']) if M1['m00'] != 0 else 0

    cx2 = int(M2['m10'] / M2['m00']) if M2['m00'] != 0 else 0
    cy2 = int(M2['m01'] / M2['m00']) if M2['m00'] != 0 else 0

    distancia_centroides = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)

    return distancia_centroides < distancia_umbral

def dibujar(datos, imagen, color):
    imagen_dibujada = imagen.copy()

    for contorno, valor in datos:
        cv2.drawContours(imagen_dibujada, [contorno], -1, color, 5)
        
        M = cv2.moments(contorno)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        cv2.putText(imagen_dibujada, str(valor), (cx + 35, cy - 40), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 5, cv2.LINE_AA)

    return imagen_dibujada





direccion_video = "data/tirada_3.mp4"
video_data = leer_video(direccion_video)

ruta_video_salida = "resultado.mp4"

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_salida = cv2.VideoWriter(ruta_video_salida, fourcc, 20, (video_data['width'], video_data['height']))


dados = []

for n in range(1, len(video_data['frames'])):
    framer = video_data['frames'][n-1][:, :, 2]
    frame = video_data['frames'][n]

    azul = frame[:, :, 0]
    rojo = frame[:, :, 2]

    zonas_resaltadas = diferencia(framer, rojo)

    rojo_binario = umbralizar_percentil(zonas_resaltadas)

    rojo_apertura = clausura(rojo_binario)

    rojo_sin_borde = eliminar_objetos_borde(rojo_apertura)

    contornos = seleccion_contornos(rojo_sin_borde)

    dados = cambio_dado(dados, azul)

    for contorno in contornos:
        nuevo = True

        for cont, _ in dados:
            if contorno_similar(cont, contorno):
                nuevo = False

        if nuevo:
            valor = contar_puntos(contorno, azul)
            dados.append(valor)
    
    imagen_final = dibujar(dados, frame, (0,0,0))

    video_salida.write(imagen_final)



video_salida.release()