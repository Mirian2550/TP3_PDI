import cv2
import numpy as np
import logging
import os

class VideoProcessor:
    def __init__(self, video_path, output_path="resultado.mp4"):
        """
        Inicializa la clase VideoProcessor.

        Parameters:
        - video_path (str): Ruta del archivo de video.
        - output_path (str): Ruta del archivo de video de salida. Por defecto es "resultado.mp4".
        """
        self.video_path = video_path
        self.output_path = output_path
        self.dados = []
        self.logger = logging.getLogger(__name__)

    def _umbralizar_percentil(self, imagen, percentil=70):
        umbral = np.percentile(imagen, percentil)
        _, imagen_binaria = cv2.threshold(imagen, umbral, 255, cv2.THRESH_BINARY)
        return imagen_binaria

    def _clausura(self, imagen_binaria, radio_kernel=4):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radio_kernel, 2 * radio_kernel))
        resultado_clausura = cv2.morphologyEx(imagen_binaria, cv2.MORPH_CLOSE, kernel)
        return resultado_clausura

    def _eliminar_objetos_borde(self, imagen_binaria):
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

    def _diferencia(self, imagen1, imagen2):
        diferencia = imagen2.astype(int) - imagen1.astype(int)
        diferencia_abs = np.abs(diferencia).astype(np.uint8)

        _, diferencia_binaria = cv2.threshold(diferencia_abs, 10, 255, cv2.THRESH_BINARY_INV)
        zonas_resaltadas = cv2.bitwise_and(imagen2, imagen2, mask=diferencia_binaria)

        return zonas_resaltadas

    def _seleccion_contornos(self, imagen):
        contornos, _ = cv2.findContours(imagen, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contornos_seleccionados = [
            contorno for contorno in contornos
            if 3000 < cv2.contourArea(contorno) < 8000
               and 0.6 < (4 * np.pi * cv2.contourArea(contorno) / (cv2.arcLength(contorno, True) ** 2))
        ]

        return contornos_seleccionados

    def _subimagen(self, imagen, contorno):
        mascara_contorno = np.zeros_like(imagen)
        cv2.drawContours(mascara_contorno, [contorno], -1, 255, thickness=cv2.FILLED)

        subimagen_contorno = np.zeros_like(imagen)
        subimagen_contorno[mascara_contorno == 255] = imagen[mascara_contorno == 255]

        x, y, w, h = cv2.boundingRect(contorno)
        subimagen_contorno_reducida = subimagen_contorno[y:y + h, x:x + w]

        return subimagen_contorno_reducida

    def _contar_puntos(self, contorno, imagen):
        subimagen_azul_reducida = self._subimagen(imagen, contorno)
        subimagen_bin = self._umbralizar_percentil(subimagen_azul_reducida, 95)
        contornos_subimagen_bin, _ = cv2.findContours(subimagen_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contornos_filtrados_subimagen_bin = [
            contorno_subimagen for contorno_subimagen in contornos_subimagen_bin
            if cv2.arcLength(contorno_subimagen, True) != 0
               and 0.8 < (4 * np.pi * cv2.contourArea(contorno_subimagen) / (
                    cv2.arcLength(contorno_subimagen, True) ** 2))
               and cv2.contourArea(contorno_subimagen) > 50
        ]
        return contorno, len(contornos_filtrados_subimagen_bin)

    def _cambio_dado(self, imagen):
        nuevos_dados = []

        for contorno, puntos_anteriores in self.dados:
            _, puntos_actuales = self._contar_puntos(contorno, imagen)

            if puntos_anteriores == puntos_actuales:
                nuevos_dados.append((contorno, puntos_actuales))

        self.dados = nuevos_dados

    def _contorno_similar(self, contorno1, contorno2, distancia_umbral=30):
        M1 = cv2.moments(contorno1)
        M2 = cv2.moments(contorno2)

        cx1 = int(M1['m10'] / M1['m00']) if M1['m00'] != 0 else 0
        cy1 = int(M1['m01'] / M1['m00']) if M1['m00'] != 0 else 0

        cx2 = int(M2['m10'] / M2['m00']) if M2['m00'] != 0 else 0
        cy2 = int(M2['m01'] / M2['m00']) if M2['m00'] != 0 else 0

        distancia_centroides = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)

        return distancia_centroides < distancia_umbral

    def _dibujar(self, imagen, color):
        imagen_dibujada = imagen.copy()

        for contorno, valor in self.dados:
            cv2.drawContours(imagen_dibujada, [contorno], -1, color, 5)

            M = cv2.moments(contorno)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            cv2.putText(imagen_dibujada, str(valor), (cx + 35, cy - 40), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 5,
                        cv2.LINE_AA)

        return imagen_dibujada

    def _procesar_video(self):
        cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            raise ValueError("No se pudo abrir el archivo de video")

        ret, frame = cap.read()

        if not ret:
            raise ValueError("No se pudo leer el primer fotograma del video")

        height, width, _ = frame.shape

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_salida = cv2.VideoWriter(self.output_path, fourcc, 20, (width, height))

        for n in range(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
            try:
                framer = frame[:, :, 2]
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                azul = frame[:, :, 2]
                rojo = frame[:, :, 0]

                zonas_resaltadas = self._diferencia(framer, rojo)

                rojo_binario = self._umbralizar_percentil(zonas_resaltadas)

                rojo_apertura = self._clausura(rojo_binario)

                rojo_sin_borde = self._eliminar_objetos_borde(rojo_apertura)

                contornos = self._seleccion_contornos(rojo_sin_borde)

                self._cambio_dado(azul)

                for contorno in contornos:
                    nuevo = True

                    for cont, _ in self.dados:
                        if self._contorno_similar(cont, contorno):
                            nuevo = False

                    if nuevo:
                        valor = self._contar_puntos(contorno, azul)
                        self.dados.append(valor)

                imagen_final = self._dibujar(frame, (0, 0, 0))

                video_salida.write(cv2.cvtColor(imagen_final, cv2.COLOR_RGB2BGR))

                ret, frame = cap.read()

            except Exception as e:
                self.logger.error(f"Error en el procesamiento del fotograma {n}: {e}")

        cap.release()
        video_salida.release()

    def process_video(self):
        """
        Procesa el video y guarda el resultado en el archivo de salida.
        """
        try:
            self._procesar_video()
        except Exception as e:
            self.logger.error(f"Error al procesar el video: {e}")