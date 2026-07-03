#!/usr/bin/env python3
import argparse
from email.mime import message
import os
import time
from pathlib import Path

import cv2
import numpy as np

try:
	from reedsolo import RSCodec, ReedSolomonError
except ImportError:
	RSCodec = None
	ReedSolomonError = Exception

# ══════════════════════════════════════════════════════════════════════
# PARÁMETROS CONSERVADOS DEL TRANSMISOR
# ══════════════════════════════════════════════════════════════════════

TAM_CELDA = 32
CELDAS = 48
PANTALLA_W = TAM_CELDA * CELDAS
PANTALLA_H = TAM_CELDA * CELDAS

TAM_MARCADOR = 7
QUIET_ZONE = 1
TAM_ESQUINA = TAM_MARCADOR + QUIET_ZONE

CELDAS_TOTALES = CELDAS ** 2
CELDAS_RESERV = 4 * (TAM_ESQUINA ** 2)
CELDAS_DATOS = CELDAS_TOTALES - CELDAS_RESERV

ARUCO_DICT_ID = cv2.aruco.DICT_5X5_100
ARUCO_IDS = {
	"TL": 0,
	"TR": 1,
	"BL": 2
}

BYTES_FRAME = 255
RS_REDUNDANCIA = 76
RS_DATOS = BYTES_FRAME - RS_REDUNDANCIA
BITS_PAQUETE = BYTES_FRAME * 8

# ══════════════════════════════════════════════════════════════════════
# PROTOCOLO DE TRAMAS
# ══════════════════════════════════════════════════════════════════════

MAGIC = b"PDS"
VERSION = 1
TIPO_PREAMBLE = 1
TIPO_DATA = 2
TIPO_END = 3

TIPOS_NOMBRE = {
	TIPO_PREAMBLE: "PREAMBLE",
	TIPO_DATA: "DATA",
	TIPO_END: "END"
}

HEADER_SIZE = 16
PAYLOAD_POR_FRAME = RS_DATOS - HEADER_SIZE

ESTADO_ESPERANDO_INICIO = "ESPERANDO_INICIO"
ESTADO_RECIBIENDO = "RECIBIENDO"


class ProtocolError(Exception):
	pass


def crc16_ccitt(data: bytes) -> int:
	crc = 0xFFFF
	for byte in data:
		crc ^= byte << 8
		for _ in range(8):
			if crc & 0x8000:
				crc = ((crc << 1) ^ 0x1021) & 0xFFFF
			else:
				crc = (crc << 1) & 0xFFFF
	return crc


def parse_source(source):
	try:
		return int(source)
	except ValueError:
		return source



# ══════════════════════════════════════════════════════════════════════
# ARUCO + HOMOGRAFÍA
# ══════════════════════════════════════════════════════════════════════


def get_aruco_detector():
	aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
	parameters = cv2.aruco.DetectorParameters()

	# Refinamiento subpíxel: crítico a distancia, donde el marcador
	# ocupa pocos píxeles y cualquier error de esquina se amplifica
	parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
	parameters.cornerRefinementWinSize = 5
	parameters.cornerRefinementMaxIterations = 50
	parameters.cornerRefinementMinAccuracy = 0.01

	# Tolerancia a marcadores pequeños en el frame (2-3 m)
	parameters.minMarkerPerimeterRate = 0.02
	parameters.maxMarkerPerimeterRate = 4.0

	# Tolerancia a blur/rotación en la aproximación poligonal
	parameters.polygonalApproxAccuracyRate = 0.05

	# Threshold adaptativo más flexible ante variación de exposición
	parameters.adaptiveThreshWinSizeMin = 3
	parameters.adaptiveThreshWinSizeMax = 30
	parameters.adaptiveThreshWinSizeStep = 5

	# Más tolerante a bits erróneos en el borde (útil bajo blur de rotación)
	parameters.maxErroneousBitsInBorderRate = 0.5

	return "new", cv2.aruco.ArucoDetector(aruco_dict, parameters)


def detect_aruco_markers(gray, detector_type, detector):
	if detector_type == "new":
		corners, ids, rejected = detector.detectMarkers(gray)
	else:
		aruco_dict, parameters = detector
		corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

	return corners, ids, rejected


def marker_reference_corners():
	margen = ((TAM_ESQUINA - TAM_MARCADOR) * TAM_CELDA) / 2
	tam_marker_px = TAM_MARCADOR * TAM_CELDA

	x_tl = margen
	y_tl = margen
	x_tr = (CELDAS - TAM_ESQUINA) * TAM_CELDA + margen
	y_tr = margen
	x_bl = margen
	y_bl = (CELDAS - TAM_ESQUINA) * TAM_CELDA + margen

	return {
		ARUCO_IDS["TL"]: np.array([
			[x_tl, y_tl],
			[x_tl + tam_marker_px, y_tl],
			[x_tl + tam_marker_px, y_tl + tam_marker_px],
			[x_tl, y_tl + tam_marker_px]
		], dtype=np.float32),
		ARUCO_IDS["TR"]: np.array([
			[x_tr, y_tr],
			[x_tr + tam_marker_px, y_tr],
			[x_tr + tam_marker_px, y_tr + tam_marker_px],
			[x_tr, y_tr + tam_marker_px]
		], dtype=np.float32),
		ARUCO_IDS["BL"]: np.array([
			[x_bl, y_bl],
			[x_bl + tam_marker_px, y_bl],
			[x_bl + tam_marker_px, y_bl + tam_marker_px],
			[x_bl, y_bl + tam_marker_px]
		], dtype=np.float32)
	}


def estimate_homography(corners, ids):
	if ids is None:
		return None, []

	ids_flat = ids.flatten().tolist()
	reference = marker_reference_corners()

	src_points, dst_points, detected_valid_ids = [], [], []
	for marker_corners, marker_id in zip(corners, ids_flat):
		if marker_id not in reference:
			continue
		src_points.extend(marker_corners[0].astype(np.float32))
		dst_points.extend(reference[marker_id])
		detected_valid_ids.append(marker_id)

	if len(detected_valid_ids) < 3:
		return None, detected_valid_ids

	src_points = np.array(src_points, dtype=np.float32)
	dst_points = np.array(dst_points, dtype=np.float32)

	method = cv2.USAC_MAGSAC if hasattr(cv2, "USAC_MAGSAC") else cv2.RANSAC
	H, mask = cv2.findHomography(src_points, dst_points, method, 4.0)

	if H is None:
		return None, detected_valid_ids

	# Rechazar homografías mal condicionadas (casi singulares)
	if abs(np.linalg.det(H)) < 1e-6:
		return None, detected_valid_ids

	# Validar error de reproyección real, no solo confiar en el inlier mask
	proyectados = cv2.perspectiveTransform(src_points.reshape(-1, 1, 2), H)
	error = np.linalg.norm(proyectados.reshape(-1, 2) - dst_points, axis=1)
	if np.mean(error) > 6.0:  # px en el espacio de referencia (48*32=1536)
		return None, detected_valid_ids

	return H, detected_valid_ids


def rectify_frame(frame, detector_type, detector):
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	corners, ids, rejected = detect_aruco_markers(gray, detector_type, detector)
	H, detected_ids = estimate_homography(corners, ids)

	debug = frame.copy()
	if ids is not None:
		cv2.aruco.drawDetectedMarkers(debug, corners, ids)

	if H is None:
		return None, debug, detected_ids

	warped = cv2.warpPerspective(frame, H, (PANTALLA_W, PANTALLA_H))
	return warped, debug, detected_ids


# ══════════════════════════════════════════════════════════════════════
# EXTRACCIÓN DE CELDAS Y DECODIFICACIÓN RS
# ══════════════════════════════════════════════════════════════════════


def obtener_celdas_datos():
	reservadas = set()

	for fila in range(TAM_ESQUINA):
		for col in range(TAM_ESQUINA):
			reservadas.add((fila, col))
			reservadas.add((fila, CELDAS - TAM_ESQUINA + col))

	for fila in range(CELDAS - TAM_ESQUINA, CELDAS):
		for col in range(TAM_ESQUINA):
			reservadas.add((fila, col))
			reservadas.add((fila, CELDAS - TAM_ESQUINA + col))

	celdas = []
	for fila in range(CELDAS):
		for col in range(CELDAS):
			if (fila, col) not in reservadas:
				celdas.append((fila, col))

	return celdas


def extract_cell_means(warped):
	gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
	celdas = obtener_celdas_datos()
	means = []
	crop_margin = max(1, int(TAM_CELDA * 0.15))

	for fila, col in celdas:
		x1 = col * TAM_CELDA + crop_margin
		y1 = fila * TAM_CELDA + crop_margin
		x2 = (col + 1) * TAM_CELDA - crop_margin
		y2 = (fila + 1) * TAM_CELDA - crop_margin
		roi = gray[y1:y2, x1:x2]
		means.append(float(np.mean(roi)))

	return np.array(means, dtype=np.float32)


def means_to_bits(means):
	p_low = np.percentile(means, 10)
	p_high = np.percentile(means, 90)
	threshold = (p_low + p_high) / 2.0
	bits = (means > threshold).astype(np.uint8).tolist()
	return bits, threshold


def bits_to_bytes(bits):
	bits = bits[:BITS_PAQUETE]
	data = bytearray()

	for i in range(0, len(bits), 8):
		byte = 0
		for bit in bits[i:i + 8]:
			byte = (byte << 1) | int(bit)
		data.append(byte)

	return bytes(data)

def reed_solomon_decode(packet):
	if RSCodec is None:
		raise RuntimeError("Falta instalar reedsolo. Ejecuta: pip install reedsolo")

	rs = RSCodec(RS_REDUNDANCIA)
	decoded = rs.decode(packet)

	if isinstance(decoded, tuple):
		payload = decoded[0]

		try:
			errata = decoded[2]
			errores_corregidos = len(errata)
		except Exception:
			errores_corregidos = 0

		return bytes(payload), errores_corregidos

	return bytes(decoded), 0


def parse_protocol_payload(decoded: bytes):
	if len(decoded) != RS_DATOS:
		raise ProtocolError("Longitud RS decodificada inválida")

	header = decoded[:HEADER_SIZE]
	payload_area = decoded[HEADER_SIZE:]

	if header[0:3] != MAGIC:
		raise ProtocolError("MAGIC inválido")

	if header[3] != VERSION:
		raise ProtocolError("Versión inválida")

	tipo = header[4]
	if tipo not in TIPOS_NOMBRE:
		raise ProtocolError("Tipo de frame inválido")

	seq = header[5]
	total = header[6]
	msg_id = header[7]
	message_len = (header[8] << 8) | header[9]
	payload_len = header[10]
	crc_rx = (header[12] << 8) | header[13]

	if total == 0:
		raise ProtocolError("Total de frames inválido")

	if payload_len > PAYLOAD_POR_FRAME:
		raise ProtocolError("Payload_len inválido")

	payload_real = payload_area[:payload_len]
	crc_calc = crc16_ccitt(header[0:12] + payload_real)
	if crc_rx != crc_calc:
		raise ProtocolError("CRC inválido")

	if tipo == TIPO_DATA and seq >= total:
		raise ProtocolError("SEQ fuera de rango")

	return {
		"tipo": tipo,
		"tipo_nombre": TIPOS_NOMBRE[tipo],
		"seq": seq,
		"total": total,
		"msg_id": msg_id,
		"message_len": message_len,
		"payload_len": payload_len,
		"payload": payload_real
	}


def decode_warped_frame(warped):
	means = extract_cell_means(warped)
	bits, threshold = means_to_bits(means)

	packet = bits_to_bytes(bits)

	decoded, errores_corregidos = reed_solomon_decode(packet)

	frame_info = parse_protocol_payload(decoded)

	frame_info["threshold"] = threshold
	frame_info["errores_rs"] = errores_corregidos
	frame_info["ber_estimada"] = errores_corregidos / BITS_PAQUETE

	return frame_info


# ══════════════════════════════════════════════════════════════════════
# ENSAMBLADOR CON ESTADO: INICIO → DATA → FIN
# ══════════════════════════════════════════════════════════════════════


class MessageAssembler:
	def __init__(self, timeout_seconds=8.0):
		self.timeout_seconds = timeout_seconds
		self.reset()

	def reset(self):
		self.estado = ESTADO_ESPERANDO_INICIO
		self.msg_id = None
		self.total = None
		self.message_len = None
		self.frames = {}
		self.end_seen = False
		self.started_at = None
		self.updated_at = None

		self.rx_start_time = None
		self.rx_end_time = None
		self.total_rs_errors = 0

	def check_timeout(self):
		if self.estado == ESTADO_RECIBIENDO and self.updated_at is not None:
			if time.time() - self.updated_at > self.timeout_seconds:
				self.reset()
				return True
		return False

	def _same_message(self, frame_info):
		return (
			frame_info["msg_id"] == self.msg_id and
			frame_info["total"] == self.total and
			frame_info["message_len"] == self.message_len
		)

	def add_frame(self, frame_info):
		self.check_timeout()
		tipo = frame_info["tipo"]
		now = time.time()

		if self.estado == ESTADO_ESPERANDO_INICIO:
			if tipo != TIPO_PREAMBLE:
				return False, "Esperando PREAMBLE"

			self.estado = ESTADO_RECIBIENDO

			self.rx_start_time = time.perf_counter()

			self.msg_id = frame_info["msg_id"]
			self.total = frame_info["total"]
			self.message_len = frame_info["message_len"]
			self.frames = {}
			self.end_seen = False
			self.started_at = now
			self.updated_at = now
			return True, f"Inicio detectado | msg_id={self.msg_id} | total={self.total} | bytes={self.message_len}"

		if self.estado == ESTADO_RECIBIENDO:
			if not self._same_message(frame_info):
				if tipo == TIPO_PREAMBLE:
					self.reset()
					return self.add_frame(frame_info)
				return False, "Frame de otro mensaje ignorado"

			self.updated_at = now

			if tipo == TIPO_PREAMBLE:
				return True, "PREAMBLE repetido"

			if tipo == TIPO_DATA:
				self.total_rs_errors += frame_info.get("errores_rs", 0)
				seq = frame_info["seq"]
				new_frame = seq not in self.frames
				self.frames[seq] = frame_info["payload"]

				if new_frame:
					return True, f"DATA {seq + 1}/{self.total} recibido"
				return True, f"DATA {seq + 1}/{self.total} repetido"

			if tipo == TIPO_END:
				self.end_seen = True
				return True, "END detectado"

		return False, "Estado inválido"

	def is_complete(self):

		completo = (self.estado == ESTADO_RECIBIENDO and self.total is not None and len(self.frames) == self.total)

		if completo and self.rx_end_time is None:
			self.rx_end_time = time.perf_counter()

		return completo

	def build_message(self):
		if not self.is_complete():
			return None

		message_bytes = bytearray()
		for i in range(self.total):
			message_bytes.extend(self.frames[i])

		message_bytes = bytes(message_bytes[:self.message_len])
		return message_bytes.decode("latin-1", errors="replace")

	def progress(self):
		if self.estado == ESTADO_ESPERANDO_INICIO:
			return "esperando inicio"
		return f"{len(self.frames)}/{self.total} | END={'sí' if self.end_seen else 'no'}"

	def reception_time(self):

		if self.rx_start_time is None:
			return None

		if self.rx_end_time is not None:
			return self.rx_end_time - self.rx_start_time

		return time.perf_counter() - self.rx_start_time

# ══════════════════════════════════════════════════════════════════════
# PROCESAMIENTO
# ══════════════════════════════════════════════════════════════════════



def process_video_or_camera(source, detector_type, detector, assembler, args):
	imprimir = 0
	cap = cv2.VideoCapture(source)
	if not cap.isOpened():
		raise RuntimeError(f"No se pudo abrir la fuente: {source}")

	if isinstance(source, int):
		cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
		cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
		cap.set(cv2.CAP_PROP_FPS, args.fps)
		cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
		cap.set(cv2.CAP_PROP_AUTOFOCUS, -5)
		cap.set(cv2.CAP_PROP_FOCUS, 10)
		
		cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
		cap.set(cv2.CAP_PROP_GAIN, 0)
		cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
		cap.set(cv2.CAP_PROP_EXPOSURE, -10)

	print("Receptor iniciado.")

	frame_index = 0
	fps_timer = time.time()
	fps_counter = 0
	fps_real = 0.0
	last_warp = None
	last_status = ""

	while True:
		ret, frame = cap.read()
		if not ret:
			print("Fin de video o no se pudo leer frame.")
			break

		frame_index += 1
		fps_counter += 1

		now = time.time()
		elapsed = now - fps_timer
		if elapsed >= 1.0:
			fps_real = fps_counter / elapsed
			fps_counter = 0
			fps_timer = now

		warped, debug, detected_ids = rectify_frame(frame, detector_type, detector)
		status = "SIN_ARUCO"
		message = None

		if warped is not None:
			last_warp = warped.copy()
			try:
				frame_info = decode_warped_frame(warped)
				ok, status = assembler.add_frame(frame_info)

				if status != last_status or frame_info["tipo"] == TIPO_DATA:
					print(f"[{frame_index}] {status} | {assembler.progress()} | FPS {fps_real:.2f}")
					last_status = status

				if assembler.is_complete() & (imprimir == 0):
					imprimir = 1
					message = assembler.build_message()

					rx_time = assembler.reception_time()

					print("\nMENSAJE DECODIFICADO:")
					print(message)

					ber_global = (
					assembler.total_rs_errors /
					max(1, assembler.total * BITS_PAQUETE)
					)

					print("\nESTADÍSTICAS DEL CANAL")
					print(f"Frames recibidos: "f"{len(assembler.frames)}/{assembler.total}")

					print(f"Bytes recuperados: "f"{assembler.message_len}")

					print(f"Errores RS corregidos: "f"{assembler.total_rs_errors}")

					print(f"BER estimada: "f"{ber_global:.6e}")	

					print(f"Tiempo total recepción+decodificación: "f"{rx_time:.6f} s")

			except ReedSolomonError:
				status = "RS_ERROR"
			except ProtocolError as exc:
				status = f"PROTOCOLO: {exc}"
			except Exception as exc:
				status = f"ERROR: {exc}"

		if args.save_warps and warped is not None:
			os.makedirs(args.save_warps, exist_ok=True)
			cv2.imwrite(os.path.join(args.save_warps, f"warp_{frame_index:06d}.png"), warped)

		if args.show:
			info = f"IDs:{detected_ids} | {assembler.estado} | {status} | {assembler.progress()} | FPS:{fps_real:.2f}"
			cv2.putText(debug, info[:120], (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
			cv2.imshow("Receptor - deteccion ArUco", debug)

			if warped is not None:
				preview = cv2.resize(warped, (768, 768))
				cv2.imshow("Receptor - frame rectificado", preview)

			key = cv2.waitKey(1) & 0xFF
			if key == ord("q"):
				break
			if key == ord("s") and last_warp is not None:
				os.makedirs("warps_manuales", exist_ok=True)
				filename = f"warps_manuales/warp_{int(time.time())}.png"
				cv2.imwrite(filename, last_warp)
				print(f"Warp guardado: {filename}")

	cap.release()
	cv2.destroyAllWindows()


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════


def main():
	parser = argparse.ArgumentParser(description="Receptor PDS con ArUco, PREAMBLE, DATA, END, secuencia, longitud y CRC.")
	parser.add_argument("--source", default="2", help="Cámara local, video, imagen o carpeta de imágenes. Ej: 0, video.mp4, frame.png, frames_tx_protocol")
	parser.add_argument("--width", type=int, default=1920, help="Ancho solicitado si la fuente es cámara.")
	parser.add_argument("--height", type=int, default=1080, help="Alto solicitado si la fuente es cámara.")
	parser.add_argument("--fps", type=int, default=60, help="FPS solicitado si la fuente es cámara.")
	parser.add_argument("--show", action="store_true", help="Muestra ventanas de detección y frame rectificado.")
	parser.add_argument("--save-warps", default="", help="Carpeta para guardar frames rectificados.")
	parser.add_argument("--timeout", type=float, default=8.0, help="Tiempo máximo sin recibir frames válidos antes de reiniciar recepción.")
	parser.add_argument("--wait-ms", type=int, default=1, help="Espera entre imágenes cuando source es una carpeta y --show está activo.")
	parser.add_argument("--continue-after-message", action="store_true", help="Después de decodificar un mensaje, sigue escuchando otro.")
	args = parser.parse_args()

	if RSCodec is None:
		print("Falta la librería reedsolo.")
		print("Instálala con: pip install reedsolo")
		return

	source = parse_source(args.source)
	detector_type, detector = get_aruco_detector()
	assembler = MessageAssembler(timeout_seconds=args.timeout)

	print("── Parámetros del receptor ─────────────────────────")
	print(f"Frame ideal:          {PANTALLA_W}x{PANTALLA_H} px")
	print(f"Grilla:               {CELDAS}x{CELDAS}")
	print(f"Tamaño celda:         {TAM_CELDA}px")
	print(f"Marcadores ArUco:     DICT_5X5_100 | IDs {ARUCO_IDS}")
	print(f"RS:                   RS(255,{RS_DATOS}) con {RS_REDUNDANCIA} bytes de paridad")
	print(f"Cabecera protocolo:   {HEADER_SIZE} bytes")
	print(f"Datos por frame DATA: {PAYLOAD_POR_FRAME} bytes")
	print("────────────────────────────────────────────────────\n")

	process_video_or_camera(source, detector_type, detector, assembler, args)


if __name__ == "__main__":
	main()
