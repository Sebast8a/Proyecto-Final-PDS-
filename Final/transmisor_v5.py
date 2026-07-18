#!/usr/bin/env python3
import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from reedsolo import RSCodec

# ══════════════════════════════════════════════════════════════════════
# PARÁMETROS CONSERVADOS DEL TRANSMISOR ORIGINAL
# ══════════════════════════════════════════════════════════════════════

TAM_CELDA = 32
CELDAS = 24
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

BYTES_FRAME = 38
RS_REDUNDANCIA = 14
RS_DATOS = BYTES_FRAME - RS_REDUNDANCIA
BITS_PAQUETE = BYTES_FRAME * 8

# ══════════════════════════════════════════════════════════════════════
# PROTOCOLO DE TRAMAS PARA CÁMARA EN TIEMPO REAL
# ══════════════════════════════════════════════════════════════════════

MAGIC = b"PDS"
VERSION = 1
TIPO_PREAMBLE = 1
TIPO_DATA = 2
TIPO_END = 3

HEADER_SIZE = 16
PAYLOAD_POR_FRAME = RS_DATOS - HEADER_SIZE
MAX_TOTAL_FRAMES = 255
MAX_MESSAGE_BYTES = 65535

# Cabecera de 16 bytes antes de Reed-Solomon:
# 0..2   MAGIC = PDS
# 3      VERSION
# 4      TIPO: 1 PREAMBLE, 2 DATA, 3 END
# 5      SEQ: número de frame de datos
# 6      TOTAL: total de frames DATA
# 7      MSG_ID: identificador corto del mensaje
# 8..9   LONGITUD total del mensaje en bytes
# 10     PAYLOAD_LEN real de este frame
# 11     RESERVADO
# 12..13 CRC16 de cabecera[0:12] + payload_real
# 14..15 RESERVADO


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


def crear_header(tipo: int, seq: int, total: int, msg_id: int, message_len: int, payload_len: int, payload_real: bytes) -> bytes:
	header = bytearray(HEADER_SIZE)
	header[0:3] = MAGIC
	header[3] = VERSION
	header[4] = tipo
	header[5] = seq & 0xFF
	header[6] = total & 0xFF
	header[7] = msg_id & 0xFF
	header[8] = (message_len >> 8) & 0xFF
	header[9] = message_len & 0xFF
	header[10] = payload_len & 0xFF
	header[11] = 0
	crc = crc16_ccitt(bytes(header[0:12]) + payload_real)
	header[12] = (crc >> 8) & 0xFF
	header[13] = crc & 0xFF
	header[14] = 0
	header[15] = 0
	return bytes(header)


def crear_payload_rs(tipo: int, seq: int, total: int, msg_id: int, message_len: int, payload_real: bytes) -> bytes:
	if len(payload_real) > PAYLOAD_POR_FRAME:
		raise ValueError("El payload supera la capacidad del frame")

	header = crear_header(tipo, seq, total, msg_id, message_len, len(payload_real), payload_real)
	payload_padded = payload_real.ljust(PAYLOAD_POR_FRAME, b"\x00")
	payload_rs = header + payload_padded

	if len(payload_rs) != RS_DATOS:
		raise ValueError("Payload interno RS_DATOS no cuadra")

	rs = RSCodec(RS_REDUNDANCIA)
	return bytes(rs.encode(payload_rs))


def texto_a_bytes(mensaje: str) -> bytes:
	return mensaje.encode("latin-1")


def armar_paquetes_protocolo(mensaje: str, repeticiones_inicio: int = 6, repeticiones_fin: int = 4) -> list:
	mensaje_bytes = texto_a_bytes(mensaje)

	if len(mensaje_bytes) > MAX_MESSAGE_BYTES:
		raise ValueError(f"Mensaje demasiado largo. Máximo {MAX_MESSAGE_BYTES} bytes")

	bloques = [
		mensaje_bytes[i:i + PAYLOAD_POR_FRAME]
		for i in range(0, len(mensaje_bytes), PAYLOAD_POR_FRAME)
	]

	if not bloques:
		bloques = [b""]

	total = len(bloques)
	if total > MAX_TOTAL_FRAMES:
		raise ValueError("El mensaje produce más de 255 frames")

	msg_id = int(time.time()) % 256
	paquetes = []

	preamble = crear_payload_rs(TIPO_PREAMBLE, 0, total, msg_id, len(mensaje_bytes), b"")
	for _ in range(repeticiones_inicio):
		paquetes.append(preamble)

	for seq, bloque in enumerate(bloques):
		paquete_data = crear_payload_rs(TIPO_DATA, seq, total, msg_id, len(mensaje_bytes), bloque)
		paquetes.append(paquete_data)

	end = crear_payload_rs(TIPO_END, total, total, msg_id, len(mensaje_bytes), b"")
	for _ in range(repeticiones_fin):
		paquetes.append(end)

	return paquetes


def paquete_a_bits(paquete: bytes) -> list:
	bits = []
	for byte in paquete:
		for i in range(7, -1, -1):
			bits.append((byte >> i) & 1)

	bits += [0] * (CELDAS_DATOS - len(bits))
	return bits


# ══════════════════════════════════════════════════════════════════════
# GENERACIÓN DE FRAMES CON ARUCO
# ══════════════════════════════════════════════════════════════════════


def crear_marcador_aruco(marker_id: int) -> Image.Image:
	aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
	tam_px = TAM_MARCADOR * TAM_CELDA

	if hasattr(cv2.aruco, "generateImageMarker"):
		marker_np = cv2.aruco.generateImageMarker(aruco_dict, marker_id, tam_px, borderBits=1)
	else:
		marker_np = np.zeros((tam_px, tam_px), dtype=np.uint8)
		cv2.aruco.drawMarker(aruco_dict, marker_id, tam_px, marker_np, 1)

	return Image.fromarray(marker_np).convert("L")


def dibujar_marcador(draw: ImageDraw.ImageDraw, col_celda: int, fila_celda: int, marker_id: int):
	px_reserva = col_celda * TAM_CELDA
	py_reserva = fila_celda * TAM_CELDA
	tam_reserva_px = TAM_ESQUINA * TAM_CELDA
	tam_marcador_px = TAM_MARCADOR * TAM_CELDA
	margen_px = (tam_reserva_px - tam_marcador_px) // 2

	marcador = crear_marcador_aruco(marker_id)
	draw._image.paste(marcador, (px_reserva + margen_px, py_reserva + margen_px))


def obtener_celdas_datos() -> list:
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


def bits_a_imagen(bits: list) -> Image.Image:
	if len(bits) != CELDAS_DATOS:
		raise ValueError(f"Se esperaban {CELDAS_DATOS} bits, llegaron {len(bits)}")

	img = Image.new("L", (PANTALLA_W, PANTALLA_H), color=128)
	draw = ImageDraw.Draw(img)

	celdas = obtener_celdas_datos()
	for bit, (fila, col) in zip(bits, celdas):
		px = col * TAM_CELDA
		py = fila * TAM_CELDA
		color = 0 if bit == 0 else 255
		draw.rectangle([px, py, px + TAM_CELDA - 1, py + TAM_CELDA - 1], fill=color)

	def dibujar_quiet_zone(col_inicio, fila_inicio):
		px = col_inicio * TAM_CELDA
		py = fila_inicio * TAM_CELDA
		tam = TAM_ESQUINA * TAM_CELDA
		draw.rectangle([px, py, px + tam - 1, py + tam - 1], fill=255)

	dibujar_quiet_zone(0, 0)
	dibujar_quiet_zone(CELDAS - TAM_ESQUINA, 0)
	dibujar_quiet_zone(0, CELDAS - TAM_ESQUINA)

	px_br = (CELDAS - TAM_ESQUINA) * TAM_CELDA
	py_br = (CELDAS - TAM_ESQUINA) * TAM_CELDA
	tam_esq = TAM_ESQUINA * TAM_CELDA
	draw.rectangle([px_br, py_br, px_br + tam_esq - 1, py_br + tam_esq - 1], fill=128)

	dibujar_marcador(draw, 0, 0, ARUCO_IDS["TL"])
	dibujar_marcador(draw, CELDAS - TAM_ESQUINA, 0, ARUCO_IDS["TR"])
	dibujar_marcador(draw, 0, CELDAS - TAM_ESQUINA, ARUCO_IDS["BL"])

	return img


def generar_frames_protocolo(mensaje: str, repeticiones_inicio: int = 6, repeticiones_fin: int = 4) -> list:
	paquetes = armar_paquetes_protocolo(mensaje, repeticiones_inicio, repeticiones_fin)
	frames = []

	for paquete in paquetes:
		bits = paquete_a_bits(paquete)
		frames.append(bits_a_imagen(bits))

	return frames


# ══════════════════════════════════════════════════════════════════════
# SALIDA: GUARDAR O MOSTRAR EN PANTALLA
# ══════════════════════════════════════════════════════════════════════


def guardar_frames(frames: list, out_dir: str):
	os.makedirs(out_dir, exist_ok=True)
	for i, frame in enumerate(frames):
		frame.save(os.path.join(out_dir, f"frame_{i:04d}.png"))
	print(f"Frames guardados en: {out_dir}")


def mostrar_frames(frames: list, fps_tx: float, escala: int = 1, fullscreen: bool = False, loop: bool = False):
	intervalo_ms = max(1, int(1000 / fps_tx))
	window_name = "Transmisor PDS ArUco - protocolo"
	cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

	if fullscreen:
		cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
	else:
		cv2.resizeWindow(window_name, PANTALLA_W * escala, PANTALLA_H * escala)

	print("Transmisor iniciado.")
	print("Presiona q para salir.")

	while True:
		for frame in frames:
			img_np = np.array(frame)
			img_bgr = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
			cv2.imshow(window_name, img_bgr)
			key = cv2.waitKey(intervalo_ms) & 0xFF
			if key == ord("q"):
				cv2.destroyAllWindows()
				return

		if not loop:
			break

	cv2.waitKey(1000)
	cv2.destroyAllWindows()


def main():
	parser = argparse.ArgumentParser(description="Transmisor PDS con ArUco, preámbulo, secuencia, longitud y fin.")
	parser.add_argument("--message", default="", help="Mensaje a transmitir. Si se omite, se pide por input().")
	parser.add_argument("--message-file", default="", help="Archivo de texto a transmitir.")
	parser.add_argument("--out-dir", default="frames_tx_protocol", help="Carpeta para guardar frames PNG.")
	parser.add_argument("--save", action="store_true", help="Guarda los frames como imágenes PNG.")
	parser.add_argument("--display", action="store_true", help="Muestra los frames en una ventana para transmitir con pantalla.")
	parser.add_argument("--fullscreen", action="store_true", help="Muestra la ventana en pantalla completa.")
	parser.add_argument("--loop", action="store_true", help="Repite la transmisión en bucle.")
	parser.add_argument("--fps-tx", type=float, default=10.0, help="Frames mostrados por segundo por el transmisor.")
	parser.add_argument("--preamble-repeat", type=int, default=6, help="Repeticiones del frame PREAMBLE.")
	parser.add_argument("--end-repeat", type=int, default=4, help="Repeticiones del frame END.")
	args = parser.parse_args()

	if args.message_file:
		mensaje = Path(args.message_file).read_text(encoding="latin-1")
	elif args.message:
		mensaje = args.message
	else:
		mensaje = input("Mensaje a transmitir: ")

	frames = generar_frames_protocolo(mensaje, args.preamble_repeat, args.end_repeat)
	mensaje_bytes = texto_a_bytes(mensaje)
	data_frames = max(1, (len(mensaje_bytes) + PAYLOAD_POR_FRAME - 1) // PAYLOAD_POR_FRAME)

	print("── Parámetros del transmisor ───────────────────────")
	print(f"Frame ideal:          {PANTALLA_W}x{PANTALLA_H} px")
	print(f"Grilla:               {CELDAS}x{CELDAS}")
	print(f"Tamaño celda:         {TAM_CELDA}px")
	print(f"Marcadores ArUco:     DICT_5X5_100 | IDs {ARUCO_IDS}")
	print(f"RS:                   RS(255,{RS_DATOS}) con {RS_REDUNDANCIA} bytes de paridad")
	print(f"Cabecera protocolo:   {HEADER_SIZE} bytes")
	print(f"Datos por frame DATA: {PAYLOAD_POR_FRAME} bytes")
	print(f"Bytes mensaje:        {len(mensaje_bytes)}")
	print(f"Frames DATA:          {data_frames}")
	print(f"Frames totales TX:    {len(frames)}")
	print(f"Tiempo @ {args.fps_tx} fps:     {len(frames) / args.fps_tx:.2f} s")
	print("────────────────────────────────────────────────────\n")

	if args.save or not args.display:
		guardar_frames(frames, args.out_dir)

	if args.display:
		mostrar_frames(frames, args.fps_tx, fullscreen=args.fullscreen, loop=args.loop)


if __name__ == "__main__":
	main()