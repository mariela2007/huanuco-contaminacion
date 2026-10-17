# -*- coding: utf-8 -*-
"""Detector ambiental orientativo: fuego vs. indicios de tala.

El detector de color no es un modelo de visión entrenado. Da prioridad a
zonas cálidas muy brillantes y saturadas (posible llama), no a tonos marrones
propios de troncos o superficies de corte. Si encuentra un .pt de Ultralytics
junto al script, también informa la clase sugerida por ese modelo.

Uso:
  python probar-modelo.py                 # analiza imágenes de esta carpeta
  python probar-modelo.py imagen.webp     # analiza una o más rutas
  python probar-modelo.py --server        # abre el sitio y conecta el modelo .pt

Instalación:
  pip install opencv-python numpy ultralytics
"""
from __future__ import annotations

import base64
import glob
import json
import os
import sys
import unicodedata
from pathlib import Path
from http.server import BaseHTTPRequestHandler

import cv2
import numpy as np

# Criterios conservadores, coordinados con el HTML.
MAX_SIDE = 640
SAT_MIN = 128               # HSV OpenCV: evita marrones poco saturados
VALUE_MIN = 178             # brillo alto: un tronco marrón suele quedar fuera
VALUE_CORE_MIN = 210        # centro amarillo/naranja muy brillante
AREA_MIN_FIRE = 0.006       # al menos 0,6 % de píxeles de llama candidata
AREA_MIN_COMPONENT = 0.001  # región conectada de al menos 0,1 %
AREA_MIN_CORE = 0.0015      # núcleo brillante de al menos 0,15 %
AREA_MIN_STRONG_ORANGE = 0.020
AREA_MIN_STRONG_COMPONENT = 0.006

FIRE_WORDS = ("fuego", "incendio", "fire", "flame", "llama", "quema", "humo")
TALA_WORDS = ("tala", "arbol", "madera", "tronco", "tocon", "deforestacion",
              "aserrio", "lena", "wood", "logging", "stump", "timber")


def normalizar(texto: str) -> str:
    plano = unicodedata.normalize("NFD", str(texto).lower())
    return "".join(c for c in plano if unicodedata.category(c) != "Mn")


def leer_imagen(ruta: str) -> np.ndarray | None:
    try:
        datos = np.fromfile(ruta, dtype=np.uint8)
        return cv2.imdecode(datos, cv2.IMREAD_COLOR) if datos.size else None
    except Exception:
        return None


def zonas(mascara: np.ndarray, area_minima: float, limite: int = 8):
    total = mascara.shape[0] * mascara.shape[1]
    n, _, stats, _ = cv2.connectedComponentsWithStats(mascara, connectivity=8)
    componentes = []
    for i in range(1, n):
        x, y, w, h, area = map(int, stats[i])
        if area >= area_minima * total:
            componentes.append((x, y, w, h, area))
    componentes.sort(key=lambda item: item[4], reverse=True)
    return componentes[:limite]


def analizar_color(imagen: np.ndarray) -> dict:
    alto, ancho = imagen.shape[:2]
    escala = min(1.0, MAX_SIDE / max(alto, ancho))
    if escala < 1:
        imagen = cv2.resize(imagen, (max(1, int(ancho * escala)), max(1, int(alto * escala))),
                            interpolation=cv2.INTER_AREA)

    alto, ancho = imagen.shape[:2]
    total = float(alto * ancho)
    hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    b, g, r = cv2.split(imagen)

    # H de OpenCV va de 0 a 180. Incluye rojo/naranja y amarillo de llama,
    # pero exige brillo y saturación altos para descartar madera marrón.
    warm_hue = (h <= 20) | (h >= 173)
    red_dominance = (r.astype(np.int16) >= (g.astype(np.int16) * 0.78)) & \
                    (r.astype(np.int16) >= (b.astype(np.int16) * 1.05))
    mask = (warm_hue & red_dominance & (s >= SAT_MIN) & (v >= VALUE_MIN)).astype(np.uint8) * 255

    # Núcleo muy luminoso amarillo/naranja: madera cortada normalmente no lo tiene.
    core_hue = (h >= 3) & (h <= 21)
    core = (core_hue & (s >= 82) & (v >= VALUE_CORE_MIN) & red_dominance).astype(np.uint8) * 255

    # Limpieza pequeña para unir la llama sin convertir el tronco entero en una región.
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    core = cv2.morphologyEx(core, cv2.MORPH_OPEN, kernel)

    comps = zonas(mask, AREA_MIN_COMPONENT)
    cores = zonas(core, AREA_MIN_CORE, limite=4)
    warm_pixels = sum(z[4] for z in comps)
    core_pixels = cv2.countNonZero(core)
    ratio = warm_pixels / total
    core_ratio = core_pixels / total
    largest_ratio = comps[0][4] / total if comps else 0.0

    # No se declara fuego por color cálido solo: exige núcleo brillante o una
    # mancha naranja excepcionalmente grande y conectada.
    hot_core_evidence = (ratio >= AREA_MIN_FIRE and
                         largest_ratio >= AREA_MIN_COMPONENT and
                         core_ratio >= AREA_MIN_CORE)
    strong_orange_evidence = (ratio >= AREA_MIN_STRONG_ORANGE and
                               largest_ratio >= AREA_MIN_STRONG_COMPONENT)
    fire = hot_core_evidence or (strong_orange_evidence and core_ratio >= 0.003)

    return {
        "fire": bool(fire),
        "ratio": ratio,
        "largest_ratio": largest_ratio,
        "core_ratio": core_ratio,
        "boxes": comps[:4],
        "width": ancho,
        "height": alto,
    }


def sugerencia_modelo(model_path: str | None, imagen: np.ndarray):
    """Devuelve la mejor clase del .pt si Ultralytics está instalado y el modelo sirve."""
    if not model_path:
        return None
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        results = model.predict(source=imagen, verbose=False)
        if not results:
            return None
        result = results[0]
        names = getattr(result, "names", getattr(model, "names", {})) or {}
        opciones = []
        probs = getattr(result, "probs", None)
        if probs is not None:
            idx = int(probs.top1)
            conf = float(probs.top1conf)
            nombre = names.get(idx, str(idx)) if isinstance(names, dict) else names[idx]
            opciones.append((str(nombre), conf))
        boxes = getattr(result, "boxes", None)
        if boxes is not None and len(boxes) > 0:
            for cls, conf in zip(boxes.cls.tolist(), boxes.conf.tolist()):
                idx = int(cls)
                nombre = names.get(idx, str(idx)) if isinstance(names, dict) else names[idx]
                opciones.append((str(nombre), float(conf)))
        return max(opciones, key=lambda item: item[1]) if opciones else None
    except Exception as exc:
        return {"error": str(exc)}


def categoria_modelo(nombre: str) -> str:
    n = normalizar(nombre)
    if any(word in n for word in FIRE_WORDS):
        return "Incendios forestales"
    if any(word in n for word in TALA_WORDS):
        return "Tala de árboles"
    return nombre


def describir(ruta: str, datos: dict, model_suggestion):
    nombre_archivo = normalizar(Path(ruta).name)
    pista_tala = any(word in nombre_archivo for word in TALA_WORDS)
    pista_fuego = any(word in nombre_archivo for word in FIRE_WORDS)
    modelo_label = None
    modelo_conf = 0.0
    if model_suggestion and "error" not in model_suggestion:
        modelo_label, modelo_conf = model_suggestion
        modelo_label = categoria_modelo(modelo_label)

    # Prioridad: un modelo semántico confiable puede reconocer tala/fuego.
    # Sin esa evidencia, una pista explícita de tala en el nombre tiene prioridad
    # sobre colores cálidos de madera, que no son prueba de llama.
    modelo_fuego_fuerte = modelo_label == "Incendios forestales" and modelo_conf >= 0.75
    if modelo_label == "Tala de árboles" and modelo_conf >= 0.50:
        veredicto = f"El modelo .pt sugiere: Tala de árboles ({modelo_conf * 100:.1f} %)"
        fuente = "modelo .pt (reconocimiento semántico; revisar la clase entrenada)"
    elif pista_tala and not modelo_fuego_fuerte:
        veredicto = "POSIBLE TALA: el nombre indica tala/madera; el color cálido no se considera fuego"
        fuente = "pista de contexto del nombre; no es visión semántica"
    elif modelo_fuego_fuerte:
        veredicto = f"El modelo .pt sugiere: Incendios forestales ({modelo_conf * 100:.1f} %)"
        fuente = "modelo .pt con confianza alta; verificar visualmente"
    elif datos["fire"]:
        veredicto = "POSIBLE INCENDIO: señal intensa de llama; verificación humana necesaria"
        fuente = "color brillante + núcleo luminoso; heurística, no semántica"
    elif modelo_label == "Incendios forestales" and modelo_conf >= 0.50:
        veredicto = f"El modelo .pt sugiere débilmente: Incendios forestales ({modelo_conf * 100:.1f} %)"
        fuente = "modelo .pt; confianza moderada, revisar manualmente"
    elif pista_fuego:
        veredicto = "Nombre relacionado con fuego, pero no hay señal cromática suficiente"
        fuente = "color insuficiente; el nombre no confirma fuego"
    else:
        veredicto = "SIN INDICIOS VISUALES SUFICIENTES DE FUEGO O TALA"
        fuente = "detector local orientativo"
    return veredicto, fuente


def guardar_anotada(ruta: str, imagen: np.ndarray, datos: dict, es_fuego: bool) -> str | None:
    if not es_fuego:
        return None
    salida = str(Path(ruta).with_name("resultado_" + Path(ruta).stem + ".jpg"))
    escala_x = imagen.shape[1] / max(1, datos["width"])
    escala_y = imagen.shape[0] / max(1, datos["height"])
    for x, y, w, h, _area in datos["boxes"]:
        p1 = (int(x * escala_x), int(y * escala_y))
        p2 = (int((x + w) * escala_x), int((y + h) * escala_y))
        cv2.rectangle(imagen, p1, p2, (0, 140, 255), 2)
        cv2.putText(imagen, "LLAMA?", (p1[0], max(22, p1[1] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 140, 255), 2, cv2.LINE_AA)
    try:
        ext = Path(salida).suffix
        ok, encoded = cv2.imencode(ext, imagen)
        if ok:
            encoded.tofile(salida)
            return salida
    except Exception:
        pass
    return None


def buscar_modelo(carpeta: Path) -> str | None:
    for candidate in (carpeta / "best.pt", carpeta / "best (2).pt"):
        if candidate.exists():
            return str(candidate)
    candidatos = sorted(carpeta.glob("*.pt"))
    return str(candidatos[0]) if candidatos else None


def listar_imagenes(args: list[str], carpeta: Path) -> list[str]:
    if args:
        return [str(Path(a).expanduser().resolve()) for a in args]
    patron = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"):
        patron.extend(glob.glob(str(carpeta / ext)))
        patron.extend(glob.glob(str(carpeta / ext.upper())))
    return sorted(p for p in set(patron) if not Path(p).name.lower().startswith("resultado_"))



# Mapeo de nombres del dataset a las categorías visibles en la página.
PATRONES_CATEGORIA = {
    "Residuos químicos": ("quimic", "chemical", "pestic", "agrochemical", "toxic", "hazardous"),
    "Contaminación de ríos": ("rio", "river", "water pollution", "contamin", "pollut", "plastic", "litter", "garbage", "trash", "basura", "vertimiento"),
    "Tala de árboles": ("tala", "arbol", "tree cutting", "tree felling", "deforest", "logging", "stump", "tocon", "timber", "madera"),
    "Incendios forestales": ("incendio", "fuego", "fire", "flame", "llama", "wildfire", "forest fire", "quema"),
}


def mapear_categoria_modelo(nombre: str) -> str:
    normal = normalizar(nombre).replace("_", " ").replace("-", " ")
    for categoria, patrones in PATRONES_CATEGORIA.items():
        if any(patron in normal for patron in patrones):
            return categoria
    return "Clase del modelo: " + str(nombre)


def obtener_nombres(model, result):
    nombres = getattr(result, "names", None) or getattr(model, "names", {}) or {}
    return nombres


def nombre_clase(nombres, idx: int) -> str:
    if isinstance(nombres, dict):
        return str(nombres.get(idx, nombres.get(str(idx), idx)))
    try:
        return str(nombres[idx])
    except Exception:
        return str(idx)


def predecir_modelo(model, imagen: np.ndarray) -> dict:
    """Predice con Ultralytics y devuelve solo puntuaciones producidas por el .pt."""
    results = model.predict(source=imagen, verbose=False)
    if not results:
        return {"category": "Sin detección", "confidence": None, "raw_label": "", "top": [], "model_type": "YOLO"}
    result = results[0]
    names = obtener_nombres(model, result)
    scores = {}
    raw_for_category = {}
    is_classification = False

    probs = getattr(result, "probs", None)
    if probs is not None:
        is_classification = True
        vector = probs.data.detach().cpu().numpy().reshape(-1).tolist()
        for idx, score in enumerate(vector):
            raw = nombre_clase(names, idx)
            category = mapear_categoria_modelo(raw)
            # Si varias clases del .pt apuntan a una categoría, conserva la mayor puntuación.
            if float(score) > scores.get(category, -1):
                scores[category] = float(score)
                raw_for_category[category] = raw
    else:
        boxes = getattr(result, "boxes", None)
        if boxes is not None and len(boxes) > 0:
            class_ids = boxes.cls.detach().cpu().numpy().astype(int).tolist()
            confidences = boxes.conf.detach().cpu().numpy().tolist()
            for idx, score in zip(class_ids, confidences):
                raw = nombre_clase(names, idx)
                category = mapear_categoria_modelo(raw)
                if float(score) > scores.get(category, -1):
                    scores[category] = float(score)
                    raw_for_category[category] = raw

    if not scores:
        return {
            "category": "Sin detección", "confidence": None, "raw_label": "",
            "top": [], "model_type": "YOLO detección" if not is_classification else "YOLO clasificación",
            "message": "El modelo no produjo una clase ni una caja para esta imagen."
        }

    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top = [{"category": cat, "confidence": score, "raw_label": raw_for_category[cat]}
           for cat, score in ordered[:4]]
    winner = top[0]
    return {
        "category": winner["category"],
        "confidence": winner["confidence"],
        "raw_label": winner["raw_label"],
        "top": top,
        "model_type": "YOLO clasificación" if is_classification else "YOLO detección",
        "message": "Puntuación emitida por el modelo local; no es una garantía de exactitud."
    }


class LocalHandler(BaseHTTPRequestHandler):
    server_version = "HuanucoLocalAI/1.0"

    def log_message(self, fmt, *args):
        print("[HTTP] " + (fmt % args))

    def send_json(self, status: int, payload: dict):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_json(200, {"ok": True})

    def do_GET(self):
        if self.path.split("?", 1)[0] == "/health":
            model = getattr(self.server, "model", None)
            self.send_json(200, {"ok": bool(model), "model": getattr(self.server, "model_name", None)})
            return
        if self.path.split("?", 1)[0] not in ("/", "/index.html"):
            self.send_error(404, "No encontrado")
            return
        html_path = Path(__file__).resolve().parent / "index.html"
        if not html_path.exists():
            self.send_error(404, "No se encontró index.html junto al script")
            return
        body = html_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path.split("?", 1)[0] != "/analyze":
            self.send_json(404, {"error": "Ruta no encontrada"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 20 * 1024 * 1024:
                self.send_json(413, {"error": "La solicitud está vacía o supera 20 MB."})
                return
            data = json.loads(self.rfile.read(length).decode("utf-8"))
            encoded = str(data.get("image", ""))
            if "," in encoded:
                encoded = encoded.split(",", 1)[1]
            raw = base64.b64decode(encoded, validate=True)
            image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                self.send_json(400, {"error": "No se pudo decodificar la imagen."})
                return
            model = getattr(self.server, "model", None)
            if model is None:
                self.send_json(503, {"error": "No hay un modelo .pt cargado. Coloca best.pt o best (2).pt junto al script."})
                return
            result = predecir_modelo(model, image)
            result["filename"] = str(data.get("filename", "imagen"))[:255]
            result["width"] = int(image.shape[1])
            result["height"] = int(image.shape[0])
            self.send_json(200, result)
        except Exception as exc:
            self.send_json(500, {"error": "Falló el análisis local: " + str(exc)})


def iniciar_servidor(carpeta: Path) -> int:
    model_path = buscar_modelo(carpeta)
    if not model_path:
        print("ERROR: no encontré un modelo .pt. Coloca best.pt o best (2).pt junto a probar-modelo.py.")
        return 1
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: falta Ultralytics. Instálalo con: pip install ultralytics")
        return 1
    try:
        print(f"Cargando modelo: {Path(model_path).name} …")
        model = YOLO(model_path)
    except Exception as exc:
        print(f"ERROR al cargar el modelo: {exc}")
        return 1

    from http.server import ThreadingHTTPServer
    server = ThreadingHTTPServer(("127.0.0.1", 8000), LocalHandler)
    server.model = model
    server.model_name = Path(model_path).name
    print("Servidor local listo.")
    print("Abre esta dirección en el navegador: http://127.0.0.1:8000/")
    print("Para detenerlo, pulsa Ctrl+C en esta ventana.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServidor detenido.")
    finally:
        server.server_close()
    return 0

def main():
    carpeta = Path(__file__).resolve().parent
    if "--server" in sys.argv[1:]:
        return iniciar_servidor(carpeta)
    archivos = listar_imagenes(sys.argv[1:], carpeta)
    if not archivos:
        print("No encontré imágenes. Coloca imágenes junto al script o pasa una ruta como argumento.")
        return 1

    model_path = buscar_modelo(carpeta)
    if model_path:
        print(f"Modelo .pt encontrado: {Path(model_path).name}")
    else:
        print("No encontré .pt: se usará solo el detector conservador de color.")

    for ruta in archivos:
        imagen = leer_imagen(ruta)
        print("\n" + "=" * 72)
        print(f"Archivo: {Path(ruta).name}")
        if imagen is None:
            print("ERROR: no pude abrir la imagen.")
            continue
        datos = analizar_color(imagen)
        sugerencia = sugerencia_modelo(model_path, imagen) if model_path else None
        veredicto, fuente = describir(ruta, datos, sugerencia)
        print(f"Resultado: {veredicto}")
        print(f"Fuente: {fuente}")
        print(f"Tonos cálidos brillantes: {datos['ratio'] * 100:.2f} %")
        print(f"Mayor región conectada: {datos['largest_ratio'] * 100:.2f} %")
        print(f"Núcleo amarillo/naranja muy luminoso: {datos['core_ratio'] * 100:.2f} %")
        if sugerencia:
            if "error" in sugerencia:
                print(f"Aviso del modelo: {sugerencia['error']}")
            else:
                print(f"Predicción .pt: {sugerencia[0]} ({sugerencia[1] * 100:.1f} %)")
        print("Nota: color solamente no prueba incendio; troncos cortados y superficies ocres no son fuego.")
        if datos["fire"]:
            salida = guardar_anotada(ruta, imagen.copy(), datos, True)
            if salida:
                print(f"Imagen anotada: {salida}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
