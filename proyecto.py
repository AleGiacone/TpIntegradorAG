"""
Datos base del proyecto: localizacion optima de un reactor modular pequeno
(SMR) mediante algoritmo genetico, entre 6 ciudades candidatas de la
Patagonia Argentina: Comodoro Rivadavia, Trelew, Puerto Madryn, Rawson,
Caleta Olivia y San Carlos de Bariloche.

Todas las coordenadas son aproximadas (centro urbano). Los costos y
caudales estan basados en datos reales cuando fue posible (fuentes citadas
en el informe) y se marcan como estimaciones cuando no hay una cifra
oficial unica disponible.
"""

import math

# ---------------------------------------------------------------------------
# Ciudades candidatas: (nombre, lat, lon, poblacion censo 2022 INDEC)
# ---------------------------------------------------------------------------
CIUDADES = [
    ("Comodoro Rivadavia",      -45.8641, -67.4966, 201_854),
    ("Trelew",                  -43.2489, -65.3051, 106_214),
    ("Puerto Madryn",           -42.7692, -65.0385, 102_143),
    ("Rawson",                  -43.3002, -65.1023,  38_129),
    ("Caleta Olivia",           -46.4386, -67.5259,  56_298),
    ("San Carlos de Bariloche", -41.1456, -71.3082, 135_755),
]

# ---------------------------------------------------------------------------
# Objetivo 1 - Agua: fuentes conocidas (nombre, lat, lon, caudal relativo 0-1)
# ---------------------------------------------------------------------------
FUENTES_AGUA = [
    ("Rio Chubut - Trelew",      -43.2489, -65.3051, 0.9),
    ("Rio Chubut - Rawson",      -43.3002, -65.1023, 0.9),
    ("Lago Musters",             -45.1833, -69.0500, 0.6),
    ("Lago Nahuel Huapi",        -41.1456, -71.3082, 1.0),
    ("Costa Atlantica - Madryn", -42.7692, -65.0385, 0.5),
]

# ---------------------------------------------------------------------------
# Objetivo 2 - Infraestructura electrica: subestaciones/nodos conocidos
# ---------------------------------------------------------------------------
SUBESTACIONES = [
    ("ET Puerto Madryn",       -42.7500, -65.0200, 500),
    ("ET Comodoro Oeste",      -45.8500, -67.5000, 500),
    ("ET Pico Truncado",       -46.7900, -67.9300, 500),
    ("Nodo Alicura-Bariloche", -41.2200, -70.7500, 132),
]
# Costo estimado de construccion de linea, USD/km, segun tension (kV)
COSTO_USD_POR_KM_ELECTRICO = {132: 500_000, 500: 900_000}

# ---------------------------------------------------------------------------
# Objetivo 3 - Rutas y puertos (para materiales importados)
# ---------------------------------------------------------------------------
RUTAS = [
    ("RN3 - Madryn",        -42.9500, -65.1500),
    ("RN3 - Trelew",         -43.2500, -65.2000),
    ("RN3 - Comodoro",       -45.8600, -67.4800),
    ("RN3 - Caleta Olivia",  -46.4400, -67.5300),
    ("RN40 - Bariloche",     -41.1500, -71.2000),
]

PUERTOS = [
    ("Puerto Madryn",             -42.7700, -65.0400),
    ("Puerto Comodoro Rivadavia", -45.8650, -67.4600),
    ("Puerto Caleta Olivia",      -46.4350, -67.5200),
    ("Puerto Rawson",             -43.3000, -65.1000),
    ("San Antonio Este",          -40.7900, -64.9500),
]

COSTO_CAMINO_USD_KM = 200_000          # construccion de acceso vial nuevo
COSTO_TRANSPORTE_USD_TON_KM = 0.12     # transporte terrestre de carga pesada
TONELADAS_MATERIAL_IMPORTADO = 5000    # carga total estimada a transportar

# ---------------------------------------------------------------------------
# Objetivo 4 - Poblacion abastecida
# ---------------------------------------------------------------------------
RADIO_MAX_POBLACION_KM = 150

# ---------------------------------------------------------------------------
# Espacio de busqueda continuo del algoritmo genetico (bounding box que
# contiene a las 6 ciudades candidatas, con un margen de seguridad)
# ---------------------------------------------------------------------------
LAT_MIN, LAT_MAX = -46.8, -40.8
LON_MIN, LON_MAX = -71.6, -64.7


def haversine(lat1, lon1, lat2, lon2):
    """Distancia en linea recta (km) entre dos puntos (lat, lon)."""
    R = 6371  # radio de la Tierra en km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))
