"""
Algoritmo genético para ubicar una central nuclear modular pequeña (SMR)
sobre el corredor Río Paraná -> Río de la Plata -> costa bonaerense.

Cada individuo es un único parámetro t en [0, 1] que se traduce a un punto
de la polilínea BOUNDARY_POINTS. El fitness combina:
  1. Costo de construcción: tendido eléctrico hasta la estación
     transformadora más cercana + envío de materiales desde el puerto
     más cercano.
  2. Densidad poblacional máxima en dos anillos alrededor del sitio:
     0-2 millas (restricción dura) y 2-10 millas (restricción blanda).

Los costos por km, las coordenadas de estaciones, puertos y ciudades y la
población de las ciudades son valores de referencia aproximados.

Salidas (carpeta resultados_ga/):
  - mapa_optimo_<timestamp>.html
  - convergencia_ga_<timestamp>.png
  - resultado_optimo_<timestamp>.json
  - log_corridas.csv (acumula todas las corridas)

Requiere los tiles GHS-POP en la carpeta ghs_pop_tiles/ (junto a este archivo).
"""

import base64
import glob
import io
import json
import math
import os
import random
from datetime import datetime

import folium
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from PIL import Image
from pyproj import Transformer
from rasterio.io import MemoryFile
from rasterio.merge import merge as rio_merge
from rasterio.warp import reproject, Resampling


# ---------------------------------------------------------------------------
# Datos de entrada
# ---------------------------------------------------------------------------

# Vértices de la polilínea (lat, lon), de norte a sur
BOUNDARY_POINTS = [
    (-35.1365905, -57.3544973),
    (-35.0167268, -57.5254052),
    (-35.0003861, -57.5809272),
    (-34.9692072, -57.6278766),
    (-34.933399, -57.6892455),
    (-34.9279807, -57.7210887),
    (-34.9032936, -57.7709109),
    (-34.8309227, -57.8738219),
    (-34.8335299, -57.9338176),
    (-34.8250227, -57.9604034),
    (-34.7813095, -58.014524),
    (-34.7763746, -58.0564952),
    (-34.7516959, -58.11589),
    (-34.7478877, -58.1710791),
    (-34.7362504, -58.1976867),
    (-34.7158706, -58.2151523),
    (-34.6680319, -58.3001247),
    (-34.6472756, -58.3323971),
    (-34.5791088, -58.3784881),
    (-34.5318891, -58.4636322),
    (-34.4478454, -58.5189071),
    (-34.3318726, -58.4454823),
    (-34.2408164, -58.7747288),
    (-34.1854536, -58.9051914),
    (-34.1374432, -58.9865589),
    (-33.805457, -59.3583775),
    (-33.6164521, -59.7864601),
    (-33.3597667, -60.1739843),
    (-33.2400112, -60.3174346),
    (-33.1891116, -60.3261536),
    (-33.1967795, -60.3952127),
    (-33.1584321, -60.4724858),
    (-33.0187966, -60.6053528),
    (-32.773039, -60.7233048),
    (-32.5132633, -60.8154697),
    (-32.3697201, -60.8436288),
    (-32.2980595, -60.9073839),
    (-32.1724501, -60.9463356),
    (-32.0445135, -60.9412948),
    (-31.6663364, -60.7380477),
]

# (nombre, lat, lon, población aproximada)
CIUDADES = [
    ("Corrientes",     -27.47, -58.83,  360_000),
    ("Santa Fe",       -31.63, -60.70,  415_000),
    ("Rosario",        -32.95, -60.65, 1_000_000),
    ("San Nicolás",    -33.34, -60.21,  145_000),
    ("Zárate",         -34.10, -59.03,  130_000),
    ("Buenos Aires",   -34.61, -58.38, 3_000_000),
    ("La Plata",       -34.92, -57.95,  740_000),
    ("Mar del Plata",  -38.00, -57.55,  620_000),
]

# Estaciones transformadoras (red de alta tensión) para el costo de tendido.
# No son subestaciones de distribución.
ESTACIONES_TRANSFORMADORAS = [
    ("ET Resistencia", -27.45, -59.00),
    ("ET Santo Tomé", -31.67, -60.78),
    ("ET Rosario Oeste", -32.90, -60.75),
    ("ET Ezeiza", -34.85, -58.55),
    ("ET Necochea", -38.55, -58.74),
]

# Puertos para el costo de envío de materiales
PUERTOS = [
    ("Puerto Buenos Aires", -34.6023, -58.3644),
    ("Puerto Rosario", -32.9200, -60.6500),
    ("Puerto La Plata", -34.8500, -57.9000),
    ("Puerto Zárate", -34.0900, -59.0000),
    ("Puerto Santa Fe", -31.6500, -60.6800),
    ("Puerto Quequén / Necochea", -38.5500, -58.7000),
    ("Puerto San Nicolás", -33.3300, -60.2000),
]

# Aeropuertos: solo referencia visual en el mapa, no entran en el fitness
AEROPUERTOS = [
    ("Aeropuerto Rosario - Islas Malvinas", -32.9036, -60.7853),
    ("Aeropuerto Ezeiza - Ministro Pistarini", -34.8222, -58.5358),
    ("Aeroparque Jorge Newbery", -34.5592, -58.4156),
    ("Aeropuerto Santa Fe", -31.7104, -60.7810),
    ("Aeropuerto Mar del Plata", -37.9342, -57.5733),
]


# ---------------------------------------------------------------------------
# Funciones geográficas
# ---------------------------------------------------------------------------

def haversine(p1, p2):
    """Distancia en km entre dos puntos (lat, lon)."""
    R = 6371.0
    lat1, lon1 = map(math.radians, p1)
    lat2, lon2 = map(math.radians, p2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def destino(point, bearing_deg, distancia_km):
    """Punto al que se llega desde `point` con un rumbo (grados) y una distancia (km)."""
    R = 6371.0
    lat1, lon1 = map(math.radians, point)
    brng = math.radians(bearing_deg)
    d_r = distancia_km / R
    lat2 = math.asin(math.sin(lat1) * math.cos(d_r) + math.cos(lat1) * math.sin(d_r) * math.cos(brng))
    lon2 = lon1 + math.atan2(
        math.sin(brng) * math.sin(d_r) * math.cos(lat1),
        math.cos(d_r) - math.sin(lat1) * math.sin(lat2),
    )
    return (math.degrees(lat2), math.degrees(lon2))


def interpolate_point(t, boundary_points):
    """Convierte t en [0, 1] en un punto (lat, lon) de la polilínea."""
    t = min(max(t, 0.0), 1.0)
    n = len(boundary_points)
    idx_float = t * (n - 1)
    i = int(idx_float)
    frac = idx_float - i
    if i >= n - 1:
        return boundary_points[-1]
    p1, p2 = boundary_points[i], boundary_points[i + 1]
    lat = p1[0] + frac * (p2[0] - p1[0])
    lon = p1[1] + frac * (p2[1] - p1[1])
    return (lat, lon)


# ---------------------------------------------------------------------------
# Densidad poblacional (GHS-POP)
#
# Producto:   GHS_POP_E2030_GLOBE_R2023A_54009_1000_V1_0 (Comisión Europea / JRC)
# Proyección: World Mollweide (ESRI:54009), celdas de 1 km
# Descarga:   https://human-settlement.emergency.copernicus.eu/ghs_pop2023.php
# Tiles necesarios para el corredor: R13_C12, R13_C13, R14_C12, R14_C13
# (GHS_POP_E2030_GLOBE_R2023A_54009_1000_V1_0_R##_C##.tif). NoData = -200.
# ---------------------------------------------------------------------------

GHS_POP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ghs_pop_tiles")
GHS_POP_NODATA = -200.0

# Se define Mollweide a mano en lugar de usar el CRS embebido en los .tif:
# con algunas versiones de PROJ/GDAL "ESRI:54009" no se resuelve bien y el
# raster queda corrido hacia un costado.
GHS_POP_PROJ4 = "+proj=moll +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs +type=crs"

MILLA_KM = 1.60934
RADIO_ZONA_BAJA_KM = 2 * MILLA_KM     # ~3.22 km
RADIO_ZONA_MEDIA_KM = 10 * MILLA_KM   # ~16.09 km

UMBRAL_DENSIDAD_BAJA = 25.0    # hab/km2 tolerados dentro de las 2 millas
UMBRAL_DENSIDAD_MEDIA = 400.0  # hab/km2 tolerados entre 2 y 10 millas

_ghs_pop_dataset = None
_ghs_pop_memfile = None  # hay que mantener viva la referencia al MemoryFile
_transformer_wgs84_a_mollweide = None


def _obtener_raster_ghs_pop():
    """Arma una sola vez un mosaico en memoria con todos los .tif de
    GHS_POP_DIR y lo devuelve junto con el transformador WGS84 -> Mollweide."""
    global _ghs_pop_dataset, _ghs_pop_memfile, _transformer_wgs84_a_mollweide
    if _ghs_pop_dataset is None:
        archivos = sorted(glob.glob(os.path.join(GHS_POP_DIR, "*.tif")))
        if not archivos:
            raise FileNotFoundError(
                f"No se encontraron tiles GHS-POP (.tif) en '{GHS_POP_DIR}'.\n"
                "Colocá ahí los tiles que cubren el corredor "
                "(R13_C12, R13_C13, R14_C12, R14_C13), descargados de "
                "https://human-settlement.emergency.copernicus.eu/ghs_pop2023.php"
            )

        datasets_origen = [rasterio.open(f) for f in archivos]
        mosaico, transform = rio_merge(datasets_origen, nodata=GHS_POP_NODATA)
        crs_original = datasets_origen[0].crs
        for ds in datasets_origen:
            ds.close()

        crs = GHS_POP_PROJ4
        print(f"[GHS-POP] CRS embebido en los tiles: {crs_original}")
        print(f"[GHS-POP] CRS forzado para la reproyección: {crs}")

        perfil = dict(
            driver="GTiff",
            height=mosaico.shape[1],
            width=mosaico.shape[2],
            count=1,
            dtype=mosaico.dtype,
            crs=crs,
            transform=transform,
            nodata=GHS_POP_NODATA,
        )
        _ghs_pop_memfile = MemoryFile()
        with _ghs_pop_memfile.open(**perfil) as dst:
            dst.write(mosaico)
        _ghs_pop_dataset = _ghs_pop_memfile.open()

        # always_xy=True: orden (lon, lat) -> (x, y)
        _transformer_wgs84_a_mollweide = Transformer.from_crs(
            "EPSG:4326", _ghs_pop_dataset.crs, always_xy=True
        )

        print(f"[GHS-POP] Mosaico armado a partir de {len(archivos)} tile(s): "
              f"{[os.path.basename(a) for a in archivos]}")
        print(f"[GHS-POP] Forma del mosaico: {mosaico.shape[1]}x{mosaico.shape[2]} px")

    return _ghs_pop_dataset, _transformer_wgs84_a_mollweide


def densidad_en_puntos(points):
    """Densidad (hab/km2) en cada punto (lat, lon), leyendo la celda del raster.
    Como cada celda mide ~1 km2, la población de la celda es la densidad."""
    dataset, transformer = _obtener_raster_ghs_pop()
    xs, ys = transformer.transform([lon for _, lon in points], [lat for lat, _ in points])
    valores = [v[0] for v in dataset.sample(zip(xs, ys))]
    return [0.0 if (v is None or v <= GHS_POP_NODATA) else max(float(v), 0.0) for v in valores]


def densidad_maxima_en_anillo(point, r_min_km, r_max_km, n_radios=5, n_angulos=16):
    """Densidad máxima (peor caso) dentro del anillo [r_min, r_max] km,
    muestreando n_radios x n_angulos puntos más el centro del sitio."""
    if r_min_km <= 0:
        radios = np.linspace(max(r_max_km * 0.15, 0.1), r_max_km, n_radios)
    else:
        radios = np.linspace(r_min_km, r_max_km, n_radios)
    puntos = [point]
    for r in radios:
        for ang in np.linspace(0, 360, n_angulos, endpoint=False):
            puntos.append(destino(point, ang, r))
    return max(densidad_en_puntos(puntos))


def evaluar_restriccion_poblacional(point):
    """Devuelve (fitness_poblacional en [0, 1], cumple_restriccion_dura, detalle).

      - 0-2 millas:  restricción dura
      - 2-10 millas: restricción blanda
      - >10 millas:  sin restricción

    La zona de 0-2 millas se muestrea con más puntos porque es la que
    decide si el sitio es viable.
    """
    dens_zona_baja = densidad_maxima_en_anillo(
        point, 0.0, RADIO_ZONA_BAJA_KM, n_radios=6, n_angulos=20
    )
    dens_zona_media = densidad_maxima_en_anillo(
        point, RADIO_ZONA_BAJA_KM, RADIO_ZONA_MEDIA_KM, n_radios=5, n_angulos=16
    )

    cumple_restriccion_dura = dens_zona_baja <= UMBRAL_DENSIDAD_BAJA

    if cumple_restriccion_dura:
        f_zona_baja = 1.0
    else:
        # Decae rápido (factor 8) para que ningún ahorro de costo compense
        # haber superado el umbral de la zona dura.
        exceso_relativo = (dens_zona_baja - UMBRAL_DENSIDAD_BAJA) / UMBRAL_DENSIDAD_BAJA
        f_zona_baja = math.exp(-8.0 * exceso_relativo)

    if dens_zona_media <= UMBRAL_DENSIDAD_MEDIA:
        f_zona_media = 1.0
    else:
        f_zona_media = math.exp(-(dens_zona_media - UMBRAL_DENSIDAD_MEDIA) / UMBRAL_DENSIDAD_MEDIA)

    # La zona dura pesa más
    fitness = 0.65 * f_zona_baja + 0.35 * f_zona_media
    detalle = dict(
        dens_zona_baja=dens_zona_baja,
        dens_zona_media=dens_zona_media,
        f_zona_baja=f_zona_baja,
        f_zona_media=f_zona_media,
        cumple_restriccion_dura=cumple_restriccion_dura,
    )
    return fitness, cumple_restriccion_dura, detalle


# ---------------------------------------------------------------------------
# Costo de construcción
# ---------------------------------------------------------------------------

COSTO_TENDIDO_USD_KM = 300_000             # USD/km de línea de alta tensión
COSTO_TRANSPORTE_MARITIMO_USD_KM = 5_000   # USD/km desde el puerto


def costo_tendido_electrico(point):
    d_red = min(haversine(point, (e[1], e[2])) for e in ESTACIONES_TRANSFORMADORAS)
    return d_red * COSTO_TENDIDO_USD_KM


def costo_envio_materiales(point):
    d_puerto = min(haversine(point, (p[1], p[2])) for p in PUERTOS)
    return d_puerto * COSTO_TRANSPORTE_MARITIMO_USD_KM


def costo_construccion_bruto(point):
    """Costo total estimado en USD (tendido + envío de materiales)."""
    return costo_tendido_electrico(point) + costo_envio_materiales(point)


def normalizar(valores):
    vmin, vmax = min(valores), max(valores)
    rango = (vmax - vmin) if (vmax - vmin) > 1e-9 else 1e-9
    return [(v - vmin) / rango for v in valores]


# ---------------------------------------------------------------------------
# Algoritmo genético
# ---------------------------------------------------------------------------

POP_SIZE = 60
N_GENERACIONES = 100
PROB_CRUZA = 0.75
PROB_MUTACION = 0.25
SIGMA_MUTACION = 0.25
TAM_TORNEO = 2

PESOS_FITNESS = dict(costo=0.75, poblacional=0.25)

# Si el sitio viola la restricción dura (0-2 millas), su fitness total se
# multiplica por este factor. No se usa 0 para que el AG conserve algo de
# gradiente y pueda alejar a la población de la zona prohibida.
PENALIZACION_RESTRICCION_DURA = 0.02


def crear_individuo():
    return {"t": random.random()}


def evaluar_poblacion(poblacion):
    """Devuelve (fitness_total, detalle) para cada individuo de la población.

    El costo se normaliza (min-max) dentro de la población de la generación:
    0 = el más barato, 1 = el más caro.
    """
    puntos = [interpolate_point(ind["t"], BOUNDARY_POINTS) for ind in poblacion]

    costos_brutos = [costo_construccion_bruto(p) for p in puntos]
    costos_norm = normalizar(costos_brutos)

    fitness_total = []
    detalle = []
    for p, c_norm in zip(puntos, costos_norm):
        f_costo = 1 - c_norm
        f_poblacional, cumple_restriccion_dura, det_pob = evaluar_restriccion_poblacional(p)
        f = (PESOS_FITNESS["costo"] * f_costo
             + PESOS_FITNESS["poblacional"] * f_poblacional)
        if not cumple_restriccion_dura:
            f *= PENALIZACION_RESTRICCION_DURA
        fitness_total.append(f)
        detalle.append(dict(
            punto=p,
            f_costo=f_costo,
            f_poblacional=f_poblacional,
            costo_construccion_usd=costo_construccion_bruto(p),
            **det_pob,
        ))
    return fitness_total, detalle


def seleccion_torneo(poblacion, fitness, k=TAM_TORNEO):
    participantes = random.sample(list(zip(poblacion, fitness)), k)
    participantes.sort(key=lambda x: x[1], reverse=True)
    return participantes[0][0]


def cruza(p1, p2):
    """Cruza aritmética: cada hijo es una combinación lineal de los padres."""
    if random.random() > PROB_CRUZA:
        return dict(p1), dict(p2)
    alpha = random.random()
    t1 = alpha * p1["t"] + (1 - alpha) * p2["t"]
    t2 = alpha * p2["t"] + (1 - alpha) * p1["t"]
    return {"t": min(max(t1, 0), 1)}, {"t": min(max(t2, 0), 1)}


def mutacion(ind, sigma):
    """Mutación gaussiana sobre t, acotada a [0, 1]."""
    if random.random() < PROB_MUTACION:
        nuevo_t = ind["t"] + random.gauss(0, sigma)
        ind["t"] = min(max(nuevo_t, 0.0), 1.0)
    return ind


def correr_ga():
    poblacion = [crear_individuo() for _ in range(POP_SIZE)]
    historia_mejor = []
    historia_promedio = []
    historia_peor = []
    historia_std = []
    mejor_global = None
    mejor_fitness_global = -1

    for gen in range(N_GENERACIONES):
        fitness, detalle = evaluar_poblacion(poblacion)

        idx_mejor = int(np.argmax(fitness))
        if fitness[idx_mejor] > mejor_fitness_global:
            mejor_fitness_global = fitness[idx_mejor]
            mejor_global = dict(poblacion[idx_mejor])
            mejor_global["detalle"] = detalle[idx_mejor]

        historia_mejor.append(fitness[idx_mejor])
        historia_promedio.append(float(np.mean(fitness)))
        historia_peor.append(float(np.min(fitness)))
        historia_std.append(float(np.std(fitness)))

        nueva_poblacion = []
        while len(nueva_poblacion) < POP_SIZE:
            padre1 = seleccion_torneo(poblacion, fitness)
            padre2 = seleccion_torneo(poblacion, fitness)
            hijo1, hijo2 = cruza(padre1, padre2)
            hijo1 = mutacion(hijo1, SIGMA_MUTACION)
            hijo2 = mutacion(hijo2, SIGMA_MUTACION)
            nueva_poblacion.append(hijo1)
            if len(nueva_poblacion) < POP_SIZE:
                nueva_poblacion.append(hijo2)

        poblacion = nueva_poblacion

    return (mejor_global, mejor_fitness_global, historia_mejor, historia_promedio,
            historia_peor, historia_std, poblacion)


# ---------------------------------------------------------------------------
# Gráfico de convergencia
# ---------------------------------------------------------------------------

def graficar_convergencia(historia_mejor, historia_promedio, historia_peor, historia_std,
                          mejor_fitness_global=None, path="convergencia_ga.png"):
    generaciones = np.arange(1, len(historia_mejor) + 1)
    promedio = np.array(historia_promedio)
    std = np.array(historia_std)
    plt.rcParams.update({
        "font.size": 14,          # tamaño base
        "axes.labelsize": 16,     # títulos de los ejes
        "axes.titlesize": 18,     # título del gráfico
        "xtick.labelsize": 14,    # números del eje X
        "ytick.labelsize": 14,    # números del eje Y
        "legend.fontsize": 13,
    })
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.fill_between(generaciones, np.clip(promedio - std, 0, 1), np.clip(promedio + std, 0, 1),
                    color="#e65100", alpha=0.15, label="Promedio ± 1 desvío estándar")

    ax.plot(generaciones, historia_mejor, label="Mejor fitness", linewidth=2.2, color="#1b5e20")
    ax.plot(generaciones, historia_promedio, label="Fitness promedio", linewidth=2,
            linestyle="--", color="#e65100")
    ax.plot(generaciones, historia_peor, label="Peor fitness", linewidth=1.6,
            linestyle=":", color="#b71c1c")

    ax.set_xlabel("Generación")
    ax.set_ylabel("Fitness (0 - 1)")
    ax.set_title("Convergencia del Algoritmo Genético\nUbicación óptima - Central Nuclear Modular")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    texto = (f"Última generación:\n"
             f"Mejor:    {historia_mejor[-1]:.3f}\n"
             f"Promedio: {historia_promedio[-1]:.3f}\n"
             f"Peor:     {historia_peor[-1]:.3f}\n"
             f"Desvío σ: {historia_std[-1]:.3f}")
    if mejor_fitness_global is not None:
        # Mejor individuo de TODA la corrida (puede ser de otra generación,
        # no necesariamente de la última) -> es el que se reporta como resultado.
        texto += f"\nMejor histórico: {mejor_fitness_global:.3f}"
    ax.text(0.015, 0.02, texto, transform=ax.transAxes, fontsize=9.5,
            verticalalignment="bottom", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.9))

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Mapa HTML con el punto óptimo
# ---------------------------------------------------------------------------

# Capa de referencia externa (apagada por defecto): densidad GPWv4 2020 de
# NASA GIBS. Es otro dataset que el usado en el fitness, por eso puede no
# coincidir con las zonas que el AG consideró de alta o baja densidad.
GIBS_GPW_LAYER = "GPW_Population_Density_2020"
GIBS_GPW_TILEMATRIXSET = "GoogleMapsCompatible_Level7"
GIBS_GPW_TILES_URL = (
    f"https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/"
    f"{GIBS_GPW_LAYER}/default/{GIBS_GPW_TILEMATRIXSET}/{{z}}/{{y}}/{{x}}.png"
)
GIBS_GPW_ATTR = (
    "Population density: NASA GIBS / CIESIN-SEDAC GPWv4 (2020) &mdash; "
    "<a href='https://earthdata.nasa.gov/gibs' target='_blank'>NASA EOSDIS GIBS</a>"
)

# Rangos de la leyenda alineados con los umbrales del modelo
DENSIDAD_BINS = [0, 1, 5, 25, UMBRAL_DENSIDAD_BAJA, 100, UMBRAL_DENSIDAD_MEDIA, 1000, 100000]
DENSIDAD_COLORES = [
    "#ffffe0", "#ffe0a3", "#ffb347", "#ff8c42",
    "#f4622e", "#d8371c", "#b0170c", "#6b0000",
]


def _bounds_de_interes(padding_km=25.0):
    """Caja (lon_min, lat_min, lon_max, lat_max) que cubre el corredor y los
    marcadores, con margen, para no reproyectar el raster completo."""
    lats = [p[0] for p in BOUNDARY_POINTS] + [c[1] for c in CIUDADES] \
        + [e[1] for e in ESTACIONES_TRANSFORMADORAS] + [p_[1] for p_ in PUERTOS]
    lons = [p[1] for p in BOUNDARY_POINTS] + [c[2] for c in CIUDADES] \
        + [e[2] for e in ESTACIONES_TRANSFORMADORAS] + [p_[2] for p_ in PUERTOS]
    lat_min, lat_max = min(lats), max(lats)
    lon_min, lon_max = min(lons), max(lons)
    pad_deg = padding_km / 100.0  # a estas latitudes 1 grado ~ 100-110 km
    return (lon_min - pad_deg, lat_min - pad_deg, lon_max + pad_deg, lat_max + pad_deg)


def _generar_overlay_densidad_ghs_pop(resolucion_deg=0.01):
    """Reproyecta el raster GHS-POP usado en el fitness (Mollweide) a Web
    Mercator y lo devuelve como (data_uri_png, bounds) para ImageOverlay."""
    dataset, _ = _obtener_raster_ghs_pop()
    src_crs = dataset.crs

    # La grilla de destino se arma en EPSG:3857 porque Leaflet estira la
    # imagen entre sus dos esquinas en Mercator. En EPSG:4326 la imagen
    # quedaría distorsionada en vertical (unos 15-20 km hacia el sur en
    # la zona de Buenos Aires).
    dst_crs = "EPSG:3857"
    west, south, east, north = _bounds_de_interes()

    transformer_a_merc = Transformer.from_crs("EPSG:4326", dst_crs, always_xy=True)
    x_min, y_min = transformer_a_merc.transform(west, south)
    x_max, y_max = transformer_a_merc.transform(east, north)

    resolucion_m = resolucion_deg * 111_320.0
    width = max(int((x_max - x_min) / resolucion_m), 1)
    height = max(int((y_max - y_min) / resolucion_m), 1)
    dst_transform = rasterio.transform.from_bounds(x_min, y_min, x_max, y_max, width, height)

    src_data = dataset.read(1)
    dst_data = np.full((height, width), GHS_POP_NODATA, dtype=np.float32)

    reproject(
        source=src_data,
        destination=dst_data,
        src_transform=dataset.transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        src_nodata=GHS_POP_NODATA,
        dst_nodata=GHS_POP_NODATA,
        resampling=Resampling.bilinear,
    )

    dst_data = np.where(dst_data <= GHS_POP_NODATA, 0.0, dst_data)

    cmap = mcolors.ListedColormap(DENSIDAD_COLORES)
    norm = mcolors.BoundaryNorm(DENSIDAD_BINS, cmap.N)
    rgba = (cmap(norm(dst_data)) * 255).astype(np.uint8)
    # Transparente donde la densidad es casi cero, para ver el mapa base
    rgba[..., 3] = np.where(dst_data <= 0.5, 0, 190).astype(np.uint8)

    img = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data_uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    # ImageOverlay espera las esquinas en lat/lon
    bounds_folium = [[south, west], [north, east]]
    return data_uri, bounds_folium


def _leyenda_densidad_ghs_pop_html():
    filas = "".join(
        f'<div style="display:flex;align-items:center;margin:1px 0;">'
        f'<span style="width:14px;height:14px;background:{color};display:inline-block;'
        f'margin-right:6px;border:1px solid #999;"></span>'
        f'<span style="font-size:11px;">{lo:g} - {hi:g} hab/km²</span></div>'
        for lo, hi, color in zip(DENSIDAD_BINS[:-1], DENSIDAD_BINS[1:], DENSIDAD_COLORES)
    )
    return f"""
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999;
                background: white; padding: 8px 10px; border: 1px solid #888;
                border-radius: 6px; box-shadow: 0 1px 4px rgba(0,0,0,0.3);">
      <div style="font-weight:bold; font-size:12px; margin-bottom:4px;">
        Densidad poblacional (GHS-POP, misma fuente que el fitness)
      </div>
      {filas}
    </div>
    """


def generar_mapa(mejor, poblacion_final, fitness_total, path="mapa_optimo.html"):
    punto_optimo = mejor["detalle"]["punto"]

    # Se usa Esri World Street Map como mapa base en vez de OpenStreetMap:
    # los servidores de tiles de OSM son de uso voluntario y bloquean con
    # 403 el acceso "genérico" de Leaflet/Folium sin configuración
    # específica (ver osm.wiki/Blocked); CartoDB, la alternativa más
    # habitual, ahora también exige API key. Esri no requiere ninguna.
    m = folium.Map(location=punto_optimo, zoom_start=6, tiles="Esri.WorldStreetMap")

    data_uri, bounds_overlay = _generar_overlay_densidad_ghs_pop()
    folium.raster_layers.ImageOverlay(
        image=data_uri,
        bounds=bounds_overlay,
        name="Densidad poblacional (GHS-POP - misma fuente que el fitness)",
        opacity=0.85,
        interactive=False,
        cross_origin=False,
        zindex=1,
    ).add_to(m)

    folium.TileLayer(
        tiles=GIBS_GPW_TILES_URL,
        attr=GIBS_GPW_ATTR,
        name="Referencia externa: NASA GIBS GPWv4 2020 (otro dataset, no el usado en el fitness)",
        overlay=True,
        control=True,
        show=False,
        min_zoom=1,
        max_zoom=18,
        max_native_zoom=7,
    ).add_to(m)

    m.get_root().html.add_child(folium.Element(_leyenda_densidad_ghs_pop_html()))

    # Corredor disponible
    folium.PolyLine(
        BOUNDARY_POINTS, color="blue", weight=3, opacity=0.6,
        tooltip="Corredor disponible (Río Paraná -> Mar del Plata)"
    ).add_to(m)

    for nombre, lat, lon, poblacion in CIUDADES:
        folium.CircleMarker(
            location=(lat, lon), radius=5, color="#6a1b9a", fill=True,
            fill_color="#6a1b9a", fill_opacity=0.8,
            popup=f"Ciudad: {nombre} (~{poblacion:,} hab)".replace(",", "."),
        ).add_to(m)

    for nombre, lat, lon in ESTACIONES_TRANSFORMADORAS:
        folium.Marker(
            location=(lat, lon),
            icon=folium.Icon(color="orange", icon="bolt", prefix="fa"),
            popup=f"Estación Transformadora: {nombre}",
        ).add_to(m)

    for nombre, lat, lon in PUERTOS:
        folium.Marker(
            location=(lat, lon),
            icon=folium.Icon(color="blue", icon="anchor", prefix="fa"),
            popup=f"Puerto: {nombre}",
        ).add_to(m)

    for nombre, lat, lon in AEROPUERTOS:
        folium.Marker(
            location=(lat, lon),
            icon=folium.Icon(color="cadetblue", icon="plane", prefix="fa"),
            popup=f"Aeropuerto: {nombre}",
        ).add_to(m)

    # Posiciones de la última generación (dispersión de la población)
    for ind in poblacion_final:
        p = interpolate_point(ind["t"], BOUNDARY_POINTS)
        folium.CircleMarker(
            location=p, radius=2, color="gray", fill=True,
            fill_opacity=0.4,
        ).add_to(m)

    # Anillos de restricción poblacional alrededor del punto óptimo
    folium.Circle(
        location=punto_optimo, radius=RADIO_ZONA_BAJA_KM * 1000,
        color="#c62828", weight=2, fill=True, fill_opacity=0.05,
        tooltip="Zona 0-2 millas: densidad debe ser muy baja",
    ).add_to(m)
    folium.Circle(
        location=punto_optimo, radius=RADIO_ZONA_MEDIA_KM * 1000,
        color="#f9a825", weight=2, fill=True, fill_opacity=0.03,
        tooltip="Zona 2-10 millas: densidad media tolerada",
    ).add_to(m)

    detalle = mejor["detalle"]
    cumple = detalle.get("cumple_restriccion_dura", True)
    estado_restriccion = (
        "CUMPLE restricción dura (0-2 mi)" if cumple
        else "VIOLA restricción dura (0-2 mi)"
    )
    popup_html = (
        f"<b>Ubicación óptima SMR</b><br>"
        f"Lat: {punto_optimo[0]:.4f}, Lon: {punto_optimo[1]:.4f}<br>"
        f"<hr>"
        f"Fitness costo construcción: {detalle['f_costo']:.3f}<br>"
        f"&nbsp;&nbsp;Costo estimado: USD {detalle['costo_construccion_usd']:,.0f}<br>"
        f"Fitness poblacional: {detalle['f_poblacional']:.3f}<br>"
        f"&nbsp;&nbsp;Densidad máx. 0-2 mi: {detalle['dens_zona_baja']:.1f} hab/km²<br>"
        f"&nbsp;&nbsp;Densidad máx. 2-10 mi: {detalle['dens_zona_media']:.1f} hab/km²<br>"
        f"&nbsp;&nbsp;{estado_restriccion}<br>"
        f"<hr>"
        f"<b>Fitness total: {fitness_total:.3f}</b>"
    ).replace(",", ".")
    folium.Marker(
        location=punto_optimo,
        icon=folium.Icon(color="red" if cumple else "black", icon="star", prefix="fa"),
        popup=folium.Popup(popup_html, max_width=320),
        tooltip="Ubicación óptima",
    ).add_to(m)

    folium.LayerControl().add_to(m)

    m.save(path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Sin semilla fija: cada corrida es distinta. Para reproducir una corrida
    # exacta, reemplazar por random.seed(N) y np.random.seed(N).
    random.seed()
    np.random.seed()

    (mejor, mejor_fitness_total, historia_mejor, historia_promedio,
     historia_peor, historia_std, poblacion_final) = correr_ga()

    print("=== RESULTADO ===")
    print(f"t óptimo (posición en la línea frontera): {mejor['t']:.4f}")
    print(f"Punto óptimo (lat, lon): {mejor['detalle']['punto']}")
    print(f"Fitness costo construcción: {mejor['detalle']['f_costo']:.3f}  "
          f"(USD {mejor['detalle']['costo_construccion_usd']:,.0f})")
    print(f"Fitness poblacional:        {mejor['detalle']['f_poblacional']:.3f}  "
          f"(dens. 0-2mi: {mejor['detalle']['dens_zona_baja']:.1f} hab/km², "
          f"dens. 2-10mi: {mejor['detalle']['dens_zona_media']:.1f} hab/km²)")
    print(f"Fitness total:              {mejor_fitness_total:.3f}")

    # Archivos con timestamp para no pisar corridas anteriores
    OUTPUT_DIR = "resultados_ga"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    path_mapa = os.path.join(OUTPUT_DIR, f"mapa_optimo_{timestamp}.html")
    path_grafico = os.path.join(OUTPUT_DIR, f"convergencia_ga_{timestamp}.png")
    path_json = os.path.join(OUTPUT_DIR, f"resultado_optimo_{timestamp}.json")

    graficar_convergencia(historia_mejor, historia_promedio, historia_peor, historia_std,
                          mejor_fitness_global=mejor_fitness_total, path=path_grafico)
    generar_mapa(mejor, poblacion_final, mejor_fitness_total, path=path_mapa)

    with open(path_json, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": timestamp,
            "t": mejor["t"],
            "punto": mejor["detalle"]["punto"],
            "fitness_costo": mejor["detalle"]["f_costo"],
            "costo_construccion_usd": mejor["detalle"]["costo_construccion_usd"],
            "fitness_poblacional": mejor["detalle"]["f_poblacional"],
            "densidad_zona_0_2mi": mejor["detalle"]["dens_zona_baja"],
            "densidad_zona_2_10mi": mejor["detalle"]["dens_zona_media"],
            "fitness_total": mejor_fitness_total,
        }, f, ensure_ascii=False, indent=2)

    # Log acumulado de todas las corridas
    path_log = os.path.join(OUTPUT_DIR, "log_corridas.csv")
    existe_log = os.path.exists(path_log)
    with open(path_log, "a", encoding="utf-8") as f:
        if not existe_log:
            f.write("timestamp,t,lat,lon,fitness_costo,costo_construccion_usd,"
                    "fitness_poblacional,densidad_0_2mi,densidad_2_10mi,"
                    "fitness_mejor,fitness_promedio,fitness_peor,fitness_std\n")
        lat, lon = mejor["detalle"]["punto"]
        f.write(f"{timestamp},{mejor['t']:.6f},{lat:.6f},{lon:.6f},"
                f"{mejor['detalle']['f_costo']:.4f},{mejor['detalle']['costo_construccion_usd']:.2f},"
                f"{mejor['detalle']['f_poblacional']:.4f},"
                f"{mejor['detalle']['dens_zona_baja']:.2f},{mejor['detalle']['dens_zona_media']:.2f},"
                f"{mejor_fitness_total:.4f},"
                f"{historia_promedio[-1]:.4f},{historia_peor[-1]:.4f},{historia_std[-1]:.4f}\n")

    print(f"\nArchivos generados en '{OUTPUT_DIR}/':")
    print(f"  - {path_mapa}")
    print(f"  - {path_grafico}")
    print(f"  - {path_json}")
    print(f"  - {path_log} (acumula todas las corridas)")