"""
Algoritmo Genético para determinar la ubicación óptima de una
Central Nuclear Modular Pequeña (SMR) en la región del corredor
Río Paraná -> Mar del Plata.

CAMBIO respecto a la versión anterior (ga_central_nuclear_v2_poblacion.py):
  - El riesgo poblacional YA NO se calcula con una lista fija de ~30
    centros poblados (partidos/ciudades con su población del Censo
    2022), sino con datos de POBLACIÓN GRILLADA REAL: GHS-POP
    (Global Human Settlement Population Grid, Comisión Europea /
    JRC), proyección Mollweide (ESRI:54009), resolución 1 km,
    escenario 2030 (E2030).
  - Para cada individuo del GA, se calcula la población real que
    vive dentro de un radio de RADIO_RIESGO_KM alrededor del punto,
    sumando directamente los píxeles del raster (cada píxel = cantidad
    de personas estimada en esa celda de 1x1 km). Esto resuelve el
    problema de fondo de las versiones anteriores: ya no importa si
    hay "una ciudad" cerca o no, se ve la densidad poblacional real,
    haya o no un nombre de localidad asociado (loteos, barrios
    cerrados, zonas rurales densas, etc. quedan reflejados igual).
  - Los tiles GHS-POP se leen de la carpeta GHS_POP_DIR (ver abajo);
    hay que colocar ahí TODOS los .tif que cubran el rectángulo de
    búsqueda. Para este corredor (aprox. lat -28 a -37, lon -55 a
    -63) hacen falta los tiles: R13_C12, R13_C13, R14_C12, R14_C13
    (nomenclatura GHS_POP_E2030_GLOBE_R2023A_54009_1000_V1_0_R##_C##.tif).
  - Se arma un mosaico único (rasterio.merge) al arrancar el script,
    y las consultas de población por radio son muy rápidas
    (~0.03 ms cada una), así que no hay problema de performance aun
    con miles de evaluaciones durante la corrida del GA.

Criterios de fitness (solo se evalúan si el punto está en tierra):
  1. Costo eléctrico       (cercanía a red de transmisión y a centros de demanda)
  2. Riesgo sísmico         (cercanía a fallas / zonificación INPRES, proxy simplificado)
  3. Riesgo poblacional     (población real dentro de un radio, según GHS-POP;
                             exclusión dura < RADIO_MIN_DURO_KM)
  4. Cercanía al corredor hídrico (toma de agua de refrigeración, ya en tierra)

Salidas:
  - mapa_optimo_<timestamp>.html    -> mapa interactivo (Leaflet/Folium)
  - convergencia_ga_<timestamp>.png -> evolución del fitness por generación
  - resultado_optimo_<timestamp>.json
  - log_corridas.csv                -> acumula todas las corridas
"""

import math
import random
import json
import os
import glob
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import folium
from global_land_mask import globe
import rasterio
from rasterio import Affine
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.merge import merge as rio_merge
from pyproj import Transformer


# ============================================================
# DATOS DE ENTRADA (REEMPLAZAR CON TUS DATOS REALES)
# ============================================================

BOUNDARY_POINTS = [
  (-35.146836,-57.3536387 ),	
  (-35.0167268,-57.5254052 ),
  (-35.0003861,-57.5809272 ),
  (-34.9692072,-57.6278766 ),
  (-34.933399,-57.6892455 )	,
  (-34.9279807,-57.7210887 ),
  (-34.9032936,-57.7709109 ),
  (-34.8309227,-57.8738219 ),
  (-34.8335299,-57.9338176 ),
  (-34.8250227,-57.9604034 ),
  (-34.7813095,-58.014524 ),
  (-34.7763746,-58.0564952 ),
  (-34.7516959,-58.11589 ),
  (-34.7478877,-58.1710791 ),
  (-34.7362504,-58.1976867 ),
  (-34.7158706,-58.2151523 ),
  (-34.6680319,-58.3001247 ),
  (-34.6472756,-58.3323971 ),
  (-34.5791088,-58.3784881 ),
  (-34.5318891,-58.4636322 ),
  (-34.4478454,-58.5189071 ),
  (-34.3318726,-58.4454823 ),
  (-34.2408164,-58.7747288 ),
  (-34.1854536,-58.9051914 ),
  (-34.1374432,-58.9865589 ),
  (-33.805457,-59.3583775 ),
  (-32.9425889,-60.6216154 ),
  (-32.5105103,-60.775424 ),
  (-31.7284854,-60.636435 ),
  (-31.1473454,-59.9325347 ),
  (-30.0171303,-59.6029448 ),
]

CIUDADES = [
    ("Corrientes", -27.47, -58.83),
    ("Santa Fe", -31.63, -60.70),
    ("Rosario", -32.95, -60.65),
    ("San Nicolás", -33.34, -60.21),
    ("Zárate", -34.10, -59.03),
    ("Buenos Aires", -34.61, -58.38),
    ("La Plata", -34.92, -57.95),
    ("Mar del Plata", -38.00, -57.55),
]

SUBESTACIONES = [
    ("Resistencia", -27.45, -59.00),
    ("Santo Tomé", -31.67, -60.78),
    ("Rosario Oeste", -32.90, -60.75),
    ("Ezeiza", -34.85, -58.55),
    ("Necochea", -38.55, -58.74),
]

# ============================================================
# RIESGO POBLACIONAL — POBLACIÓN GRILLADA REAL (GHS-POP)
# ============================================================
#
# Carpeta donde deben estar TODOS los .tif de GHS-POP descargados
# (tiles que cubran el rectángulo de búsqueda). Para este corredor
# hacen falta: R13_C12, R13_C13, R14_C12, R14_C13.
GHS_POP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ghs_pop_tiles")

RADIO_RIESGO_KM = 15.0       # radio en el que se suma población real alrededor del punto
RADIO_MIN_DURO_KM = 15.0     # exclusión regulatoria dura (mismo radio en este caso)

_ghs_mosaico = None
_ghs_transform = None
_ghs_res_m = None
_ghs_offsets_mask = None  # (dr, dc) de píxeles dentro del radio, precalculado una sola vez
_transformer_a_moll = Transformer.from_crs("EPSG:4326", "ESRI:54009", always_xy=True)


def _cargar_mosaico_ghs_pop():
    """Arma un único mosaico en memoria a partir de todos los .tif en GHS_POP_DIR."""
    global _ghs_mosaico, _ghs_transform, _ghs_res_m, _ghs_offsets_mask

    archivos = sorted(glob.glob(os.path.join(GHS_POP_DIR, "*.tif")))
    if not archivos:
        raise FileNotFoundError(
            f"No se encontraron tiles GHS-POP (.tif) en '{GHS_POP_DIR}'.\n"
            "Colocá ahí los tiles que cubren el corredor de estudio: "
            "R13_C12, R13_C13, R14_C12, R14_C13 "
            "(GHS_POP_E2030_GLOBE_R2023A_54009_1000_V1_0_R##_C##.tif)."
        )

    datasets = [rasterio.open(f) for f in archivos]
    mosaico, transform = rio_merge(datasets, nodata=-200)
    for ds in datasets:
        ds.close()

    banda = mosaico[0].astype(np.float64)
    banda[banda < 0] = 0.0  # nodata (-200, típicamente océano/sin dato) -> 0 habitantes

    _ghs_mosaico = banda
    _ghs_transform = transform
    _ghs_res_m = abs(transform.a)

    # Máscara circular de offsets (fila, col) relativos al centro, calculada
    # una sola vez y reutilizada en cada consulta (todas usan el mismo radio).
    radio_px = int(math.ceil((RADIO_RIESGO_KM * 1000) / _ghs_res_m))
    dr, dc = np.meshgrid(
        np.arange(-radio_px, radio_px + 1), np.arange(-radio_px, radio_px + 1), indexing="ij"
    )
    dist_m = np.sqrt((dr * _ghs_res_m) ** 2 + (dc * _ghs_res_m) ** 2)
    dentro = dist_m <= RADIO_RIESGO_KM * 1000
    _ghs_offsets_mask = (dr[dentro], dc[dentro])

    print(f"[GHS-POP] Mosaico armado a partir de {len(archivos)} tile(s): "
          f"{[os.path.basename(a) for a in archivos]}")
    print(f"[GHS-POP] Forma del mosaico: {banda.shape}, resolución: {_ghs_res_m:.1f} m/px, "
          f"población total en el mosaico: {banda.sum():,.0f}")


def _fila_columna(point):
    lat, lon = point
    x, y = _transformer_a_moll.transform(lon, lat)
    return rasterio.transform.rowcol(_ghs_transform, x, y)


def poblacion_en_radio(point, radio_km=RADIO_RIESGO_KM):
    """Suma la población real (GHS-POP) dentro de radio_km del punto."""
    if _ghs_mosaico is None:
        _cargar_mosaico_ghs_pop()

    row, col = _fila_columna(point)
    n_filas, n_cols = _ghs_mosaico.shape
    if not (0 <= row < n_filas and 0 <= col < n_cols):
        return 0.0  # punto fuera de la cobertura de los tiles cargados

    dr, dc = _ghs_offsets_mask
    filas = row + dr
    cols = col + dc
    validos = (filas >= 0) & (filas < n_filas) & (cols >= 0) & (cols < n_cols)
    return float(_ghs_mosaico[filas[validos], cols[validos]].sum())


def hay_poblacion_cerca(point, umbral_habitantes_px=1.0, radio_busqueda_km=RADIO_MIN_DURO_KM):
    """
    Para la exclusión dura: True si dentro de radio_busqueda_km hay algún
    píxel con población por encima del umbral (evita colocar la central
    a metros de un núcleo poblacional, aunque sea chico).
    """
    if _ghs_mosaico is None:
        _cargar_mosaico_ghs_pop()
    row, col = _fila_columna(point)
    n_filas, n_cols = _ghs_mosaico.shape
    if not (0 <= row < n_filas and 0 <= col < n_cols):
        return False
    dr, dc = _ghs_offsets_mask
    filas = row + dr
    cols = col + dc
    validos = (filas >= 0) & (filas < n_filas) & (cols >= 0) & (cols < n_cols)
    return bool((_ghs_mosaico[filas[validos], cols[validos]] >= umbral_habitantes_px).any())

# Margen (en grados) para definir el rectángulo de búsqueda alrededor
# del corredor. El cromosoma (lat, lon) se genera y se acota dentro
# de este rectángulo.
MARGEN_BUSQUEDA_DEG = 2.0


# ============================================================
# FUNCIONES GEOGRÁFICAS
# ============================================================

def haversine(p1, p2):
    """Distancia en km entre dos puntos (lat, lon)."""
    R = 6371.0
    lat1, lon1 = map(math.radians, p1)
    lat2, lon2 = map(math.radians, p2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _a_plano_local(point, origen):
    """
    Proyección equirectangular simple (lat, lon) -> (x, y) en km,
    centrada en 'origen'. Válida como aproximación local (decenas
    a pocos cientos de km), suficiente para medir distancia a un
    segmento de la polilínea.
    """
    lat, lon = point
    lat0, lon0 = origen
    lat0_rad = math.radians(lat0)
    x = (lon - lon0) * 111.320 * math.cos(lat0_rad)
    y = (lat - lat0) * 110.574
    return x, y


def distancia_punto_segmento_km(point, seg_a, seg_b):
    """Distancia mínima (km) entre 'point' y el segmento seg_a-seg_b."""
    origen = seg_a
    px, py = _a_plano_local(point, origen)
    ax, ay = 0.0, 0.0
    bx, by = _a_plano_local(seg_b, origen)

    dx, dy = bx - ax, by - ay
    largo2 = dx * dx + dy * dy
    if largo2 < 1e-9:
        return math.hypot(px - ax, py - ay)

    t = ((px - ax) * dx + (py - ay) * dy) / largo2
    t = min(max(t, 0.0), 1.0)
    proj_x, proj_y = ax + t * dx, ay + t * dy
    return math.hypot(px - proj_x, py - proj_y)


def distancia_a_corredor(point, boundary_points):
    """Distancia mínima (km) del punto a la polilínea completa."""
    d_min = float("inf")
    for i in range(len(boundary_points) - 1):
        d = distancia_punto_segmento_km(point, boundary_points[i], boundary_points[i + 1])
        d_min = min(d_min, d)
    return d_min


def en_tierra(point):
    """
    True si (lat, lon) cae sobre tierra firme, usando una máscara de
    tierra/agua real (global-land-mask, resolución ~10 km).
    Esto es lo que evita que el GA converja en pleno río o mar: la
    cercanía a la polilínea del corredor por sí sola no distingue de
    qué lado de la costa está el punto.
    """
    lat, lon = point
    return bool(globe.is_land(lat, lon))


# ============================================================
# RIESGO SÍSMICO (proxy simplificado tipo INPRES)
# ============================================================

LON_MIN_RIESGO = -66.0  # borde cordillerano (riesgo máximo)
LON_MAX_RIESGO = -56.0  # borde costero (riesgo mínimo)


def riesgo_sismico(point):
    lat, lon = point
    base = (lon - LON_MAX_RIESGO) / (LON_MIN_RIESGO - LON_MAX_RIESGO)
    base = min(max(base, 0.0), 1.0)
    ruido = 0.05 * math.sin(lat * 3.1) * math.cos(lon * 2.7)
    return min(max(base + ruido, 0.0), 1.0)


# ============================================================
# COSTO ELÉCTRICO (0 = más barato, luego se normaliza e invierte)
# ============================================================

def costo_electrico_bruto(point, w_red=0.6, w_demanda=0.4):
    d_red = min(haversine(point, (s[1], s[2])) for s in SUBESTACIONES)
    d_demanda = min(haversine(point, (c[1], c[2])) for c in CIUDADES)
    return w_red * d_red + w_demanda * d_demanda


def normalizar(valores):
    vmin, vmax = min(valores), max(valores)
    rango = (vmax - vmin) if (vmax - vmin) > 1e-9 else 1e-9
    return [(v - vmin) / rango for v in valores]





# ============================================================
# CERCANÍA AL CORREDOR HÍDRICO
# (necesaria para la toma de agua de refrigeración; reemplaza la
#  restricción implícita que antes daba mover el cromosoma sobre
#  la polilínea)
# ============================================================

ANCHO_CORREDOR_KM = 40.0  # distancia a la que la penalización cae fuerte


def f_agua(point, ancho_km=ANCHO_CORREDOR_KM):
    d = distancia_a_corredor(point, BOUNDARY_POINTS)
    return math.exp(-(d / ancho_km) ** 2)  # 1.0 sobre el eje, decae al alejarse


# ============================================================
# ALGORITMO GENÉTICO
# ============================================================

POP_SIZE = 60
N_GENERACIONES = 1000
PROB_CRUZA = 0.75
PROB_MUTACION = 0.25
SIGMA_MUTACION_KM = 15.0  # desvío estándar de la mutación, en km
TAM_TORNEO = 2

PESOS_FITNESS = dict(costo=0.25, sismico=0.25, riesgo_poblacional=0.25, agua=0.25)

# Rectángulo de búsqueda: envolvente de BOUNDARY_POINTS + margen
LAT_MIN = min(p[0] for p in BOUNDARY_POINTS) - MARGEN_BUSQUEDA_DEG
LAT_MAX = max(p[0] for p in BOUNDARY_POINTS) + MARGEN_BUSQUEDA_DEG
LON_MIN = min(p[1] for p in BOUNDARY_POINTS) - MARGEN_BUSQUEDA_DEG
LON_MAX = max(p[1] for p in BOUNDARY_POINTS) + MARGEN_BUSQUEDA_DEG


MAX_INTENTOS_MUESTREO_TIERRA = 200


def crear_individuo():
    """
    Genera un individuo por rechazo: sortea (lat, lon) dentro del
    rectángulo de búsqueda hasta que caiga en tierra. Evita que la
    población inicial arranque con puntos en el río/mar, que de
    entrada tendrían fitness 0.
    """
    for _ in range(MAX_INTENTOS_MUESTREO_TIERRA):
        lat = random.uniform(LAT_MIN, LAT_MAX)
        lon = random.uniform(LON_MIN, LON_MAX)
        if en_tierra((lat, lon)):
            return {"lat": lat, "lon": lon}
    # Fallback (no debería ocurrir salvo un rectángulo casi todo de agua):
    # se devuelve igual, quedará con fitness 0 y será reemplazado por
    # selección/cruza en las próximas generaciones.
    return {"lat": lat, "lon": lon}


def evaluar_poblacion(poblacion):
    """Devuelve fitness normalizado [0,1] para toda la población.

    Restricción dura: cualquier punto que no esté en tierra firme
    recibe fitness = 0, sin importar qué tan bueno sea el resto de
    los términos. No tiene sentido evaluar costo eléctrico o riesgo
    sísmico de un punto en medio del Río de la Plata.
    """
    puntos = [(ind["lat"], ind["lon"]) for ind in poblacion]
    validos = [en_tierra(p) for p in puntos]

    # Normalizamos el costo eléctrico solo sobre los puntos válidos,
    # para que un punto en el agua (que podría estar "cerca" en línea
    # recta de una subestación) no distorsione la escala.
    puntos_validos = [p for p, v in zip(puntos, validos) if v]
    if puntos_validos:
        costos_brutos_validos = [costo_electrico_bruto(p) for p in puntos_validos]
        vmin, vmax = min(costos_brutos_validos), max(costos_brutos_validos)

        riesgos_brutos_validos = [poblacion_en_radio(p) for p in puntos_validos]
        rmin, rmax = min(riesgos_brutos_validos), max(riesgos_brutos_validos)
    else:
        vmin, vmax = 0.0, 1.0
        rmin, rmax = 0.0, 1.0
    rango = (vmax - vmin) if (vmax - vmin) > 1e-9 else 1e-9
    rango_riesgo = (rmax - rmin) if (rmax - rmin) > 1e-9 else 1e-9

    fitness_total = []
    detalle = []
    for p, valido in zip(puntos, validos):
        if not valido:
            fitness_total.append(0.0)
            detalle.append(dict(punto=p, en_tierra=False,
                                 f_costo=0.0, f_sismico=0.0, f_riesgo_poblacional=0.0, f_agua=0.0))
            continue

        # Exclusión regulatoria dura: no admitir el punto si hay
        # población real (según GHS-POP) dentro del radio mínimo duro.
        if hay_poblacion_cerca(p):
            fitness_total.append(0.0)
            detalle.append(dict(punto=p, en_tierra=True,
                                 f_costo=0.0, f_sismico=0.0, f_riesgo_poblacional=0.0, f_agua=0.0))
            continue

        c_norm = (costo_electrico_bruto(p) - vmin) / rango
        f_costo = 1 - c_norm
        f_sismico = 1 - riesgo_sismico(p)

        r_norm = (poblacion_en_radio(p) - rmin) / rango_riesgo
        f_riesgo_poblacional = 1 - r_norm  # menos población expuesta = mejor

        f_hidrico = f_agua(p)
        f = (PESOS_FITNESS["costo"] * f_costo
             + PESOS_FITNESS["sismico"] * f_sismico
             + PESOS_FITNESS["riesgo_poblacional"] * f_riesgo_poblacional
             + PESOS_FITNESS["agua"] * f_hidrico)
        fitness_total.append(f)
        detalle.append(dict(punto=p, en_tierra=True, f_costo=f_costo, f_sismico=f_sismico,
                             f_riesgo_poblacional=f_riesgo_poblacional, f_agua=f_hidrico))
    return fitness_total, detalle


def seleccion_torneo(poblacion, fitness, k=TAM_TORNEO):
    participantes = random.sample(list(zip(poblacion, fitness)), k)
    participantes.sort(key=lambda x: x[1], reverse=True)
    return participantes[0][0]


def cruza(p1, p2):
    """Crossover aritmético (blend). Mismo alpha para ambas
    componentes: el hijo queda sobre el segmento que une a los
    padres en el plano (lat, lon)."""
    if random.random() > PROB_CRUZA:
        return dict(p1), dict(p2)
    alpha = random.random()
    lat1 = alpha * p1["lat"] + (1 - alpha) * p2["lat"]
    lon1 = alpha * p1["lon"] + (1 - alpha) * p2["lon"]
    lat2 = alpha * p2["lat"] + (1 - alpha) * p1["lat"]
    lon2 = alpha * p2["lon"] + (1 - alpha) * p1["lon"]
    return (
        {"lat": min(max(lat1, LAT_MIN), LAT_MAX), "lon": min(max(lon1, LON_MIN), LON_MAX)},
        {"lat": min(max(lat2, LAT_MIN), LAT_MAX), "lon": min(max(lon2, LON_MIN), LON_MAX)},
    )


def mutacion(ind, sigma_km):
    """Mutación gaussiana isotrópica en km: se convierte sigma_km a
    grados de lat/lon por separado, porque 1° de longitud vale menos
    km que 1° de latitud a medida que uno se aleja del ecuador."""
    if random.random() < PROB_MUTACION:
        lat_rad = math.radians(ind["lat"])
        sigma_lat_deg = sigma_km / 110.574
        coslat = max(math.cos(lat_rad), 1e-6)
        sigma_lon_deg = sigma_km / (111.320 * coslat)
        nuevo_lat = ind["lat"] + random.gauss(0, sigma_lat_deg)
        nuevo_lon = ind["lon"] + random.gauss(0, sigma_lon_deg)
        ind["lat"] = min(max(nuevo_lat, LAT_MIN), LAT_MAX)
        ind["lon"] = min(max(nuevo_lon, LON_MIN), LON_MAX)
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

        # sigma de mutación: se puede hacer decreciente con la
        # generación para pasar de exploración a explotación; acá se
        # deja un decaimiento lineal simple entre SIGMA_MUTACION_KM y
        # ~1/5 de ese valor.
        frac = gen / max(N_GENERACIONES - 1, 1)
        sigma_km = SIGMA_MUTACION_KM * (1.0 - 0.8 * frac)

        nueva_poblacion = []
        while len(nueva_poblacion) < POP_SIZE:
            padre1 = seleccion_torneo(poblacion, fitness)
            padre2 = seleccion_torneo(poblacion, fitness)
            hijo1, hijo2 = cruza(padre1, padre2)
            hijo1 = mutacion(hijo1, sigma_km)
            hijo2 = mutacion(hijo2, sigma_km)
            nueva_poblacion.append(hijo1)
            if len(nueva_poblacion) < POP_SIZE:
                nueva_poblacion.append(hijo2)

        poblacion = nueva_poblacion

    return mejor_global, historia_mejor, historia_promedio, historia_peor, historia_std, poblacion


# ============================================================
# GRÁFICO DE CONVERGENCIA
# ============================================================

def graficar_convergencia(historia_mejor, historia_promedio, historia_peor, historia_std,
                            path="convergencia_ga.png"):
    generaciones = np.arange(1, len(historia_mejor) + 1)
    promedio = np.array(historia_promedio)
    std = np.array(historia_std)

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
    ax.set_title("Convergencia del Algoritmo Genético\nUbicación óptima - Central Nuclear Modular (cromosoma 2D)")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    texto = (f"Última generación:\n"
              f"Mejor:    {historia_mejor[-1]:.3f}\n"
              f"Promedio: {historia_promedio[-1]:.3f}\n"
              f"Peor:     {historia_peor[-1]:.3f}\n"
              f"Desvío σ: {historia_std[-1]:.3f}")
    ax.text(0.015, 0.02, texto, transform=ax.transAxes, fontsize=9.5,
            verticalalignment="bottom", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.9))

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ============================================================
# MAPA HTML CON EL PUNTO ÓPTIMO
# ============================================================

def _reproyectar_ghs_a_wgs84(mosaico, transform_moll, downsample_factor=2):
    src_height, src_width = mosaico.shape
    src_crs = "ESRI:54009"
    dst_crs = "EPSG:4326"

    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs, dst_crs, src_width, src_height,
        *rasterio.transform.array_bounds(src_height, src_width, transform_moll)
    )

    if downsample_factor > 1:
        # Escala uniforme en X e Y: cada pixel de salida pasa a cubrir
        # 'downsample_factor' veces el ancho/alto original.
        dst_transform = dst_transform * Affine.scale(downsample_factor)
        dst_width = max(1, dst_width // downsample_factor)
        dst_height = max(1, dst_height // downsample_factor)

    destino = np.zeros((dst_height, dst_width), dtype=np.float64)

    reproject(
        source=mosaico,
        destination=destino,
        src_transform=transform_moll,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.average,
        src_nodata=0.0,
        dst_nodata=0.0,
    )

    return destino, dst_transform

def generar_mapa(mejor, mejor_fitness_total, poblacion_final, path="mapa_optimo.html"):
    punto_optimo = mejor["detalle"]["punto"]

    m = folium.Map(location=punto_optimo, zoom_start=6, tiles="OpenStreetMap")

    # Corredor hídrico de referencia (ya no es el único lugar válido,
    # ahora es el eje de la franja penalizada por f_agua)
    folium.PolyLine(
        BOUNDARY_POINTS, color="blue", weight=3, opacity=0.6,
        tooltip="Eje del corredor hídrico (Río Paraná -> Mar del Plata)"
    ).add_to(m)

    for nombre, lat, lon in CIUDADES:
        folium.CircleMarker(
            location=(lat, lon), radius=5, color="#6a1b9a", fill=True,
            fill_color="#6a1b9a", fill_opacity=0.8,
            popup=f"Ciudad (demanda/costo): {nombre}",
        ).add_to(m)

    # Overlay de densidad poblacional real (GHS-POP) sobre el mapa.
    # Nota: el mosaico está en Mollweide; para el overlay en Folium
    # (que trabaja en lat/lon) se usa el rectángulo envolvente en
    # lat/lon de las cuatro esquinas del mosaico como aproximación —
    # suficiente para visualización a esta escala regional, aunque
    # introduce una distorsión geométrica menor.
    if _ghs_mosaico is not None:
        # 1. Pasamos el mapa por la función nueva para corregir la proyección
        img_wgs84, transform_wgs84 = _reproyectar_ghs_a_wgs84(_ghs_mosaico, _ghs_transform)

        # 2. Le damos colorcito
        img_log = np.log1p(img_wgs84)
        img_max = img_log.max() if img_log.max() > 0 else 1.0
        img_norm = np.clip(img_log / img_max, 0, 1)
        cmap = matplotlib.colormaps["YlOrRd"]
        rgba = cmap(img_norm)
        rgba[..., 3] = np.where(img_wgs84 > 0.5, 0.55, 0.0)

        # 3. Calculamos los bordes exactos ya corregidos
        h, w = img_wgs84.shape
        oeste, norte = transform_wgs84 * (0, 0)
        este, sur = transform_wgs84 * (w, h)

        # 4. Lo pegamos en Folium
        folium.raster_layers.ImageOverlay(
            image=rgba,
            bounds=[[sur, oeste], [norte, este]],
            opacity=0.7,
            name="Densidad poblacional (GHS-POP, reproyectado a WGS84)",
        ).add_to(m)
        folium.LayerControl().add_to(m)

    for nombre, lat, lon in SUBESTACIONES:
        folium.Marker(
            location=(lat, lon),
            icon=folium.Icon(color="orange", icon="bolt", prefix="fa"),
            popup=f"Subestación: {nombre}",
        ).add_to(m)

    # Población final: ahora dispersa en 2D, no solo sobre la línea
    for ind in poblacion_final:
        p = (ind["lat"], ind["lon"])
        folium.CircleMarker(
            location=p, radius=2, color="gray", fill=True,
            fill_opacity=0.4,
        ).add_to(m)

    detalle = mejor["detalle"]
    popup_html = (
        f"<b>Ubicación óptima SMR</b><br>"
        f"Lat: {punto_optimo[0]:.4f}, Lon: {punto_optimo[1]:.4f}<br>"
        f"Fitness costo eléctrico: {detalle['f_costo']:.3f}<br>"
        f"Fitness riesgo sísmico: {detalle['f_sismico']:.3f}<br>"
        f"Fitness riesgo poblacional: {detalle['f_riesgo_poblacional']:.3f}<br>"
        f"Fitness cercanía al corredor hídrico: {detalle['f_agua']:.3f}<br>"
        f"<b>Fitness total: {mejor_fitness_total:.3f}</b>"
    )
    folium.Marker(
        location=punto_optimo,
        icon=folium.Icon(color="red", icon="star", prefix="fa"),
        popup=folium.Popup(popup_html, max_width=300),
        tooltip="Ubicación óptima",
    ).add_to(m)

    m.save(path)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    # Sin seed fija: cada corrida explora la aleatoriedad de forma
    # distinta. Para reproducibilidad exacta de una corrida puntual,
    # descomentar random.seed(N) / np.random.seed(N).
    random.seed()
    np.random.seed()

    mejor, historia_mejor, historia_promedio, historia_peor, historia_std, poblacion_final = correr_ga()
    mejor_fitness_total = historia_mejor[-1]

    print("=== RESULTADO ===")
    print(f"Punto óptimo (lat, lon):        {mejor['detalle']['punto']}")
    print(f"Fitness costo eléctrico:        {mejor['detalle']['f_costo']:.3f}")
    print(f"Fitness riesgo sísmico:         {mejor['detalle']['f_sismico']:.3f}")
    print(f"Fitness riesgo poblacional:     {mejor['detalle']['f_riesgo_poblacional']:.3f}")
    print(f"Fitness cercanía corredor agua: {mejor['detalle']['f_agua']:.3f}")
    print(f"Fitness total:                  {mejor_fitness_total:.3f}")

    OUTPUT_DIR = "resultados_ga"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    path_mapa = os.path.join(OUTPUT_DIR, f"mapa_optimo_{timestamp}.html")
    path_grafico = os.path.join(OUTPUT_DIR, f"convergencia_ga_{timestamp}.png")
    path_json = os.path.join(OUTPUT_DIR, f"resultado_optimo_{timestamp}.json")

    graficar_convergencia(historia_mejor, historia_promedio, historia_peor, historia_std, path=path_grafico)
    generar_mapa(mejor, mejor_fitness_total, poblacion_final, path=path_mapa)

    with open(path_json, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": timestamp,
            "punto": mejor["detalle"]["punto"],
            "fitness_costo": mejor["detalle"]["f_costo"],
            "fitness_sismico": mejor["detalle"]["f_sismico"],
            "fitness_riesgo_poblacional": mejor["detalle"]["f_riesgo_poblacional"],
            "fitness_agua": mejor["detalle"]["f_agua"],
            "fitness_total": mejor_fitness_total,
        }, f, ensure_ascii=False, indent=2)

    path_log = os.path.join(OUTPUT_DIR, "log_corridas.csv")
    existe_log = os.path.exists(path_log)
    with open(path_log, "a", encoding="utf-8") as f:
        if not existe_log:
            f.write("timestamp,lat,lon,fitness_costo,fitness_sismico,fitness_riesgo_poblacional,fitness_agua,"
                    "fitness_mejor,fitness_promedio,fitness_peor,fitness_std\n")
        lat, lon = mejor["detalle"]["punto"]
        f.write(f"{timestamp},{lat:.6f},{lon:.6f},"
                f"{mejor['detalle']['f_costo']:.4f},{mejor['detalle']['f_sismico']:.4f},"
                f"{mejor['detalle']['f_riesgo_poblacional']:.4f},{mejor['detalle']['f_agua']:.4f},"
                f"{mejor_fitness_total:.4f},"
                f"{historia_promedio[-1]:.4f},{historia_peor[-1]:.4f},{historia_std[-1]:.4f}\n")

    print(f"\nArchivos generados en '{OUTPUT_DIR}/':")
    print(f"  - {path_mapa}")
    print(f"  - {path_grafico}")
    print(f"  - {path_json}")
    print(f"  - {path_log} (acumula todas las corridas)")