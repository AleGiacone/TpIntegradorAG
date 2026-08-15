"""
Algoritmo Genético para determinar la ubicación óptima de una
Central Nuclear Modular Pequeña (SMR) en la región del corredor
Río Paraná -> Mar del Plata.

Cambio respecto a la versión anterior:
  - El cromosoma ya no es un escalar t sobre una polilínea, sino un
    vector libre v = (lat, lon).
  - El crossover es aritmético (blend) sobre ambas componentes.
  - Como al liberar el cromosoma se pierde la restricción implícita
    de "estar sobre el corredor ribereño" que antes daba la polilínea,
    se agrega un término explícito f_agua que penaliza alejarse del
    corredor (necesario por la toma de agua de refrigeración de la
    central). El corredor sigue definido por BOUNDARY_POINTS, pero
    ahora funciona como el eje de una franja, no como el único lugar
    válido.
  - La mutación gaussiana se hace isotrópica en km (no en grados),
    porque 1° de longitud no equivale a la misma distancia que 1° de
    latitud a estas latitudes.

Criterios de fitness:
  1. Costo eléctrico   (cercanía a red de transmisión y a centros de demanda)
  2. Riesgo sísmico     (cercanía a fallas / zonificación INPRES, proxy simplificado)
  3. Restricción de distancia a ciudades (óptimo entre 20 y 30 km)
  4. Cercanía al corredor hídrico (toma de agua de refrigeración)

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
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import folium


# ============================================================
# DATOS DE ENTRADA (REEMPLAZAR CON TUS DATOS REALES)
# ============================================================

BOUNDARY_POINTS = [
    (-35.23347, -57.26014),
    (-34.5949, -58.39336),
    (-34.43723, -58.51808),
    (-34.02051, -58.36545),
    (-33.69495, -59.60311),
    (-32.8453, -60.70723),
    (-31.79303, -60.65568),
    (-30.07331, -59.58177),
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
    ("Atucha (referencia)", -33.97, -59.20),
    ("Ezeiza", -34.85, -58.55),
    ("Necochea", -38.55, -58.74),
]

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
# RESTRICCIÓN DE DISTANCIA A CIUDAD (óptimo entre 20 y 30 km)
# ============================================================

def penalizacion_ciudad(point, d_optimo=25.0, sigma=8.0, d_min_duro=15.0):
    d = min(haversine(point, (c[1], c[2])) for c in CIUDADES)
    if d < d_min_duro:
        return 0.0  # demasiado cerca: descartado
    return math.exp(-((d - d_optimo) ** 2) / (2 * sigma ** 2))


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
N_GENERACIONES = 100
PROB_CRUZA = 0.75
PROB_MUTACION = 0.25
SIGMA_MUTACION_KM = 15.0  # desvío estándar de la mutación, en km
TAM_TORNEO = 2

PESOS_FITNESS = dict(costo=0.25, sismico=0.25, ciudad=0.25, agua=0.25)

# Rectángulo de búsqueda: envolvente de BOUNDARY_POINTS + margen
LAT_MIN = min(p[0] for p in BOUNDARY_POINTS) - MARGEN_BUSQUEDA_DEG
LAT_MAX = max(p[0] for p in BOUNDARY_POINTS) + MARGEN_BUSQUEDA_DEG
LON_MIN = min(p[1] for p in BOUNDARY_POINTS) - MARGEN_BUSQUEDA_DEG
LON_MAX = max(p[1] for p in BOUNDARY_POINTS) + MARGEN_BUSQUEDA_DEG


def crear_individuo():
    return {
        "lat": random.uniform(LAT_MIN, LAT_MAX),
        "lon": random.uniform(LON_MIN, LON_MAX),
    }


def evaluar_poblacion(poblacion):
    """Devuelve fitness normalizado [0,1] para toda la población."""
    puntos = [(ind["lat"], ind["lon"]) for ind in poblacion]

    costos_brutos = [costo_electrico_bruto(p) for p in puntos]
    costos_norm = normalizar(costos_brutos)  # 0 = más barato

    fitness_total = []
    detalle = []
    for p, c_norm in zip(puntos, costos_norm):
        f_costo = 1 - c_norm
        f_sismico = 1 - riesgo_sismico(p)
        f_ciudad = penalizacion_ciudad(p)
        f_hidrico = f_agua(p)
        f = (PESOS_FITNESS["costo"] * f_costo
             + PESOS_FITNESS["sismico"] * f_sismico
             + PESOS_FITNESS["ciudad"] * f_ciudad
             + PESOS_FITNESS["agua"] * f_hidrico)
        fitness_total.append(f)
        detalle.append(dict(punto=p, f_costo=f_costo, f_sismico=f_sismico,
                             f_ciudad=f_ciudad, f_agua=f_hidrico))
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
            popup=f"Ciudad: {nombre}",
        ).add_to(m)

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
        f"Fitness distancia a ciudad: {detalle['f_ciudad']:.3f}<br>"
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
    print(f"Fitness distancia ciudad:       {mejor['detalle']['f_ciudad']:.3f}")
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
            "fitness_ciudad": mejor["detalle"]["f_ciudad"],
            "fitness_agua": mejor["detalle"]["f_agua"],
            "fitness_total": mejor_fitness_total,
        }, f, ensure_ascii=False, indent=2)

    path_log = os.path.join(OUTPUT_DIR, "log_corridas.csv")
    existe_log = os.path.exists(path_log)
    with open(path_log, "a", encoding="utf-8") as f:
        if not existe_log:
            f.write("timestamp,lat,lon,fitness_costo,fitness_sismico,fitness_ciudad,fitness_agua,"
                    "fitness_mejor,fitness_promedio,fitness_peor,fitness_std\n")
        lat, lon = mejor["detalle"]["punto"]
        f.write(f"{timestamp},{lat:.6f},{lon:.6f},"
                f"{mejor['detalle']['f_costo']:.4f},{mejor['detalle']['f_sismico']:.4f},"
                f"{mejor['detalle']['f_ciudad']:.4f},{mejor['detalle']['f_agua']:.4f},"
                f"{mejor_fitness_total:.4f},"
                f"{historia_promedio[-1]:.4f},{historia_peor[-1]:.4f},{historia_std[-1]:.4f}\n")

    print(f"\nArchivos generados en '{OUTPUT_DIR}/':")
    print(f"  - {path_mapa}")
    print(f"  - {path_grafico}")
    print(f"  - {path_json}")
    print(f"  - {path_log} (acumula todas las corridas)")
