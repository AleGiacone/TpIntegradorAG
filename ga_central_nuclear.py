"""
Algoritmo Genético para determinar la ubicación óptima de una
Central Nuclear Modular Pequeña (SMR) a lo largo del corredor
Río Paraná -> Mar del Plata.

  1. Costo eléctrico   (cercanía a red de transmisión y a centros de demanda)
  2. Riesgo sísmico     (cercanía a fallas / zonificación INPRES)
  3. Restricción de distancia a ciudades (óptimo entre 20 y 30 km)

Salidas:
  - mapa_optimo.html   -> mapa interactivo (Leaflet/Folium) con el punto óptimo
  - convergencia_ga.png -> evolución del fitness (mejor y promedio) por generación


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

random.seed(42)
np.random.seed(42)


# DATOS DE ENTRADA (REEMPLAZAR CON TUS DATOS REALES)

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

# Subestaciones

SUBESTACIONES = [
    ("Resistencia", -27.45, -59.00),
    ("Santo Tomé", -31.67, -60.78),
    ("Rosario Oeste", -32.90, -60.75),
    ("Atucha (referencia)", -33.97, -59.20),
    ("Ezeiza", -34.85, -58.55),
    ("Necochea", -38.55, -58.74),
]

# FUNCIONES GEOGRÁFICAS


def haversine(p1, p2):
    """Distancia en km entre dos puntos (lat, lon)."""
    R = 6371.0
    lat1, lon1 = map(math.radians, p1)
    lat2, lon2 = map(math.radians, p2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def interpolate_point(t, boundary_points):
  # Interpolación lineal sobre la polilínea de la línea frontera.
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



# RIESGO SÍSMICO (proxy simplificado tipo INPRES)

LON_MIN_RIESGO = -66.0  # borde cordillerano (riesgo máximo)
LON_MAX_RIESGO = -56.0  # borde costero (riesgo mínimo)


def riesgo_sismico(point):
    lat, lon = point
    base = (lon - LON_MAX_RIESGO) / (LON_MIN_RIESGO - LON_MAX_RIESGO)
    base = min(max(base, 0.0), 1.0)
    ruido = 0.05 * math.sin(lat * 3.1) * math.cos(lon * 2.7)
    return min(max(base + ruido, 0.0), 1.0)



# COSTO ELÉCTRICO (0 = más barato, luego se normaliza e invierte)


def costo_electrico_bruto(point, w_red=0.6, w_demanda=0.4):
    d_red = min(haversine(point, (s[1], s[2])) for s in SUBESTACIONES)
    d_demanda = min(haversine(point, (c[1], c[2])) for c in CIUDADES)
    return w_red * d_red + w_demanda * d_demanda


def normalizar(valores):
    vmin, vmax = min(valores), max(valores)
    rango = (vmax - vmin) if (vmax - vmin) > 1e-9 else 1e-9
    return [(v - vmin) / rango for v in valores]



# 5. RESTRICCIÓN DE DISTANCIA A CIUDAD (óptimo entre 20 y 30 km)


def penalizacion_ciudad(point, d_optimo=25.0, sigma=8.0, d_min_duro=15.0):
    d = min(haversine(point, (c[1], c[2])) for c in CIUDADES)
    if d < d_min_duro:
        return 0.0  # demasiado cerca: descartado
    return math.exp(-((d - d_optimo) ** 2) / (2 * sigma ** 2))


# ALGORITMO GENÉTICO

POP_SIZE = 60
N_GENERACIONES = 100
PROB_CRUZA = 0.75
PROB_MUTACION = 0.25
SIGMA_MUTACION_INICIAL = 0.12
TAM_TORNEO = 2

PESOS_FITNESS = dict(costo=0.33, sismico=0.33, ciudad=0.33)


def crear_individuo():
    return {"t": random.random()}


def evaluar_poblacion(poblacion):
    """Devuelve fitness normalizado [0,1] para toda la población."""
    puntos = [interpolate_point(ind["t"], BOUNDARY_POINTS) for ind in poblacion]

    costos_brutos = [costo_electrico_bruto(p) for p in puntos]
    costos_norm = normalizar(costos_brutos)  # 0 = más barato

    fitness_total = []
    detalle = []
    for p, c_norm in zip(puntos, costos_norm):
        f_costo = 1 - c_norm
        f_sismico = 1 - riesgo_sismico(p)
        f_ciudad = penalizacion_ciudad(p)
        f = (PESOS_FITNESS["costo"] * f_costo
             + PESOS_FITNESS["sismico"] * f_sismico
             + PESOS_FITNESS["ciudad"] * f_ciudad)
        fitness_total.append(f)
        detalle.append(dict(punto=p, f_costo=f_costo, f_sismico=f_sismico, f_ciudad=f_ciudad))
    return fitness_total, detalle


def seleccion_torneo(poblacion, fitness, k=TAM_TORNEO):
    participantes = random.sample(list(zip(poblacion, fitness)), k)
    participantes.sort(key=lambda x: x[1], reverse=True)
    return participantes[0][0]


def cruza(p1, p2):
    if random.random() > PROB_CRUZA:
        return dict(p1), dict(p2)
    alpha = random.random()
    t1 = alpha * p1["t"] + (1 - alpha) * p2["t"]
    t2 = alpha * p2["t"] + (1 - alpha) * p1["t"]
    return {"t": min(max(t1, 0), 1)}, {"t": min(max(t2, 0), 1)}


def mutacion(ind, sigma):
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

        # sigma de mutación decreciente (exploración -> explotación)
        sigma = 0.25
        orden = sorted(zip(poblacion, fitness), key=lambda x: x[1], reverse=True)
        nueva_poblacion = []

        while len(nueva_poblacion) < POP_SIZE:
            padre1 = seleccion_torneo(poblacion, fitness)
            padre2 = seleccion_torneo(poblacion, fitness)
            hijo1, hijo2 = cruza(padre1, padre2)
            hijo1 = mutacion(hijo1, sigma)
            hijo2 = mutacion(hijo2, sigma)
            nueva_poblacion.append(hijo1)
            if len(nueva_poblacion) < POP_SIZE:
                nueva_poblacion.append(hijo2)

        poblacion = nueva_poblacion

    return mejor_global, historia_mejor, historia_promedio, historia_peor, historia_std, poblacion



# GRÁFICO DE CONVERGENCIA 
 

def graficar_convergencia(historia_mejor, historia_promedio, historia_peor, historia_std,
                            path="convergencia_ga.png"):
    generaciones = np.arange(1, len(historia_mejor) + 1)
    promedio = np.array(historia_promedio)
    std = np.array(historia_std)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Banda de +/- 1 desvío estándar alrededor del promedio
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

    # Cuadro de texto con los valores finales (última generación)
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



# MAPA HTML CON EL PUNTO ÓPTIMO


def generar_mapa(mejor, poblacion_final, path="mapa_optimo.html"):
    punto_optimo = mejor["detalle"]["punto"]

    m = folium.Map(location=punto_optimo, zoom_start=6, tiles="OpenStreetMap")

    # Línea frontera (corredor de ubicaciones disponibles)
    folium.PolyLine(
        BOUNDARY_POINTS, color="blue", weight=3, opacity=0.6,
        tooltip="Corredor disponible (Río Paraná -> Mar del Plata)"
    ).add_to(m)

    # Ciudades
    for nombre, lat, lon in CIUDADES:
        folium.CircleMarker(
            location=(lat, lon), radius=5, color="#6a1b9a", fill=True,
            fill_color="#6a1b9a", fill_opacity=0.8,
            popup=f"Ciudad: {nombre}",
        ).add_to(m)

    # Subestaciones
    for nombre, lat, lon in SUBESTACIONES:
        folium.Marker(
            location=(lat, lon),
            icon=folium.Icon(color="orange", icon="bolt", prefix="fa"),
            popup=f"Subestación: {nombre}",
        ).add_to(m)

    # Últimas ubicaciones evaluadas por la población (para visualizar dispersión)
    for ind in poblacion_final:
        p = interpolate_point(ind["t"], BOUNDARY_POINTS)
        folium.CircleMarker(
            location=p, radius=2, color="gray", fill=True,
            fill_opacity=0.4,
        ).add_to(m)

    # Punto óptimo
    detalle = mejor["detalle"]
    popup_html = (
        f"<b>Ubicación óptima SMR</b><br>"
        f"Lat: {punto_optimo[0]:.4f}, Lon: {punto_optimo[1]:.4f}<br>"
        f"Fitness costo eléctrico: {detalle['f_costo']:.3f}<br>"
        f"Fitness riesgo sísmico: {detalle['f_sismico']:.3f}<br>"
        f"Fitness distancia a ciudad: {detalle['f_ciudad']:.3f}<br>"
        f"<b>Fitness total: {mejor_fitness_total:.3f}</b>"
    )
    folium.Marker(
        location=punto_optimo,
        icon=folium.Icon(color="red", icon="star", prefix="fa"),
        popup=folium.Popup(popup_html, max_width=300),
        tooltip="Ubicación óptima",
    ).add_to(m)

    m.save(path)



# MAIN


if __name__ == "__main__":
    # --- Sin seed fija: cada corrida explora la aleatoriedad de forma
    #     distinta. Si en algún momento necesitás reproducibilidad exacta
    #     de una corrida puntual, descomentá random.seed(N) / np.random.seed(N).
    random.seed()
    np.random.seed()

    mejor, historia_mejor, historia_promedio, historia_peor, historia_std, poblacion_final = correr_ga()
    mejor_fitness_total = historia_mejor[-1]

    print("=== RESULTADO ===")
    print(f"t óptimo (posición en la línea frontera): {mejor['t']:.4f}")
    print(f"Punto óptimo (lat, lon): {mejor['detalle']['punto']}")
    print(f"Fitness costo eléctrico:  {mejor['detalle']['f_costo']:.3f}")
    print(f"Fitness riesgo sísmico:   {mejor['detalle']['f_sismico']:.3f}")
    print(f"Fitness distancia ciudad: {mejor['detalle']['f_ciudad']:.3f}")
    print(f"Fitness total:            {mejor_fitness_total:.3f}")

    # --- Carpeta y nombres de archivo con timestamp para no pisar
    #     corridas anteriores y poder comparar resultados entre sí.
    OUTPUT_DIR = "resultados_ga"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    path_mapa = os.path.join(OUTPUT_DIR, f"mapa_optimo_{timestamp}.html")
    path_grafico = os.path.join(OUTPUT_DIR, f"convergencia_ga_{timestamp}.png")
    path_json = os.path.join(OUTPUT_DIR, f"resultado_optimo_{timestamp}.json")

    graficar_convergencia(historia_mejor, historia_promedio, historia_peor, historia_std, path=path_grafico)
    generar_mapa(mejor, poblacion_final, path=path_mapa)

    with open(path_json, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": timestamp,
            "t": mejor["t"],
            "punto": mejor["detalle"]["punto"],
            "fitness_costo": mejor["detalle"]["f_costo"],
            "fitness_sismico": mejor["detalle"]["f_sismico"],
            "fitness_ciudad": mejor["detalle"]["f_ciudad"],
            "fitness_total": mejor_fitness_total,
        }, f, ensure_ascii=False, indent=2)

    # --- Log acumulado de todas las corridas, para comparar resultados
    #     entre sí sin tener que abrir cada JSON individual.
    path_log = os.path.join(OUTPUT_DIR, "log_corridas.csv")
    existe_log = os.path.exists(path_log)
    with open(path_log, "a", encoding="utf-8") as f:
        if not existe_log:
            f.write("timestamp,t,lat,lon,fitness_costo,fitness_sismico,fitness_ciudad,"
                    "fitness_mejor,fitness_promedio,fitness_peor,fitness_std\n")
        lat, lon = mejor["detalle"]["punto"]
        f.write(f"{timestamp},{mejor['t']:.6f},{lat:.6f},{lon:.6f},"
                f"{mejor['detalle']['f_costo']:.4f},{mejor['detalle']['f_sismico']:.4f},"
                f"{mejor['detalle']['f_ciudad']:.4f},{mejor_fitness_total:.4f},"
                f"{historia_promedio[-1]:.4f},{historia_peor[-1]:.4f},{historia_std[-1]:.4f}\n")

    print(f"\nArchivos generados en '{OUTPUT_DIR}/':")
    print(f"  - {path_mapa}")
    print(f"  - {path_grafico}")
    print(f"  - {path_json}")
    print(f"  - {path_log} (acumula todas las corridas)")