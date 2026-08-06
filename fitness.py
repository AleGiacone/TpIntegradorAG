"""
Funciones de evaluacion (heuristicas) para cada uno de los 4 objetivos
especificos del proyecto, y la funcion de aptitud (fitness) combinada
que usa el algoritmo genetico.
"""

from data import (
    FUENTES_AGUA, SUBESTACIONES, COSTO_USD_POR_KM_ELECTRICO,
    RUTAS, PUERTOS, COSTO_CAMINO_USD_KM, COSTO_TRANSPORTE_USD_TON_KM,
    TONELADAS_MATERIAL_IMPORTADO, CIUDADES, RADIO_MAX_POBLACION_KM,
    haversine, punto_en_zona_construida,
)


# --- Objetivo 1: acceso a agua --------------------------------------------
import math

def score_agua(lat, lon, escala_km=8):
    """
    0-1, mayor es mejor. Usa decaimiento EXPONENCIAL (no lineal) porque
    en la practica una central necesita una toma de agua directa: estar
    a 5 km del agua es mucho mejor que estar a 30 km, y estar a 30 km es
    casi tan malo como estar a 100 km (la obra de toma ya es inviable).

    escala_km controla que tan rapido cae el score: a distancia = escala_km,
    el score ya cayo a ~37% (1/e). Con escala_km=8, estar pegado al agua
    (0-2 km) da score cercano a 1, y a partir de ~20-25 km el score ya es
    marginal, empujando al algoritmo a preferir ubicaciones sobre la costa
    o la orilla de un rio/lago, como ocurre en la mayoria de las centrales
    reales.
    """
    mejor = 0
    for _, flat, flon, caudal in FUENTES_AGUA:
        d = haversine(lat, lon, flat, flon)
        cercania = math.exp(-d / escala_km)
        mejor = max(mejor, cercania * caudal)
    return mejor


# --- Objetivo 2: infraestructura electrica --------------------------------
def costo_electrico(lat, lon):
    """Costo estimado en USD de conectar el sitio a la red. Menor es mejor."""
    mejor_costo = None
    for _, slat, slon, kv in SUBESTACIONES:
        d = haversine(lat, lon, slat, slon)
        costo = d * COSTO_USD_POR_KM_ELECTRICO[kv]
        if mejor_costo is None or costo < mejor_costo:
            mejor_costo = costo
    return mejor_costo


# --- Objetivo 3: rutas y puertos ------------------------------------------
def costo_logistico(lat, lon):
    """Costo estimado en USD (acceso vial + transporte desde puerto). Menor es mejor."""
    d_ruta = min(haversine(lat, lon, rlat, rlon) for _, rlat, rlon in RUTAS)
    costo_acceso = d_ruta * COSTO_CAMINO_USD_KM

    d_puerto = min(haversine(lat, lon, plat, plon) for _, plat, plon in PUERTOS)
    costo_transporte = d_puerto * COSTO_TRANSPORTE_USD_TON_KM * TONELADAS_MATERIAL_IMPORTADO

    return costo_acceso + costo_transporte


# --- Objetivo 4: poblacion abastecida --------------------------------------
def poblacion_abastecida(lat, lon):
    """Habitantes 'efectivos' cubiertos. Mayor es mejor."""
    total = 0
    for _, clat, clon, pob in CIUDADES:
        d = haversine(lat, lon, clat, clon)
        if d <= RADIO_MAX_POBLACION_KM:
            peso = 1 - (d / RADIO_MAX_POBLACION_KM)
            total += pob * peso
    return total


def evaluar_individuo(lat, lon):
    """Diccionario con los 4 valores crudos (sin normalizar) de un punto."""
    return {
        "agua": score_agua(lat, lon),
        "costo_electrico": costo_electrico(lat, lon),
        "costo_logistico": costo_logistico(lat, lon),
        "poblacion": poblacion_abastecida(lat, lon),
    }


def normalizar_valor(v, vmin, vmax, invertir=False):
    """Normaliza un unico valor a [0, 1] usando limites (vmin, vmax) fijos."""
    if vmax == vmin:
        return 1.0
    x = (v - vmin) / (vmax - vmin)
    x = min(max(x, 0.0), 1.0)  # recorta por si el valor cae fuera del rango muestreado
    return 1 - x if invertir else x


# Pesos de cada objetivo en la funcion de aptitud combinada (deben sumar 1)
PESOS = {
    "agua": 0.25,
    "costo_electrico": 0.25,
    "costo_logistico": 0.25,
    "poblacion": 0.25,
}


def calcular_rangos_globales(lat_min, lat_max, lon_min, lon_max, n_muestras=3000, seed=1):
    """
    Muestrea puntos al azar en TODO el espacio de busqueda para estimar el
    rango (min, max) real de cada objetivo. Esto se calcula UNA sola vez,
    antes de correr el algoritmo genetico, y se usa como referencia fija
    para normalizar en todas las generaciones. Sin esto, normalizar por
    generacion (min-max de la poblacion actual) hace que siempre exista
    un "mejor" cercano a 1 y un "peor" cercano a 0, aunque la poblacion
    ya haya convergido -> el grafico de evolucion no mostraria convergencia
    real, solo el ranking relativo dentro de cada generacion.
    """
    import random
    rnd = random.Random(seed)
    muestras = [(rnd.uniform(lat_min, lat_max), rnd.uniform(lon_min, lon_max))
                for _ in range(n_muestras)]
    crudos = [evaluar_individuo(lat, lon) for lat, lon in muestras]

    rangos = {}
    for clave in ["agua", "costo_electrico", "costo_logistico", "poblacion"]:
        valores = [c[clave] for c in crudos]
        rangos[clave] = (min(valores), max(valores))
    return rangos


def fitness_poblacion(individuos, rangos):
    """
    Recibe una lista de individuos [(lat, lon), ...] y los rangos globales
    fijos (de calcular_rangos_globales). Devuelve:
      - lista de fitness (uno por individuo, normalizado 0-1 de forma
        COMPARABLE entre generaciones)
      - lista de valores crudos (para inspeccion/depuracion)

    Los individuos que caen dentro de una zona ya construida (ciudad o
    puerto existente, ver punto_en_zona_construida) reciben fitness 0,
    ya que ese terreno no esta disponible para construir el SMR.
    """
    crudos = [evaluar_individuo(lat, lon) for lat, lon in individuos]

    fitness = []
    for (lat, lon), c in zip(individuos, crudos):
        if punto_en_zona_construida(lat, lon):
            fitness.append(0.0)
            continue
        a = normalizar_valor(c["agua"], *rangos["agua"])
        e = normalizar_valor(c["costo_electrico"], *rangos["costo_electrico"], invertir=True)
        l = normalizar_valor(c["costo_logistico"], *rangos["costo_logistico"], invertir=True)
        p = normalizar_valor(c["poblacion"], *rangos["poblacion"])
        f = (PESOS["agua"] * a + PESOS["costo_electrico"] * e +
             PESOS["costo_logistico"] * l + PESOS["poblacion"] * p)
        fitness.append(f)
    return fitness, crudos
