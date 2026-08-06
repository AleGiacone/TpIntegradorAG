"""
Algoritmo genetico para localizacion optima de un SMR.
Individuo = (lat, lon) -> coordenadas continuas.
Usa unicamente la libreria estandar 'random'.
"""

import random
from data import LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, CIUDADES
from fitness import fitness_poblacion, calcular_rangos_globales


def crear_individuo():
    """Genera un individuo inicial cerca de alguna de las 6 ciudades
    (mas realista que un punto puramente al azar en todo el bounding box,
    ya que el SMR debe estar en una zona con algun grado de acceso previo)."""
    _, clat, clon, _ = random.choice(CIUDADES)
    lat = clat + random.gauss(0, 0.3)   # dispersion ~ +/- 30 km
    lon = clon + random.gauss(0, 0.3)
    lat = min(max(lat, LAT_MIN), LAT_MAX)
    lon = min(max(lon, LON_MIN), LON_MAX)
    return (lat, lon)


def crear_poblacion(tam):
    return [crear_individuo() for _ in range(tam)]


def seleccion_torneo(poblacion, fitness, k=3):
    """Selecciona un individuo por torneo entre k participantes al azar."""
    participantes = random.sample(list(zip(poblacion, fitness)), k)
    ganador = max(participantes, key=lambda x: x[1])
    return ganador[0]


def cruce(padre1, padre2):
    """Cruce aritmetico (blend crossover): combinacion lineal de ambos padres."""
    alpha = random.random()
    lat = alpha * padre1[0] + (1 - alpha) * padre2[0]
    lon = alpha * padre1[1] + (1 - alpha) * padre2[1]
    return (lat, lon)


def mutacion(individuo, prob=0.2, sigma=0.15):
    """Mutacion gaussiana independiente en lat y lon."""
    lat, lon = individuo
    if random.random() < prob:
        lat += random.gauss(0, sigma)
    if random.random() < prob:
        lon += random.gauss(0, sigma)
    lat = min(max(lat, LAT_MIN), LAT_MAX)
    lon = min(max(lon, LON_MIN), LON_MAX)
    return (lat, lon)


def ejecutar_ga(tam_poblacion=60, generaciones=100, prob_cruce=0.8,
                 prob_mutacion=0.2, elitismo=2, seed=None):
    """
    Ejecuta el algoritmo genetico completo.

    Devuelve:
      historial: lista de dicts con metricas de cada generacion
      mejor: tupla (individuo, fitness, valores_crudos) de la mejor
             solucion encontrada en la ultima generacion
    """
    if seed is not None:
        random.seed(seed)

    # Rangos globales fijos: se calculan UNA vez, muestreando todo el
    # espacio de busqueda, para que el fitness sea comparable entre
    # generaciones (ver docstring de calcular_rangos_globales).
    rangos = calcular_rangos_globales(LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, seed=1)

    poblacion = crear_poblacion(tam_poblacion)
    historial = []

    for gen in range(generaciones):
        fitness, crudos = fitness_poblacion(poblacion, rangos)

        # ordenar poblacion por fitness descendente
        ordenados = sorted(zip(poblacion, fitness, crudos), key=lambda x: x[1], reverse=True)
        poblacion_ordenada = [o[0] for o in ordenados]
        fitness_ordenado = [o[1] for o in ordenados]

        mejor_ind = poblacion_ordenada[0]
        mejor_fit = fitness_ordenado[0]
        promedio_fit = sum(fitness_ordenado) / len(fitness_ordenado)
        peor_fit = fitness_ordenado[-1]

        historial.append({
            "generacion": gen,
            "mejor_fitness": mejor_fit,
            "promedio_fitness": promedio_fit,
            "peor_fitness": peor_fit,
            "mejor_lat": mejor_ind[0],
            "mejor_lon": mejor_ind[1],
        })

        # --- construir la siguiente generacion ---
        nueva_poblacion = poblacion_ordenada[:elitismo]  # elitismo
        while len(nueva_poblacion) < tam_poblacion:
            padre1 = seleccion_torneo(poblacion_ordenada, fitness_ordenado)
            padre2 = seleccion_torneo(poblacion_ordenada, fitness_ordenado)
            if random.random() < prob_cruce:
                hijo = cruce(padre1, padre2)
            else:
                hijo = padre1
            hijo = mutacion(hijo, prob=prob_mutacion)
            nueva_poblacion.append(hijo)

        poblacion = nueva_poblacion

    # evaluacion final de la ultima generacion
    fitness_final, crudos_final = fitness_poblacion(poblacion, rangos)
    ordenados_final = sorted(zip(poblacion, fitness_final, crudos_final),
                              key=lambda x: x[1], reverse=True)
    mejor = ordenados_final[0]

    return historial, mejor
