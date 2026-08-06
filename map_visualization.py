"""
Genera un mapa interactivo (con imagen satelital real) mostrando:
  - la ubicacion optima encontrada por el algoritmo genetico
  - las 6 ciudades candidatas
  - un circulo indicando el radio de cobertura poblacional (150 km)

Usa la libreria 'folium', que arma el mapa a partir de tiles reales
(satelite/calles) sin necesidad de API key. El resultado es un archivo
.html que se abre en cualquier navegador.
"""

import folium
from data import CIUDADES, RADIO_MAX_POBLACION_KM


def generar_mapa(mejor_lat, mejor_lon, mejor_fitness, mejor_crudo, ruta_salida):
    """Crea el mapa HTML y lo guarda en ruta_salida."""

    # Mapa centrado en la ubicacion optima, con vista satelital (Esri World Imagery)
    mapa = folium.Map(
        location=[mejor_lat, mejor_lon],
        zoom_start=7,
        tiles="Esri.WorldImagery",  # imagen satelital real
    )

    # Capa alternativa de calles/rutas, seleccionable desde el mapa
    folium.TileLayer("OpenStreetMap", name="Mapa de calles").add_to(mapa)

    # --- Marcadores de las 6 ciudades candidatas ---
    for nombre, clat, clon, poblacion in CIUDADES:
        folium.Marker(
            location=[clat, clon],
            popup=f"<b>{nombre}</b><br>Poblacion: {poblacion:,} hab.",
            tooltip=nombre,
            icon=folium.Icon(color="blue", icon="home", prefix="fa"),
        ).add_to(mapa)

    # --- Marcador de la ubicacion optima encontrada ---
    popup_html = (
        f"<b>Ubicacion optima del SMR</b><br>"
        f"Lat: {mejor_lat:.5f}, Lon: {mejor_lon:.5f}<br>"
        f"Fitness: {mejor_fitness:.4f}<br>"
        f"Score agua: {mejor_crudo['agua']:.2f}<br>"
        f"Costo electrico: USD {mejor_crudo['costo_electrico']:,.0f}<br>"
        f"Costo logistico: USD {mejor_crudo['costo_logistico']:,.0f}<br>"
        f"Poblacion abastecida: {mejor_crudo['poblacion']:,.0f} hab."
    )
    folium.Marker(
        location=[mejor_lat, mejor_lon],
        popup=popup_html,
        tooltip="Ubicacion optima del SMR",
        icon=folium.Icon(color="red", icon="star", prefix="fa"),
    ).add_to(mapa)

    # --- Circulo de referencia: radio de cobertura poblacional ---
    folium.Circle(
        location=[mejor_lat, mejor_lon],
        radius=RADIO_MAX_POBLACION_KM * 1000,  # folium usa metros
        color="red",
        fill=True,
        fill_opacity=0.08,
        popup=f"Radio de cobertura poblacional ({RADIO_MAX_POBLACION_KM} km)",
    ).add_to(mapa)

    folium.LayerControl().add_to(mapa)

    mapa.save(ruta_salida)
    return ruta_salida
