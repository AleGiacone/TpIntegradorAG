# Optimización del despliegue de una SMR con Algoritmos Genéticos

Trabajo Práctico Integrador — Cátedra Algoritmos Genéticos, UTN Facultad Regional Rosario (2026).

Prototipo que ubica una Central Nuclear Modular Pequeña (SMR) a lo largo del corredor
Río Paraná → Río de la Plata → costa bonaerense, minimizando el costo de construcción
y la densidad poblacional expuesta.

## Cómo funciona

Cada individuo del algoritmo genético es un único número `t ∈ [0, 1]` que se traduce a
un punto real sobre una polilínea de 40 vértices que recorre el corredor. El fitness de
cada punto combina:

- **Costo de construcción**: tendido eléctrico hasta la estación transformadora más
  cercana + envío marítimo de materiales desde el puerto más cercano.
- **Densidad poblacional**: densidad máxima en dos anillos alrededor del sitio, sobre
  datos reales de población (GHS-POP) — anillo de 0-2 millas (restricción dura) y de
  2-10 millas (restricción blanda).

El detalle completo del modelo, la justificación de cada variable y los parámetros del
algoritmo están en el documento `G14_Concrecion_del_modelo` en la seccion de documentacion.

## Requisitos

- Python 3.10+
- Dependencias:

  ```bash
  pip install numpy matplotlib folium rasterio pyproj Pillow
  ```

## Datos necesarios: tiles GHS-POP

El script necesita los tiles de población GHS-POP (Comisión Europea / JRC) para
calcular la densidad poblacional. **No se incluyen en este repositorio** por su tamaño.

1. Descargarlos desde: https://human-settlement.emergency.copernicus.eu/ghs_pop2023.php
   - Producto: `GHS_POP_E2030_GLOBE_R2023A_54009_1000_V1_0`
   - Tiles necesarios para este corredor: `R13_C12`, `R13_C13`, `R14_C12`, `R14_C13`
2. Crear una carpeta `ghs_pop_tiles/` junto a `ga_central_nuclear.py` y colocar ahí los
   `.tif` descargados.

```
.
├── ga_central_nuclear.py
└── ghs_pop_tiles/
    ├── GHS_POP_E2030_GLOBE_R2023A_54009_1000_V1_0_R13_C12.tif
    ├── GHS_POP_E2030_GLOBE_R2023A_54009_1000_V1_0_R13_C13.tif
    ├── GHS_POP_E2030_GLOBE_R2023A_54009_1000_V1_0_R14_C12.tif
    └── GHS_POP_E2030_GLOBE_R2023A_54009_1000_V1_0_R14_C13.tif
```

## Ejecución

```bash
python3 ga_central_nuclear.py
```

Cada corrida usa una semilla aleatoria distinta, por lo que el resultado puede variar
levemente entre ejecuciones. Para reproducir una corrida exacta, reemplazar en el bloque
`if __name__ == "__main__":` las líneas `random.seed()` y `np.random.seed()` por una
semilla fija (por ejemplo `random.seed(42)`).

## Salidas

Cada corrida genera, en `resultados_ga/`, archivos con marca de tiempo para no pisar
corridas anteriores:

| Archivo | Contenido |
|---|---|
| `mapa_optimo_<fecha>.html` | Mapa interactivo (Leaflet/Folium): densidad poblacional (GHS-POP), corredor, ciudades, estaciones transformadoras, puertos, aeropuertos, dispersión de la última generación y punto óptimo con su detalle |
| `convergencia_ga_<fecha>.png` | Evolución del mejor, promedio y peor fitness por generación |
| `resultado_optimo_<fecha>.json` | Coordenadas, costo, densidades y fitness del mejor individuo de toda la corrida |
| `log_corridas.csv` | Una fila por corrida, para comparar resultados entre ejecuciones |

## Parámetros principales

| Parámetro | Valor |
|---|---|
| Tamaño de población | 60 |
| Generaciones | 100 |
| Selección | Torneo, k = 2 |
| Cruza | Aritmética, p = 0,75 |
| Mutación | Gaussiana, p = 0,25, σ = 0,25 |
| Pesos del fitness | costo 0,5 / poblacional 0,5 |
| Pesos de densidad | anillo duro 0,65 / anillo blando 0,35 |
| Umbrales de densidad | 25 hab/km² (0-2 mi) / 400 hab/km² (2-10 mi) |
| Penalización por restricción dura | ×0,02 |

Los costos unitarios (300.000 USD/km de tendido, 5.000 USD/km de transporte marítimo)
son valores de referencia; ver el documento de concrecion del modelo para la justificación y
las fuentes.

## Limitaciones conocidas

- Los costos unitarios son aproximados; conviene reemplazarlos por cotizaciones reales
  (CAMMESA, entes reguladores provinciales) para un uso real.
- Las distancias a estaciones transformadoras y puertos se calculan en línea recta
  (Haversine), no sobre el trazado real de la línea ni la ruta fluvial.
- El costo se normaliza dentro de cada generación, por lo que `f_costo` no es
  directamente comparable en términos absolutos entre corridas distintas.
- No se modela el riesgo sísmico: todo el corredor cae en la Zona 0 (muy baja
  peligrosidad) del reglamento INPRES-CIRSOC 103, por lo que no aporta como variable
  discriminante en esta región.
- El reemplazo generacional no tiene elitismo; el mejor individuo de toda la corrida se
  rastrea aparte (no necesariamente es el de la última generación).

## Integrantes — Grupo 14, Comisión 303

- Ribotta, Tomás (52309)
- Mazalan, Ariel (52867)
- Giacone, Alessandro (52664)
- Busnadiego, Martin (53020)
