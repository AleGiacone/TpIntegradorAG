"""
Punto de entrada del proyecto: ejecuta el algoritmo genetico y genera
graficos (con pandas) que muestran como fluctua el fitness a traves
de las generaciones.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from genetic_algorithm import ejecutar_ga
from data import CIUDADES
from map_visualization import generar_mapa

OUT_DIR = "output"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1) Ejecutar el algoritmo genetico
# ---------------------------------------------------------------------------
historial, mejor = ejecutar_ga(tam_poblacion=100, generaciones=200, seed=None)
mejor_individuo, mejor_fitness, mejor_crudo = mejor

df = pd.DataFrame(historial)
df.to_csv(os.path.join(OUT_DIR, "historial_generaciones.csv"), index=False)

print("=" * 60)
print("MEJOR UBICACION ENCONTRADA")
print("=" * 60)
print(f"Latitud:   {mejor_individuo[0]:.5f}")
print(f"Longitud:  {mejor_individuo[1]:.5f}")
print(f"Fitness:   {mejor_fitness:.4f}")
print("Valores crudos por objetivo:")
for k, v in mejor_crudo.items():
    print(f"  {k}: {v:,.2f}")

# ---------------------------------------------------------------------------
# 2) Grafico: evolucion del fitness (mejor / promedio / peor) por generacion
# ---------------------------------------------------------------------------
ax = df.plot(
    x="generacion",
    y=["mejor_fitness", "promedio_fitness", "peor_fitness"],
    figsize=(10, 6),
    linewidth=2,
)
ax.set_title("Evolucion del fitness a traves de las generaciones")
ax.set_xlabel("Generacion")
ax.set_ylabel("Fitness (0 = peor, 1 = mejor)")
ax.legend(["Mejor", "Promedio", "Peor"])
ax.grid(alpha=0.3)
plt.tight_layout()
now = datetime.now().strftime("%Y%m%d_%H%M%S")
plt.savefig(os.path.join(OUT_DIR, f"evolucion_fitness_{now}.png"), dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 3) Grafico: trayectoria de la mejor solucion en el espacio (lat, lon)
# ---------------------------------------------------------------------------
fig, ax2 = plt.subplots(figsize=(8, 8))
ax2.plot(df["mejor_lon"], df["mejor_lat"], "-o", markersize=3, alpha=0.5,
          label="Trayectoria del mejor individuo")
ax2.scatter(df["mejor_lon"].iloc[-1], df["mejor_lat"].iloc[-1],
            color="red", s=150, zorder=5, label="Ubicacion final")

for nombre, clat, clon, _ in CIUDADES:
    ax2.scatter(clon, clat, color="black", marker="^", s=60)
    ax2.annotate(nombre, (clon, clat), fontsize=8, xytext=(4, 4), textcoords="offset points")

ax2.set_title("Trayectoria de la mejor solucion en el espacio de busqueda")
ax2.set_xlabel("Longitud")
ax2.set_ylabel("Latitud")
ax2.legend()
ax2.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "trayectoria_mejor_solucion.png"), dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 4) Mapa interactivo (foto satelital) con la ubicacion optima
# ---------------------------------------------------------------------------
ruta_mapa = generar_mapa(
    mejor_individuo[0], mejor_individuo[1], mejor_fitness, mejor_crudo,
    os.path.join(OUT_DIR, "mapa_ubicacion.html"),
)

print("\nArchivos generados en:", OUT_DIR)
print(" - historial_generaciones.csv")
print(" - evolucion_fitness.png")
print(" - trayectoria_mejor_solucion.png")
print(" - mapa_ubicacion.html  (abrir en el navegador)")
