# Filtrado digital de vibraciones de un motor

Proyecto didáctico que genera una vibración sintética con componentes de giro, impactos amortiguados, deriva lenta, interferencia tonal y ruido blanco. Implementa y compara filtros pasa bajos, pasa altos y pasa bandas con Python, SciPy, NumPy, pandas y Matplotlib.

## Estructura

- `analisis_vibracion.py`: construye la señal, diseña los filtros, calcula métricas y genera gráficas.
- `graficas/`: figuras resultantes del análisis.
- `resultados/`: métricas CSV y parámetros/resumen JSON.

## Ejecución

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python analisis_vibracion.py
```

La semilla fija hace que la parte aleatoria del ruido sea reproducible. El filtrado hacia adelante y atrás es útil para el análisis fuera de línea porque evita un desfase neto; para instrumentación en tiempo real se requiere un filtro causal y una evaluación de latencia.

## Publicación en GitHub

Crea un repositorio vacío, extrae el ZIP, entra a la carpeta del proyecto y ejecuta:

```bash
git init
git add .
git commit -m "Agrega análisis de filtros para vibración de motor"
git branch -M main
git remote add origin URL_DEL_REPOSITORIO
git push -u origin main
```
