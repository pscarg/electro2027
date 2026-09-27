# Electromagnetismo 2027: notebooks

Estos notebooks acompañan las 30 clases de Electromagnetismo 2027. Cada uno es **autocontenido**: se abre en Google Colab y se ejecuta sin instalar nada. Están en castellano y usan unidades gaussianas.

**Cómo usarlos.** Cada experimento sigue tres pasos:

1. **Predecí**: antes de ejecutar, escribí o dibujá lo que esperás ver.
2. **Ejecutá**: corré la celda y mové los controles.
3. **¿Qué pasó?**: compará lo que obtuviste con tu predicción y con la teoría.

Al final de cada notebook hay actividades **Explorá**, conectadas con los problemas de la guía.

<!-- TABLA-INICIO -->
| Clase | Notebook | Abrir |
|---|---|---|
| 01 | Clase 1 — Vectores, tensores y campos: ¿qué es físico y qué son coordenadas? | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pscarg/electro2027/blob/main/notebooks/01_tensores_campos.ipynb) |
| 02 | Clase 2 — Coulomb, la delta de Dirac y la ley de Gauss | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pscarg/electro2027/blob/main/notebooks/02_coulomb_gauss.ipynb) |
| 03 | Clase 3 — El potencial electrostático | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pscarg/electro2027/blob/main/notebooks/03_potencial.ipynb) |
| 04 | Clase 4 — Energía electrostática y el tensor de tensiones de Maxwell | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pscarg/electro2027/blob/main/notebooks/04_energia_tensiones.ipynb) |
<!-- TABLA-FIN -->

---

## Para la cátedra

**Estructura del repositorio.**
- `notebooks/`: los notebooks que usan los estudiantes. Se guardan **sin salidas**, para que primero predigan y después miren.
- `fuentes/`: las fuentes en formato *percent* (texto plano, fácil de revisar). Los notebooks se generan con `tools/construir.py`.
- `tools/verificar.py`: ejecuta todos los notebooks y controla las verificaciones analíticas (✔/✘), el tiempo de ejecución y los enlaces desde las notas en LaTeX.
- `herramientas.md`: registro de las funciones que se copian entre notebooks.

**Publicación.**
- El repositorio es `github.com/pscarg/electro2027`. El nombre de usuario figura en una sola línea, `\newcommand{\colabbase}` de `clases/electro2027.sty`: si cambia, se edita esa línea y se corre `python tools/construir.py`, que regenera los badges.
- El repositorio git local vive **fuera** de Dropbox, en `~/repos/electro2027`, para evitar conflictos de sincronización.
- Para publicar cambios:

```
python tools/construir.py
python tools/verificar.py
bash tools/publicar.sh "mensaje del commit"
```

`publicar.sh` copia esta carpeta al repositorio local, hace el commit y lo sube. Requiere haber iniciado sesión una vez con `gh auth login`.

**Verificación local** (en un entorno con numpy, scipy, matplotlib, sympy, ipywidgets 7, nbformat y nbconvert):

```
python tools/construir.py
python tools/verificar.py            # todas
python tools/verificar.py 01 02      # algunas
```
