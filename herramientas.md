# Herramientas copiadas entre notebooks

Los notebooks no importan módulos compartidos, porque cada uno debe funcionar solo en Colab. Las funciones que reaparecen se **copian** y se marcan así:

```python
# herramienta: NOMBRE vN (NBxx)
...
# fin herramienta
```

`tools/verificar.py` comprueba que todas las copias de `NOMBRE vN` sean idénticas. Para cambiar una herramienta:
- si es una corrección, se actualizan todas las copias;
- si cambia la interfaz, se sube la versión (v2) en los notebooks nuevos.

| Herramienta | Versión | Creada en | Usada en | Qué hace |
|---|---|---|---|---|
| `configuracion` | v1 | 01 | todos | Importaciones, estilo de gráficos, `deslizador`, `interactuar`, `verificar`, `guardar`. |
| `relajacion` | — | 07 | 08, 10, 15, 25 | Laplace/Poisson por Gauss–Seidel rojo-negro; coeficiente variable con `spsolve`. |
| `biot_savart` | — | 12 | 13, 14, 17, 26 | Campo **B** de poligonales. |
| `boris` | — | 14 | 29 (versión relativista) | Integrador de la fuerza de Lorentz. |
| `fdtd1d` | — | 18 | 27 | Maxwell en 1D, esquema de Yee. |
