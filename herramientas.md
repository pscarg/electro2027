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
| `relajacion` | v1 | 07 | 08, 10, 15, 25 | `relajar`: Laplace/Poisson en 2D por Gauss–Seidel rojo-negro (sobrerrelajación con `omega`), nodos fijos de Dirichlet, historia de la energía; `energia_discreta`: $U_h$. El coeficiente variable (clases 15 y 25) irá con `spsolve` en una herramienta aparte. |
| `momentos_axial` | v1 | 05 | 08 (imágenes frente a numérico) | Método de momentos para conductores con simetría de revolución: σ, capacidad, matriz C. |
| `biot_savart` | v1 | 12 | 13, 14, 17, 26 | `campo_poligonal`: **B** exacto de una poligonal (fórmula de cada segmento), por bloques; `espira`, `campo_espiras` (solenoide como pila de espiras). |
| `boris` | — | 14 | 29 (versión relativista) | Integrador de la fuerza de Lorentz. |
| `fdtd1d` | — | 18 | 27 | Maxwell en 1D, esquema de Yee. |
