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
| `relajacion` | v1 | 07 | 08, 10, 15, 25 | `relajar`: Laplace/Poisson en 2D por Gauss–Seidel rojo-negro (sobrerrelajación con `omega`), nodos fijos de Dirichlet, historia de la energía; `energia_discreta`: $U_h$. El coeficiente variable está en la herramienta `conduccion` (NB15). |
| `momentos_axial` | v1 | 05 | 08 (imágenes frente a numérico) | Método de momentos para conductores con simetría de revolución: σ, capacidad, matriz C. |
| `conduccion` | v1 | 15 | 25 (dieléctricos, con ε en lugar de σ) | `resolver_conduccion`: ∇·(κ∇φ) = −f con `spsolve`, κ en las caras = media armónica, bordes no fijos aislantes; `divergencia_flujo`: ∇·(κ∇φ). |
| `biot_savart` | v1 | 12 | 13, 14, 17, 18, 26 | `campo_poligonal`: **B** exacto de una poligonal (fórmula de cada segmento), por bloques; `espira`, `campo_espiras` (solenoide como pila de espiras). |
| `potencial_vector` | v1 | 13 | 14, 17 | `potencial_poligonal`: **A** exacto (calibre de Coulomb) de una poligonal, con ln[(R₁+R₂+ℓ)/(R₁+R₂−ℓ)] por segmento. |
| `boris` | v1 | 14 | 29 (versión relativista) | `boris`: integrador de Boris (no relativista) de M dv/dt = q(E + v×B/c); con E = 0 conserva \|v\| exactamente. |
| `fdtd1d` | v1 | 18 | 27 | `yee1d`: Maxwell en 1D (E_y en nodos, B_z en puntos medios y medio paso atrás), esquema de Yee con c = 1, número de Courant S ≤ 1 (exacto con S = 1), fuente J_y opcional, bordes absorbentes de Mur. |
