# %% [markdown]
# # Clase 13 — La ley de Ampère y el potencial vector
#
# **Objetivos**
# - Ver en un solenoide largo que $\mathbf B$ es uniforme adentro y casi nulo afuera, mientras el **potencial vector no se anula afuera**: $A_\varphi=\Phi/2\pi s$.
# - Comprobar la ley de Ampère con caminos arbitrarios alrededor de un circuito cerrado, y ver que falla para un tramo de corriente abierto.
# - Comparar dos calibres del mismo campo, y verificar que el potencial de Biot–Savart cumple $\nabla\cdot\mathbf A=0$.
# - Medir el salto de $\mathbf B$ al cruzar una lámina de corriente.
#
# **Material relacionado:** notas de la Clase 13. Guía 5: problemas 3 a 6 y 8.
#
# **Unidades.** Gaussianas, adimensionales: longitudes en unidades del radio del solenoide, corrientes con `I_c` $=I/c=1$.

# %%
# herramienta: configuracion v1
import os
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider, IntSlider
from IPython.display import HTML, display

COLORES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
RAMPA = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]   # un solo tono, de claro a oscuro (tiempos, órdenes)
plt.rcParams.update({
    "figure.figsize": (6.4, 4.8), "figure.dpi": 90, "font.size": 11,
    "axes.grid": True, "grid.alpha": 0.25, "axes.prop_cycle": plt.cycler(color=COLORES),
    "lines.linewidth": 2, "animation.embed_limit": 30,
})

PRUEBA = bool(os.environ.get("E2027_PRUEBA"))          # solo para la verificación automática
CARPETA_FIGURAS = os.environ.get("EXPORTAR_FIGURAS")   # solo para exportar figuras a las notas

def deslizador(descripcion, valor, minimo, maximo, paso):
    """Control deslizante que redibuja al soltarlo."""
    return FloatSlider(value=valor, min=minimo, max=maximo, step=paso,
                       description=descripcion, continuous_update=False)

def interactuar(funcion, **controles):
    """Muestra los controles y redibuja la figura cada vez que se mueven."""
    if PRUEBA:
        funcion(**{k: getattr(c, "value", c) for k, c in controles.items()})
    else:
        interact(funcion, **controles)

def verificar(nombre, obtenido, esperado, tol=1e-3):
    """Compara un resultado numérico con el analítico (error relativo; absoluto si el esperado es 0)."""
    escala = abs(esperado) if esperado != 0 else 1.0
    err = abs(obtenido - esperado) / escala
    marca = "✔" if err < tol else "✘"
    print(f"{marca} {nombre}: numérico = {obtenido:.6g}, esperado = {esperado:.6g}, error = {err:.1e}")

def guardar(fig, nombre):
    if CARPETA_FIGURAS:
        fig.savefig(os.path.join(CARPETA_FIGURAS, nombre + ".png"), dpi=150, bbox_inches="tight")
# fin herramienta

# %%
# herramienta: biot_savart v1 (NB12)
def campo_poligonal(puntos, vertices, I_c=1.0, cerrada=True, bloque=400):
    """Campo B en `puntos` (M × 3) de una corriente que recorre la poligonal de `vertices` (N × 3).

    Usa la fórmula exacta de cada segmento recto: con a = P1 − r y b = P2 − r,
        B = (I/c) (a × b) (|a| + |b|) / (|a| |b| (|a| |b| + a·b)).
    I_c = I/c. Si cerrada=True, une el último vértice con el primero.
    """
    puntos = np.atleast_2d(np.asarray(puntos, float)); V = np.asarray(vertices, float)
    P1 = V; P2 = np.roll(V, -1, axis=0) if cerrada else V[1:]
    P1 = P1 if cerrada else V[:-1]
    B = np.zeros_like(puntos)
    for i in range(0, len(puntos), bloque):                     # de a bloques, para no llenar la memoria
        r = puntos[i:i + bloque, None, :]
        a = P1[None, :, :] - r; b = P2[None, :, :] - r
        na = np.linalg.norm(a, axis=2); nb = np.linalg.norm(b, axis=2)
        den = na * nb * (na * nb + np.sum(a * b, axis=2))
        with np.errstate(divide="ignore", invalid="ignore"):
            f = np.where(den > 1e-300, (na + nb) / den, 0.0)
        B[i:i + bloque] = I_c * np.sum(np.cross(a, b) * f[..., None], axis=1)
    return B

def espira(R=1.0, N=360, z=0.0):
    """Vértices de una espira circular de radio R en el plano z (polígono de N lados, corriente antihoraria vista desde +z)."""
    t = np.linspace(0, 2 * np.pi, N, endpoint=False)
    return np.stack([R * np.cos(t), R * np.sin(t), np.full(N, z)], axis=1)

def campo_espiras(puntos, lista_z, R=1.0, N=120, I_c=1.0):
    """Campo de varias espiras coaxiales (un solenoide como pila de espiras)."""
    return sum(campo_poligonal(puntos, espira(R, N, z), I_c) for z in lista_z)
# fin herramienta

# %%
# herramienta: potencial_vector v1 (NB13)
def potencial_poligonal(puntos, vertices, I_c=1.0, cerrada=True, bloque=400):
    """Potencial vector A = (I/c) ∮ dl/|r − r'| de una poligonal, en el calibre de Coulomb.

    Para cada segmento de largo ℓ, ∫ dl/|r − r'| = ln[(R₁ + R₂ + ℓ)/(R₁ + R₂ − ℓ)], con R₁, R₂ las distancias a sus extremos.
    """
    puntos = np.atleast_2d(np.asarray(puntos, float)); V = np.asarray(vertices, float)
    P1 = V if cerrada else V[:-1]; P2 = np.roll(V, -1, axis=0) if cerrada else V[1:]
    ell = np.linalg.norm(P2 - P1, axis=1); u = (P2 - P1) / ell[:, None]
    A = np.zeros_like(puntos)
    for i in range(0, len(puntos), bloque):
        r = puntos[i:i + bloque, None, :]
        R1 = np.linalg.norm(r - P1[None], axis=2); R2 = np.linalg.norm(r - P2[None], axis=2)
        f = np.log((R1 + R2 + ell) / (R1 + R2 - ell))
        A[i:i + bloque] = I_c * f @ u
    return A
# fin herramienta

# %% [markdown]
# ## Experimento 1 ★ — Un solenoide: $\mathbf B=0$ afuera, $\mathbf A\neq0$
#
# Un solenoide de radio $R=1$ y largo $L=40$, con $n=10$ espiras por unidad de largo. Según las notas, lejos de los extremos $B_z=4\pi nI/c$ adentro y $0$ afuera, y en el calibre de Coulomb $A_\varphi=Bs/2$ adentro y $\Phi/2\pi s$ afuera, con $\Phi=B\pi R^2$.
#
# ### Predecí
# Dibujá $B_z(s)$ y $A_\varphi(s)$ en el plano medio, para $s$ de 0 a 3. ¿Dónde se anula cada uno?

# %%
n_esp, L = 10, 40
lista_z = -L / 2 + (np.arange(n_esp * L) + 0.5) / n_esp
anillos = [espira(1.0, 60, z) for z in lista_z]
ss = np.linspace(0.05, 3.0, 60)
P = np.stack([ss, 0 * ss, 0 * ss], axis=1)                  # sobre el eje x, en el plano medio
B = sum(campo_poligonal(P, v) for v in anillos)
A = sum(potencial_poligonal(P, v) for v in anillos)
B0 = 4 * np.pi * n_esp; Phi = B0 * np.pi

fig, axs = plt.subplots(1, 2, figsize=(12, 4.3))
axs[0].plot(ss, B[:, 2], "o", ms=3, color=COLORES[0], label="Biot–Savart numérico")
axs[0].plot(ss, np.where(ss < 1, B0, 0), "k--", lw=1, label="4πnI/c adentro, 0 afuera")
axs[0].set(xlabel="s / R", ylabel="B_z", title="campo en el plano medio"); axs[0].legend()
axs[1].plot(ss, A[:, 1], "o", ms=3, color=COLORES[1], label="A_φ numérico")
axs[1].plot(ss, np.where(ss < 1, B0 * ss / 2, Phi / (2 * np.pi * ss)), "k--", lw=1, label="Bs/2 adentro, Φ/2πs afuera")
axs[1].set(xlabel="s / R", ylabel="A_φ", title="potencial vector en el plano medio"); axs[1].legend()
guardar(fig, "nb13_solenoide"); plt.show()

i_in, i_out = np.argmin(abs(ss - 0.5)), np.argmin(abs(ss - 2.0))
verificar("B_z adentro = 4πnI/c", B[i_in, 2], B0, tol=3e-3)
verificar("|B| afuera (s = 2R) / B adentro", np.linalg.norm(B[i_out]) / B0, 0.0, tol=1e-2)
verificar("A_φ adentro = Bs/2 (s = 0.5R)", A[i_in, 1], B0 * ss[i_in] / 2, tol=1e-2)
verificar("A_φ afuera = Φ/2πs (s = 2R)", A[i_out, 1], Phi / (2 * np.pi * ss[i_out]), tol=1e-2)
t = np.linspace(0, 2 * np.pi, 200, endpoint=False)
circulo = np.stack([2 * np.cos(t), 2 * np.sin(t), 0 * t], axis=1)
Ac = sum(potencial_poligonal(circulo, v) for v in anillos)
dl = np.roll(circulo, -1, axis=0) - circulo
verificar("∮A·dl sobre s = 2R = Φ (el flujo, aunque ahí B ≈ 0)", np.sum(0.5 * (Ac + np.roll(Ac, -1, axis=0)) * dl), Phi, tol=1e-2)

# %% [markdown]
# ### ¿Qué pasó?
# El campo es uniforme adentro, $4\pi nI/c$, y prácticamente cero afuera. El potencial vector, en cambio, no se anula afuera: decae como $\Phi/2\pi s$, y su circulación alrededor del solenoide es el flujo $\Phi$ que está **adentro**, aunque el camino pasa por donde $\mathbf B\simeq0$.
#
# ## Experimento 2 — La ley de Ampère con caminos cualesquiera
#
# Un circuito rectangular cerrado (lados de $2\times6$, en el plano $xz$) y tres caminos: una circunferencia que rodea a uno de sus lados, un camino ondulado e inclinado que también lo rodea, y uno que no lo rodea. Después, un **tramo** de corriente abierto, sin circuito que lo cierre.
#
# ### Predecí
# ¿Qué circulación da cada camino? Para el tramo abierto, ¿depende la circulación del radio del círculo?

# %%
circuito = [[-1, 0, -3], [1, 0, -3], [1, 0, 3], [-1, 0, 3]]      # el lado x = 1 lleva corriente según +z
def circulacion(curva, vertices, cerrada=True):
    B = campo_poligonal(curva, vertices, cerrada=cerrada)
    dl = np.roll(curva, -1, axis=0) - curva
    return np.sum(0.5 * (B + np.roll(B, -1, axis=0)) * dl)

t = np.linspace(0, 2 * np.pi, 3000, endpoint=False)
caminos = {"circunferencia alrededor del lado x = 1": np.stack([1 + 0.5 * np.cos(t), 0.5 * np.sin(t), 0 * t], axis=1),
           "camino ondulado e inclinado, alrededor": np.stack([1 + 0.7 * np.cos(t), 0.4 * np.sin(t) * (1 + 0.3 * np.sin(5 * t)), 0.5 * np.cos(t)], axis=1),
           "camino que no rodea ningún lado": np.stack([0.3 * np.cos(t), 2 + 0.5 * np.sin(t), 0 * t], axis=1)}
valores = {k: circulacion(v, circuito) for k, v in caminos.items()}
for k, v in valores.items():
    print(f"{k:42s}: ∮B·dl = {v:9.5f}   (4πI/c = {4 * np.pi:.5f})")
verificar("circunferencia que rodea al circuito: 4πI/c", valores["circunferencia alrededor del lado x = 1"], 4 * np.pi, tol=1e-4)
verificar("camino ondulado que lo rodea: 4πI/c", valores["camino ondulado e inclinado, alrededor"], 4 * np.pi, tol=1e-4)
verificar("camino que no lo rodea: 0", valores["camino que no rodea ningún lado"], 0.0, tol=1e-6)

tramo = [[0, 0, -1], [0, 0, 1]]
print("\nTramo abierto de z = −1 a z = 1, circunferencias en el plano z = 0:")
for s in (0.2, 1.0, 5.0):
    curva = np.stack([s * np.cos(t), s * np.sin(t), 0 * t], axis=1)
    print(f"   radio {s:4.1f}: ∮B·dl = {circulacion(curva, tramo, cerrada=False):8.4f}")

# %% [markdown]
# ### ¿Qué pasó?
# Para el circuito cerrado, cualquier camino que lo rodea da $4\pi I/c$, y el que no lo rodea da $0$: la ley de Ampère no depende de la forma del camino. Para el tramo abierto la circulación **depende del radio**: ahí $\nabla\cdot\mathbf J\neq0$ (la carga se acumularía en los extremos) y Ampère no vale. Un tramo así solo existe por un instante, mientras se carga algo, y lo arreglamos en la Clase 18 con la corriente de desplazamiento de Maxwell.
#
# ## Experimento 3 — Calibres
#
# El campo uniforme $B\hat{\mathbf z}$ con el calibre simétrico, $\mathbf A=\frac B2(-y,x,0)$, y con el de Landau, $\mathbf A'=Bx\,\hat{\mathbf y}$. Calculamos el rotor de cada uno con diferencias finitas en una grilla. Después verificamos que el potencial de una espira (calculado con la herramienta, que usa la fórmula de Biot–Savart para $\mathbf A$) tiene divergencia nula.
#
# ### Predecí
# ¿Dan el mismo $\mathbf B$? ¿Tienen la misma divergencia?

# %%
Bv = 1.0
x = np.linspace(-1, 1, 41); h = x[1] - x[0]; X, Y = np.meshgrid(x, x, indexing="ij")
Ax_s, Ay_s = -Bv / 2 * Y, Bv / 2 * X
Ax_l, Ay_l = 0 * X, Bv * X
rot = lambda Ax, Ay: np.gradient(Ay, h, axis=0) - np.gradient(Ax, h, axis=1)
fig, axs = plt.subplots(1, 2, figsize=(10, 4.4))
for ax, (Ax_, Ay_), nombre in [(axs[0], (Ax_s, Ay_s), "calibre simétrico"), (axs[1], (Ax_l, Ay_l), "calibre de Landau")]:
    ax.quiver(X[::4, ::4], Y[::4, ::4], Ax_[::4, ::4], Ay_[::4, ::4], color=COLORES[0])
    ax.set(aspect="equal", title=f"{nombre}: A (B_z = {rot(Ax_, Ay_).mean():.3f})"); ax.grid(False)
plt.show()
verificar("rotor del calibre simétrico = B", np.abs(rot(Ax_s, Ay_s) - Bv).max(), 0.0, tol=1e-12)
verificar("rotor del calibre de Landau = B", np.abs(rot(Ax_l, Ay_l) - Bv).max(), 0.0, tol=1e-12)

rng = np.random.default_rng(2); hh = 1e-4; esp = espira(1.0, 360)
div = []
for p in rng.uniform(-1.5, 1.5, (15, 3)):
    if abs(np.hypot(p[0], p[1]) - 1) < 0.2 and abs(p[2]) < 0.2:
        continue
    d = sum((potencial_poligonal([p + hh * e], esp)[0, k] - potencial_poligonal([p - hh * e], esp)[0, k]) / (2 * hh) for k, e in enumerate(np.eye(3)))
    div.append(abs(d) / np.linalg.norm(potencial_poligonal([p], esp)[0]))
verificar("∇·A / |A| para el potencial de una espira (calibre de Coulomb)", max(div), 0.0, tol=1e-6)

# %% [markdown]
# ### ¿Qué pasó?
# Los dos potenciales son muy distintos (uno gira alrededor del origen, el otro apunta siempre según $y$ y crece con $x$), pero los dos dan exactamente el mismo campo: difieren en el gradiente de $\chi=\frac B2xy$. El potencial que sale de Biot–Savart tiene divergencia nula, como dedujimos en las notas usando $\nabla\cdot\mathbf J=0$.
#
# ## Experimento 4 — El salto en una lámina de corriente
#
# Una lámina en el plano $z=0$, hecha de muchos hilos paralelos a $x$, muy juntos (separación $0.05$), con densidad superficial $K/c=1$ según $+\hat{\mathbf x}$. Medimos $\mathbf B$ a una pequeña distancia arriba y abajo, en el centro.
#
# ### Predecí
# ¿Qué componentes de $\mathbf B$ saltan, y cuánto?

# %%
delta, ancho, largo = 0.05, 200.0, 400.0
ys = np.arange(-ancho / 2, ancho / 2 + 1e-9, delta)
def campo_lamina(P):
    total = np.zeros((len(P), 3))
    for y0 in ys:
        total += campo_poligonal(P, [[-largo / 2, y0, 0], [largo / 2, y0, 0]], I_c=delta, cerrada=False)
    return total
zs = np.array([0.5, -0.5])
Bl = campo_lamina(np.stack([0 * zs, 0.013 + 0 * zs, zs], axis=1))
print(f"arriba (z = +0.5): B = {np.round(Bl[0], 5)}")
print(f"abajo  (z = −0.5): B = {np.round(Bl[1], 5)}")
verificar("salto de B_y = −4πK/c", Bl[0, 1] - Bl[1, 1], -4 * np.pi, tol=1e-2)
verificar("B_z (normal) continua: B_z(arriba) − B_z(abajo)", Bl[0, 2] - Bl[1, 2], 0.0, tol=1e-6)

# %% [markdown]
# ### ¿Qué pasó?
# $B_y$, la componente tangencial perpendicular a la corriente, salta $-4\pi K/c$: vale $\mp2\pi K/c$ a cada lado. $B_x$ y $B_z$ no saltan. Es $\mathbf B_2-\mathbf B_1=\frac{4\pi}{c}\mathbf K\times\hat{\mathbf n}$. (A una distancia de la lámina mucho mayor que la separación entre hilos, la lámina "granulada" ya no se distingue de una continua: es el mismo efecto de la Clase 2 con los hilos cargados.)
#
# ## Explorá
# 1. **Guía 5, P3.** Armá un cilindro macizo y uno hueco con muchos hilos paralelos, y dos hilos con corrientes $\pm I$; calculá $\mathbf B$ con `campo_poligonal` y comparalo con lo que te dio Ampère.
# 2. **Guía 5, P6.** Armá un toroide con muchas espiras rectangulares (o de otra sección) alrededor del eje $z$, y verificá tu resultado adentro y afuera. ¿Depende el campo de la forma de la sección?
# 3. **Guía 5, P8.** Una línea de flujo es el límite de un solenoide muy fino y largo. Calculá su potencial con `potencial_poligonal` (radio $0.01$, largo $200$) y comparalo con los dos potenciales que encontraste. Verificá con diferencias finitas que el rotor de cada uno se anula fuera de la línea.
# 4. **Guía 5, P5.** Un cilindro con corriente uniforme y una cavidad cilíndrica descentrada: armalo con hilos (los de la cavidad con corriente opuesta) y mirá cómo es el campo dentro de la cavidad.
