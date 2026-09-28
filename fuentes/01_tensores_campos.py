# %% [markdown]
# # Clase 1 — Vectores, tensores y campos: ecuaciones que no dependen de los ejes
#
# **Objetivos**
# - Ver la divergencia y el rotor como *deformaciones locales*: cuánto se expande una mancha y cuánto gira.
# - Descubrir que "líneas curvas" no significa "rotor distinto de cero", y al revés.
# - Comprobar numéricamente qué cambia y qué no cambia al rotar o reflejar los ejes: componentes contra invariantes, vectores contra pseudovectores.
#
# **Material relacionado:** notas de la Clase 1. Guía 1: problemas 1, 2, 4 y 5.
#
# **Cómo usar este notebook.** En cada experimento, primero **predecí** (escribí o dibujá tu respuesta), después **ejecutá** y recién entonces leé **¿qué pasó?**.

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

from scipy.linalg import expm

# %% [markdown]
# ## Experimento 1 ★ — Una mancha de tinta en tres flujos
#
# Imaginemos que el campo $\mathbf F(\mathbf x)$ es la velocidad de un fluido. Soltamos una pequeña **mancha circular** de tinta, con una **cruz** dibujada adentro, y seguimos su movimiento: cada punto se mueve según $\dot{\mathbf x} = \mathbf F(\mathbf x)$.
#
# Usamos tres campos **lineales** $\mathbf F = A\,\mathbf x$:
#
# | campo | $\mathbf F$ | $\nabla\cdot\mathbf F$ | $(\nabla\times\mathbf F)_z$ |
# |---|---|---|---|
# | expansión | $(x,\ y)$ | ? | ? |
# | rotación rígida | $(-y,\ x)$ | ? | ? |
# | **corte** | $(y,\ 0)$ | ? | ? |
#
# ### Predecí
# 1. Completá la tabla a mano.
# 2. En cada caso, ¿la mancha cambia de **área**? ¿La cruz **gira**?
# 3. En el flujo de **corte** las líneas de corriente son rectas horizontales. ¿Puede girar la cruz?

# %%
def mancha(centro, radio=0.25, n=200):
    """Puntos del borde de una mancha circular (arreglo 2 x n)."""
    t = np.linspace(0, 2 * np.pi, n)
    return np.array(centro, float)[:, None] + radio * np.array([np.cos(t), np.sin(t)])

def cruz(centro, largo=0.25):
    """Centro y extremos de dos brazos perpendiculares (arreglo 2 x 3)."""
    c = np.array(centro, float)
    return np.array([c, c + [largo, 0.0], c + [0.0, largo]]).T

def flujo_lineal(A):
    """Mapa exacto x(t) = exp(A t) x(0) para el campo lineal F = A x."""
    return lambda puntos, t: expm(np.array(A, float) * t) @ puntos

def area(borde):
    """Área encerrada por un polígono (fórmula del cordón de zapato)."""
    x, y = borde
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def rotacion_media(flujo, centro, t=1e-3, largo=1e-3):
    """Velocidad angular media de dos brazos pequeños y perpendiculares (≈ rotor / 2)."""
    q = flujo(cruz(centro, largo), t)
    a1 = np.arctan2(q[1, 1] - q[1, 0], q[0, 1] - q[0, 0])
    a2 = np.arctan2(q[1, 2] - q[1, 0], q[0, 2] - q[0, 0]) - np.pi / 2
    return (a1 + a2) / 2 / t

def panel(ax, flujo, campo, titulo, centro=(0.9, 0.3), tiempos=(0.0, 0.4, 0.8, 1.2)):
    x = np.linspace(-2, 2, 41)
    X, Y = np.meshgrid(x, x)
    Fx, Fy = campo(X, Y)
    ax.streamplot(X, Y, Fx, Fy, color="0.8", density=0.9, linewidth=0.8, arrowsize=0.8)
    for t, color in zip(tiempos, RAMPA):
        ax.fill(*flujo(mancha(centro), t), color=color, alpha=0.45, lw=0)
        q = flujo(cruz(centro), t)
        for k in (1, 2):
            ax.plot(q[0, [0, k]], q[1, [0, k]], color="k", lw=1.5)
    ax.set(xlim=(-2, 2), ylim=(-2, 2), aspect="equal", title=titulo)
    ax.set_xticks([]); ax.set_yticks([])

casos = {   # nombre: (matriz A, campo, punto de partida de la mancha)
    "expansión  (x, y)":       ([[1, 0], [0, 1]], lambda X, Y: (X, Y), (0.3, 0.1)),
    "rotación rígida (−y, x)": ([[0, -1], [1, 0]], lambda X, Y: (-Y, X), (0.9, 0.3)),
    "corte  (y, 0)":           ([[0, 1], [0, 0]], lambda X, Y: (Y, 0 * X), (-1.1, 0.9)),
}
fig, axs = plt.subplots(1, 3, figsize=(13, 4.6))
for ax, (nombre, (A, campo, centro)) in zip(axs, casos.items()):
    tiempos = (0.0, 0.3, 0.6, 0.9) if "expansión" in nombre else (0.0, 0.4, 0.8, 1.2)
    panel(ax, flujo_lineal(A), campo, nombre, centro=centro, tiempos=tiempos)
fig.suptitle("La mancha a cuatro tiempos sucesivos (más oscuro = más tarde)", y=1.0)
plt.tight_layout(); guardar(fig, "nb01_manchas"); plt.show()

print(f"{'campo':26s} {'div F':>6s} {'rot F':>6s} {'área(t=1)/área(0)':>18s} {'giro medio':>11s}")
for nombre, (A, _, _) in casos.items():
    A = np.array(A, float)
    f = flujo_lineal(A)
    razon = area(f(mancha((0.9, 0.3)), 1.0)) / area(mancha((0.9, 0.3)))
    print(f"{nombre:26s} {np.trace(A):6.1f} {A[1, 0] - A[0, 1]:6.1f} {razon:18.3f} {rotacion_media(f, (0.9, 0.3)):11.3f}")

# %% [markdown]
# ### ¿Qué pasó?
# - **Expansión:** la mancha crece pero la cruz **no gira**. El área crece como $e^{(\nabla\cdot\mathbf F)\,t} = e^{2t}$.
# - **Rotación rígida:** la mancha da vueltas alrededor del origen **y además gira sobre sí misma**. El área se conserva porque la divergencia es nula.
# - **Corte:** ¡la cruz **gira**, aunque las líneas son rectas! El brazo horizontal no se mueve, pero el vertical se inclina. El giro medio de los dos brazos es $\tfrac12(\nabla\times\mathbf F)_z = -\tfrac12$.
#
# **La divergencia mide cuánto se expande un volumen pequeño, y el rotor cuánto gira** (la mitad del rotor es la velocidad angular local). Ninguno de los dos se lee en la forma global de las líneas de campo.
#
# Verificamos los números contra las fórmulas:

# %%
f_exp = flujo_lineal([[1, 0], [0, 1]])
verificar("área(1)/área(0) en la expansión = e^2", area(f_exp(mancha((0.9, 0.3)), 1.0)) / area(mancha((0.9, 0.3))), np.exp(2), tol=1e-3)
verificar("giro medio en el corte = rot/2 = -1/2", rotacion_media(flujo_lineal([[0, 1], [0, 0]]), (0.9, 0.3)), -0.5, tol=1e-2)
verificar("giro medio en la rotación rígida = 1", rotacion_media(flujo_lineal([[0, -1], [1, 0]]), (0.9, 0.3)), 1.0, tol=1e-2)

# %% [markdown]
# ### Ahora vos: cualquier campo lineal
# Con $A = \begin{pmatrix} a & b \\ c & d\end{pmatrix}$ el campo es $\mathbf F = (a x + b y,\ c x + d y)$. Entonces $\nabla\cdot\mathbf F = a + d$ (la **traza**) y $(\nabla\times\mathbf F)_z = c - b$ (la **parte antisimétrica**).
#
# Probá con $b = -c$ (rotación pura), $b = c$ (deformación sin giro) y $a = -d$ (deformación sin cambio de área).

# %%
def dibujar_lineal(a=0.0, b=1.0, c=0.0, d=0.0, t=1.0):
    A = [[a, b], [c, d]]
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    panel(ax, flujo_lineal(A), lambda X, Y: (a * X + b * Y, c * X + d * Y),
          f"div = {a + d:.2f},  rot = {c - b:.2f}", tiempos=np.linspace(0, t, 4))
    plt.show()

interactuar(dibujar_lineal, a=deslizador("a", 0.0, -1, 1, 0.1), b=deslizador("b", 1.0, -1, 1, 0.1),
            c=deslizador("c", 0.0, -1, 1, 0.1), d=deslizador("d", 0.0, -1, 1, 0.1),
            t=deslizador("t", 1.0, 0.1, 2, 0.1))

# %% [markdown]
# ## Experimento 2 — Dos remolinos con las mismas líneas
#
# Comparamos dos campos cuyas líneas de corriente son **circunferencias**:
# - la rotación rígida $\mathbf F = (-y,\ x)$;
# - el **vórtice irrotacional** $\mathbf F = \dfrac{(-y,\ x)}{x^2+y^2} = \dfrac{\hat{\boldsymbol\varphi}}{s}$.
#
# ### Predecí
# En ambos casos la mancha da vueltas alrededor del origen. ¿En cuál **gira la cruz sobre sí misma**? Calculá el rotor del vórtice para $s \neq 0$.

# %%
def flujo_vortice(puntos, t):
    """Vórtice irrotacional: cada punto gira alrededor del origen con velocidad angular 1/s²."""
    x, y = puntos
    s2 = x**2 + y**2
    ang = t / s2
    return np.array([x * np.cos(ang) - y * np.sin(ang), x * np.sin(ang) + y * np.cos(ang)])

def campo_vortice(X, Y):
    s2 = X**2 + Y**2 + 1e-9
    return -Y / s2, X / s2

fig, axs = plt.subplots(1, 2, figsize=(10, 4.8))
panel(axs[0], flujo_lineal([[0, -1], [1, 0]]), lambda X, Y: (-Y, X), "rotación rígida", centro=(1.2, 0), tiempos=(0, 0.5, 1.0, 1.5))
panel(axs[1], flujo_vortice, campo_vortice, "vórtice irrotacional", centro=(1.2, 0), tiempos=(0, 0.5, 1.0, 1.5))
plt.tight_layout(); guardar(fig, "nb01_vortices"); plt.show()

verificar("giro medio en el vórtice irrotacional = 0", rotacion_media(flujo_vortice, (1.2, 0.0)), 0.0, tol=1e-2)

# %% [markdown]
# ### ¿Qué pasó?
# En el vórtice irrotacional la mancha **da la vuelta sin girar sobre sí misma**. El borde interno va más rápido que el externo, y ese corte compensa exactamente la rotación que produciría la curvatura de las líneas. El rotor es nulo en todo punto con $s\neq 0$, y sin embargo la circulación alrededor del origen vale $2\pi$: toda la "rotación" está concentrada en el eje. Es la forma del campo magnético de un hilo con corriente (Clases 12 y 13), y la situación se describe con $\nabla\times\mathbf F = 2\pi\,\delta(x)\delta(y)\,\hat z$ (Clase 2).

# %% [markdown]
# ## Experimento 3 — Rotar los ejes: qué cambia y qué no
#
# Tomamos una rotación $R$ al azar en 3D ($R^TR = I$, $\det R = +1$) y una reflexión $P$ ($\det P = -1$). Transformamos vectores $v_i' = R_{ij}v_j$ y un tensor $T'_{ij} = R_{ik}R_{j\ell}T_{k\ell}$ con `np.einsum`, que usa exactamente la convención de índices repetidos.
#
# ### Predecí
# ¿Cuáles de estas cantidades cambian al rotar? $v_1$, $\ |\mathbf v|$, $\ \mathbf v\cdot\mathbf w$, $\ T_{12}$, $\ T_{ii}$, $\ \det T$, $\ T_{ij}T_{ij}$.

# %%
rng = np.random.default_rng(1)

def rotacion_al_azar():
    Q, Rt = np.linalg.qr(rng.normal(size=(3, 3)))
    Q = Q @ np.diag(np.sign(np.diag(Rt)))
    return Q if np.linalg.det(Q) > 0 else -Q

R = rotacion_al_azar()
P = np.diag([-1.0, 1.0, 1.0])                        # espejo x -> -x
v, w = rng.normal(size=3), rng.normal(size=3)
T = rng.normal(size=(3, 3))

vp = np.einsum("ij,j->i", R, v)
wp = np.einsum("ij,j->i", R, w)
Tp = np.einsum("ik,jl,kl->ij", R, R, T)

filas = [("v_1", v[0], vp[0]), ("|v|", np.linalg.norm(v), np.linalg.norm(vp)),
         ("v·w", v @ w, vp @ wp), ("T_12", T[0, 1], Tp[0, 1]),
         ("T_ii (traza)", np.trace(T), np.trace(Tp)), ("det T", np.linalg.det(T), np.linalg.det(Tp)),
         ("T_ij T_ij", np.sum(T * T), np.sum(Tp * Tp))]
print(f"{'cantidad':14s} {'original':>10s} {'rotada':>10s}")
for nombre, a, b in filas:
    print(f"{nombre:14s} {a:10.4f} {b:10.4f}   {'invariante' if np.isclose(a, b) else 'cambia'}")

# %% [markdown]
# ### ¿Qué pasó?
# Las **componentes** ($v_1$, $T_{12}$) cambian. Las **contracciones completas** ($\mathbf v\cdot\mathbf w = v_iw_i$, $T_{ii}$, $T_{ij}T_{ij}$), las normas y $\det T$ son escalares: no cambian. Por eso una ley física escrita como igualdad entre tensores vale en todos los sistemas de ejes.
#
# ### Producto vectorial ante una reflexión
# Si $\mathbf C = \mathbf v\times\mathbf w$ fuera un vector común, al reflejar debería transformar como $P\mathbf C$. Comparemos con el producto vectorial de los vectores reflejados.

# %%
for nombre, M in [("rotación R", R), ("reflexión P", P)]:
    izq = np.cross(M @ v, M @ w)               # producto de los vectores transformados
    der_vector = M @ np.cross(v, w)            # si fuera un vector polar
    der_pseudo = np.linalg.det(M) * der_vector # si es un pseudovector
    print(f"{nombre:12s}: (Mv)×(Mw) = {np.round(izq, 3)}  |  M(v×w) = {np.round(der_vector, 3)}  |  det(M) M(v×w) = {np.round(der_pseudo, 3)}")

verificar("(Pv)×(Pw) = det(P) P(v×w)  [componente x]", np.cross(P @ v, P @ w)[0], (np.linalg.det(P) * P @ np.cross(v, w))[0])

# %% [markdown]
# El producto vectorial transforma con un $\det(M)$ adicional: es un **pseudovector**. Con una rotación no se nota ($\det R = 1$); con un espejo, sí. La velocidad angular, el momento angular y el campo magnético $\mathbf B$ son pseudovectores.
#
# ## Experimento 4 — El símbolo de Levi-Civita con `einsum`
# Construimos $\epsilon_{ijk}$ y verificamos, componente por componente, la identidad
# $$\epsilon_{ijk}\epsilon_{imn} = \delta_{jm}\delta_{kn} - \delta_{jn}\delta_{km}$$
# y la ley de transformación $R_{im}R_{jn}R_{kp}\,\epsilon_{mnp} = \det(R)\,\epsilon_{ijk}$.

# %%
eps = np.zeros((3, 3, 3))
for i, j, k in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
    eps[i, j, k], eps[i, k, j] = 1.0, -1.0
delta = np.eye(3)

izq = np.einsum("ijk,imn->jkmn", eps, eps)
der = np.einsum("jm,kn->jkmn", delta, delta) - np.einsum("jn,km->jkmn", delta, delta)
verificar("max |εε − (δδ − δδ)| sobre las 81 componentes", np.abs(izq - der).max(), 0.0, tol=1e-12)

for nombre, M in [("rotación", R), ("reflexión", P)]:
    eps_t = np.einsum("im,jn,kp,mnp->ijk", M, M, M, eps)
    print(f"{nombre}: R R R ε = {np.linalg.det(M):+.0f} × ε ?", np.allclose(eps_t, np.linalg.det(M) * eps))

# %% [markdown]
# ## Explorá
# 1. **Deformación pura (Guía 1, P1c).** En el Experimento 1 poné $a=1$, $d=-1$, $b=c=0$. ¿Cambia el área? ¿Gira la cruz? Escribí $A = S + \Omega$, con $S$ simétrica y $\Omega$ antisimétrica, e identificá qué parte produce cada efecto.
# 2. **Guía 1, P2.** Transformá con `einsum` el tensor $u = \begin{pmatrix}1&0\\-3&2\end{pmatrix}$ con la transformación del enunciado. Calculá su determinante: ¿es una rotación o una transformación impropia? ¿Qué cambia si $u$ es un pseudotensor?
# 3. **Guía 1, P4.** Verificá numéricamente $\epsilon_{ijk}\epsilon_{ijl} = 2\delta_{kl}$ y $\epsilon_{ijk}\epsilon_{ijk} = 6$.
# 4. **Vórtice de Rankine.** Programá un campo que sea rotación rígida para $s<1$ y vórtice irrotacional para $s>1$. ¿Dónde gira la cruz? Dibujá $(\nabla\times\mathbf F)_z$ en función de $s$.
# 5. **Guía 1, P5.** Tomá $\mathbf E$ (vector polar) y $\mathbf B$ (pseudovector) al azar y reflejalos con $P$. ¿Qué pasa con $\mathbf E\cdot\mathbf E$, $\mathbf B\cdot\mathbf B$ y $\mathbf E\cdot\mathbf B$? ¿Cuál es un **pseudoescalar**?
