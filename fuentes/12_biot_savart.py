# %% [markdown]
# # Clase 12 — Corrientes, la ley de Biot–Savart y la fuerza de Lorentz
#
# **Objetivos**
# - Calcular numéricamente el campo magnético de cualquier circuito formado por segmentos rectos (una **poligonal**), con la fórmula exacta de cada segmento.
# - Ver el campo de una espira y de un solenoide finito, y comprobar los resultados de las notas: el centro de un arco y de un cuadrado, y el **extremo de un solenoide largo**.
# - Comprobar que $\nabla\cdot\mathbf B=0$ y mirar la circulación de $\mathbf B$ alrededor de un hilo (un adelanto de la Clase 13).
# - Calcular fuerzas entre corrientes, y poner números a la conexión relativista.
#
# **Material relacionado:** notas de la Clase 12. Guía 5: problemas 1 y 2.
#
# **Unidades.** Gaussianas, adimensionales: longitudes en unidades del radio de la espira o del lado del cuadrado, y campos en unidades de $I/(c\,a)$ (en el código, `I_c` $=I/c=1$).

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

# %% [markdown]
# ## Experimento 1 ★ — Espiras y solenoides
#
# Primero comprobamos la herramienta con los resultados de las notas: un segmento recto, el centro de una espira circular ($2\pi I/cR$) y el centro de una espira cuadrada ($8\sqrt2\,I/ca$). Después miramos el campo de un solenoide de radio $R=1$ y largo $L$, armado con espiras apiladas.
#
# ### Predecí
# 1. ¿Cómo son las líneas de campo de una espira? ¿Y las de un solenoide largo, adentro y afuera?
# 2. En un solenoide largo, ¿cuánto vale el campo en el eje, en el extremo, comparado con el centro?

# %%
# segmento recto de z₁ = −0.3 a z₂ = 1.7, punto a distancia s = 0.8
s, z1, z2 = 0.8, -0.3, 1.7
B = campo_poligonal([[s, 0, 0]], [[0, 0, z1], [0, 0, z2]], cerrada=False)[0]
verificar("segmento recto: (I/cs)(sin α₂ − sin α₁)", B[1], (z2 / np.hypot(s, z2) - z1 / np.hypot(s, z1)) / s, tol=1e-12)
verificar("centro de una espira circular: 2πI/cR", campo_poligonal([[0, 0, 0]], espira(1.0, 2000))[0, 2], 2 * np.pi, tol=1e-5)
cuadrado = [[0.5, 0.5, 0], [-0.5, 0.5, 0], [-0.5, -0.5, 0], [0.5, -0.5, 0]]
verificar("centro de una espira cuadrada: 8√2 I/ca", campo_poligonal([[0, 0, 0]], cuadrado)[0, 2], 8 * np.sqrt(2), tol=1e-12)

def mapa(lista_z, titulo, ax, ext=3.0, n=61):
    x = np.linspace(-ext, ext, n); z = np.linspace(-ext, ext, n)
    X, Z = np.meshgrid(x, z)
    P = np.stack([X.ravel(), 0 * X.ravel(), Z.ravel()], axis=1)
    B = campo_espiras(P, lista_z)
    Bx, Bz = B[:, 0].reshape(X.shape), B[:, 2].reshape(X.shape)
    ax.pcolormesh(X, Z, np.log10(np.hypot(Bx, Bz) + 1e-3), cmap="viridis", shading="auto", vmin=-1.5, vmax=np.log10(4 * np.pi * 10) + 0.3)
    ax.streamplot(x, z, Bx, Bz, color="w", density=1.2, linewidth=0.7, arrowsize=0.7)
    for zz in lista_z:
        ax.plot([-1, 1], [zz, zz], "o", ms=2, color=COLORES[1])
    ax.set(aspect="equal", xlabel="x", ylabel="z", title=titulo); ax.grid(False)

def solenoide(L=6.0):
    lista = -L / 2 + (np.arange(int(round(10 * L))) + 0.5) / 10   # 10 espiras por unidad de largo, en el medio de cada tramo
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    mapa([0.0], "una espira (log₁₀ |B| y dirección)", axs[0])
    mapa(lista, f"solenoide, L = {L:g}R", axs[1], ext=max(3.0, 0.6 * L + 1))
    plt.show()

interactuar(solenoide, L=deslizador("L / R", 6.0, 1.0, 12.0, 1.0))

# %% [markdown]
# Ahora el campo sobre el eje de solenoides de distinto largo, normalizado por su valor en el centro.

# %%
fig, ax = plt.subplots(figsize=(7, 4.4))
cocientes = {}
for L, c in zip([2, 5, 10, 40], RAMPA):
    lista = -L / 2 + (np.arange(10 * L) + 0.5) / 10
    zs = np.linspace(-0.75 * L, 0.75 * L, 121)
    Bz = campo_espiras(np.stack([0 * zs, 0 * zs, zs], axis=1), lista, N=60)[:, 2]
    Bc = campo_espiras([[0, 0, 0]], lista, N=60)[0, 2]
    Be = campo_espiras([[0, 0, L / 2]], lista, N=60)[0, 2]
    cocientes[L] = Be / Bc
    ax.plot(zs / L, Bz / Bc, color=c, label=f"L = {L}R: extremo/centro = {Be / Bc:.3f}")
ax.axvline(0.5, color="0.6", lw=0.8, ls=":"); ax.axhline(0.5, color="0.6", lw=0.8, ls=":")
ax.set(xlabel="z / L", ylabel="B_z / B_z(centro)", title="campo sobre el eje de un solenoide"); ax.legend(fontsize=9)
guardar(fig, "nb12_solenoide"); plt.show()
verificar("extremo/centro para L = 40R → 1/2", cocientes[40], 0.5, tol=1e-2)

# %% [markdown]
# ### ¿Qué pasó?
# La herramienta reproduce los tres resultados exactos de las notas. Las líneas de una espira rodean al alambre y, lejos, se parecen a las de un dipolo. En un solenoide largo el campo es casi uniforme adentro, débil afuera, y en el extremo vale la mitad que en el centro, como predice el argumento de simetría de las notas. Para solenoides cortos el cociente es mayor que $1/2$: el centro ya no "ve" un solenoide infinito.
#
# ## Experimento 2 — ¿Fuentes? ¿Circulación?
#
# Calculamos numéricamente $\nabla\cdot\mathbf B$ cerca de la espira, y la circulación $\oint\mathbf B\cdot d\mathbf l$ sobre caminos cerrados alrededor de un hilo largo.
#
# ### Predecí
# ¿Tiene fuentes el campo magnético? ¿La circulación depende de la forma del camino?

# %%
rng = np.random.default_rng(4)
h = 1e-4
div_max = 0.0
for p in rng.uniform(-1.5, 1.5, (20, 3)):
    if abs(np.hypot(p[0], p[1]) - 1) < 0.2 and abs(p[2]) < 0.2:
        continue
    d = sum((campo_poligonal([p + h * e], espira())[0, k] - campo_poligonal([p - h * e], espira())[0, k]) / (2 * h)
            for k, e in enumerate(np.eye(3)))
    escala = np.linalg.norm(campo_poligonal([p], espira())[0])
    div_max = max(div_max, abs(d) / escala)
verificar("|∇·B| / |B| en 20 puntos al azar (diferencias finitas)", div_max, 0.0, tol=1e-5)

hilo = [[0, 0, -500], [0, 0, 500]]
def circulacion(curva):
    """∮ B·dl sobre una curva cerrada (M × 3), con la regla del trapecio."""
    B = campo_poligonal(curva, hilo, cerrada=False)
    dl = np.roll(curva, -1, axis=0) - curva
    Bm = 0.5 * (B + np.roll(B, -1, axis=0))
    return np.sum(np.sum(Bm * dl, axis=1))

t = np.linspace(0, 2 * np.pi, 4000, endpoint=False)
caminos = {"circunferencia de radio 1": np.stack([np.cos(t), np.sin(t), 0 * t], axis=1),
           "elipse inclinada, descentrada": np.stack([0.3 + 2 * np.cos(t), 0.5 * np.sin(t), 0.4 * np.cos(t)], axis=1),
           "camino que no rodea al hilo": np.stack([3 + 0.5 * np.cos(t), 0.5 * np.sin(t), 0 * t], axis=1)}
for nombre, curva in caminos.items():
    print(f"{nombre:32s}: ∮B·dl = {circulacion(curva):.6f}   (4π = {4 * np.pi:.6f})")

# %% [markdown]
# ### ¿Qué pasó?
# La divergencia es cero en todas partes (a la precisión de las diferencias finitas): **el campo magnético no tiene fuentes**. Y la circulación alrededor del hilo da siempre lo mismo, $4\pi$ (es decir, $4\pi I/c$), cualquiera sea el camino, mientras rodee al hilo; si no lo rodea, da cero. Las dos cosas se deducen de Biot–Savart en la Clase 13: son $\nabla\cdot\mathbf B=0$ y la ley de Ampère.
#
# ## Experimento 3 — Fuerzas entre corrientes
#
# La fuerza sobre un hilo es $\mathbf F=\frac Ic\int d\mathbf l\times\mathbf B$. La calculamos para dos hilos paralelos largos ($L=400$, separados $d=1$) y para una espira cuadrada en un campo uniforme.
#
# ### Predecí
# ¿Se atraen o se repelen dos hilos con corrientes del mismo sentido? ¿Qué fuerza neta siente una espira en un campo uniforme?

# %%
def fuerza_hilo(vertices2, B_en, I2_c=1.0, n=4000, cerrada=False):
    """F = (I₂/c) ∫ dl × B, discretizando la poligonal en n tramos por segmento."""
    V = np.asarray(vertices2, float)
    if cerrada:
        V = np.vstack([V, V[:1]])
    F = np.zeros(3)
    for P, Q in zip(V[:-1], V[1:]):
        u = np.linspace(0, 1, n + 1); medios = P + np.outer(0.5 * (u[1:] + u[:-1]), Q - P)
        dl = (Q - P) / n
        F += I2_c * np.sum(np.cross(np.broadcast_to(dl, medios.shape), B_en(medios)), axis=0)
    return F

L, d = 400.0, 1.0
hilo1 = [[0, 0, -L / 2], [0, 0, L / 2]]
B1 = lambda P: campo_poligonal(P, hilo1, cerrada=False)
F_par = fuerza_hilo([[d, 0, -L / 2], [d, 0, L / 2]], B1)
F_anti = fuerza_hilo([[d, 0, L / 2], [d, 0, -L / 2]], B1)
print(f"corrientes paralelas:  F_x = {F_par[0]:.4f}   (negativa: hacia el hilo 1)")
print(f"corrientes opuestas:   F_x = {F_anti[0]:.4f}")
verificar("fuerza entre hilos largos: 2 I₁I₂ L / c²d (signo: atracción)", -F_par[0], 2 * L / d, tol=1e-2)

B_unif = lambda P: np.tile([0.3, -0.7, 1.1], (len(P), 1))
F_esp = fuerza_hilo(cuadrado, B_unif, cerrada=True, n=10)
verificar("espira en un campo uniforme: |F| = 0", np.linalg.norm(F_esp), 0.0, tol=1e-12)

# %% [markdown]
# ### ¿Qué pasó?
# Con corrientes del mismo sentido los hilos se **atraen**, con la fuerza $2I_1I_2L/c^2d$ de las notas (la pequeña diferencia viene de que los hilos no son infinitos); con corrientes opuestas se repelen. La espira en un campo uniforme no siente fuerza neta: $\oint d\mathbf l=0$. Sí siente un torque, que estudiamos en la Clase 14.
#
# ## Experimento 4 — Los números de la conexión relativista
#
# Un cable de cobre de 1 mm² con 1 A. Calculamos la velocidad de los electrones, la carga neta que aparece en el sistema de una carga que viaja con ellos, y comparamos la fuerza eléctrica en ese sistema con la magnética en el laboratorio.
#
# ### Predecí
# ¿Qué fracción de la carga de los electrones aparece como carga neta del cable en el sistema de la carga?

# %%
c = 2.998e10                       # cm/s
e = 4.803e-10                      # statC
n = 8.5e22                         # electrones de conducción por cm³ (cobre)
A = 1e-2                           # cm²
I = 2.998e9                        # 1 A en statA
u = I / (n * e * A)
lam0 = n * e * A                   # statC/cm de los electrones (y de los iones)
gam = 1 / np.sqrt(1 - (u / c)**2)
lam_prima = lam0 * gam * (u / c)**2     # λ₀(γ − 1/γ) escrito sin restar números casi iguales
s_ = 1.0; q = e
F_lab = q * (u / c) * 2 * I / (c * s_)
F_prima = 2 * q * lam_prima / s_
print(f"velocidad de los electrones: u = {u:.3g} cm/s   (u/c = {u / c:.2g})")
print(f"densidad lineal de los electrones: λ₀ = {lam0:.3g} statC/cm")
print(f"carga neta en el sistema de la carga: λ' = {lam_prima:.3g} statC/cm   (λ'/λ₀ = {lam_prima / lam0:.2g})")
print(f"fuerza magnética en el laboratorio:  {F_lab:.4g} dyn")
print(f"fuerza eléctrica en el sistema de q: {F_prima:.4g} dyn")
verificar("F' / γ = F_lab (exacto, con la transformación de la fuerza de la Clase 29)", F_prima / gam, F_lab, tol=1e-12)

# %% [markdown]
# ### ¿Qué pasó?
# Los electrones se mueven a menos de 0.1 mm/s, y la carga neta que aparece por la contracción de Lorentz es una parte en $10^{25}$ de la carga de los electrones. Aun así, la fuerza eléctrica que produce en el sistema de la carga es exactamente la fuerza magnética del laboratorio. La fuerza magnética es pequeña entre cargas sueltas, pero un cable tiene unas $10^{23}$ cargas por centímetro cúbico cuya fuerza eléctrica se cancela, y queda solo este residuo.
#
# ## Explorá
# 1. **Guía 5, P2.** Calculá con `campo_poligonal` el campo sobre el eje de una espira y de un solenoide finito, y compará con tus fórmulas. ¿Cuántos lados necesita el polígono para que la espira se parezca a un círculo?
# 2. **Bobinas de Helmholtz.** Dos espiras coaxiales de radio $R$ separadas una distancia $d$, con la misma corriente. Buscá numéricamente el $d$ que hace el campo más uniforme en el centro (segunda derivada nula), y compará con lo que sale de tu fórmula del problema 2.
# 3. **Guía 5, P1.** Un anillo de $N$ cargas puntuales que gira es una corriente "granulada". Con el campo de cargas lentas, $\mathbf B=\frac1c\mathbf v\times\mathbf E$, calculá el promedio en el tiempo del campo en el centro y compará con la espira continua. ¿Cómo depende de $N$ el campo instantáneo?
# 4. **Formas con el mismo perímetro.** Compará el campo en el centro de un triángulo, un cuadrado, un hexágono y un círculo con el mismo perímetro. ¿Qué forma da el campo mínimo?
