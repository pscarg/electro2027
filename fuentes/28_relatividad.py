# %% [markdown]
# # Clase 28 — Relatividad especial: Lorentz, Minkowski y causalidad
#
# **Objetivos**
# - Dibujar un diagrama de Minkowski y ver cómo se inclinan los ejes de otro sistema al cambiar la rapidez: la relatividad de la simultaneidad.
# - Comprobar que las transformaciones de Lorentz conservan el intervalo y que las rapideces se suman.
# - Perseguir un pulso de luz con Galileo y con Lorentz, y sumar velocidades con muchos impulsos chicos.
# - Medir tiempos propios a lo largo de líneas de mundo (los mellizos) y ver de dónde sale la contracción de una fila de cargas.
#
# **Material relacionado:** notas de la Clase 28. Guía 12: problemas 1 y 2.
#
# **Unidades.** $c=1$: el tiempo se mide en las mismas unidades que la distancia ($ct$).

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
from scipy.integrate import trapezoid, cumulative_trapezoid

# %% [markdown]
# ## Experimento 1 ★ — El diagrama de Minkowski y la simultaneidad
#
# Un evento es un punto $(ct,x)$. Un sistema $S'$ que se mueve con velocidad $u=c\tanh\zeta$ respecto de $S$ (con $\zeta$ la **rapidez**) le asigna las coordenadas (sección 2 de las notas)
# $$\begin{pmatrix}ct'\\ x'\end{pmatrix}=\begin{pmatrix}\cosh\zeta&-\sinh\zeta\\ -\sinh\zeta&\cosh\zeta\end{pmatrix}\begin{pmatrix}ct\\ x\end{pmatrix} .$$
# En el diagrama de $S$, el eje $ct'$ es la línea $x'=0$, es decir $x=\beta\,ct$, la línea de mundo del origen de $S'$; el eje $x'$ es $t'=0$, es decir $ct=\beta x$. Los dos se inclinan hacia la diagonal de la luz. Las hipérbolas $c^2t^2-x^2=\pm1$ marcan la unidad en cada eje, porque el intervalo es el mismo en los dos sistemas.
#
# **El tren de Einstein.** Dos rayos caen en los extremos de un andén de largo 4, en $x=\pm2$, al mismo tiempo para el andén ($t=0$). Un tren pasa con velocidad $u$.
#
# ### Predecí
# Para un pasajero del tren que viaja hacia $+x$, ¿cuál de los dos rayos cae primero: el de adelante, el de atrás, o los dos a la vez?

# %%
def boost(zeta):
    """Transformación de Lorentz en (ct, x) a un sistema que se mueve con rapidez zeta (u = c tanh zeta)."""
    ch, sh = np.cosh(zeta), np.sinh(zeta)
    return np.array([[ch, -sh], [-sh, ch]])

def intervalo(ev):
    """c²t² − x² de un evento (ct, x) o de una lista de eventos (2 × N)."""
    return ev[0]**2 - ev[1]**2

def diagrama(ax, zeta, eventos=None, lim=4.0):
    L = boost(-zeta)                                    # de S' a S
    s = np.linspace(-2 * lim, 2 * lim, 2)
    ax.plot([0, 0], [-lim, lim], color="k", lw=1.5); ax.plot([-lim, lim], [0, 0], color="k", lw=1.5)
    ax.plot(s, s, "--", color=COLORES[3], lw=1.2); ax.plot(s, -s, "--", color=COLORES[3], lw=1.2)
    for k in range(-6, 7):                               # grilla de S': líneas de t' y de x' constantes
        for P, Q in ((L @ np.array([k, -10.0]), L @ np.array([k, 10.0])), (L @ np.array([-10.0, k]), L @ np.array([10.0, k]))):
            ax.plot([P[1], Q[1]], [P[0], Q[0]], color=COLORES[0], lw=0.5, alpha=0.35)
    for P, Q, nombre in ((L @ [-10, 0], L @ [10, 0], "ct'"), (L @ [0, -10], L @ [0, 10], "x'")):
        ax.plot([P[1], Q[1]], [P[0], Q[0]], color=COLORES[0], lw=2)
    h = np.linspace(-2.5, 2.5, 200)
    for sg in (1, -1):                                   # c²t² − x² = 1 (tipo tiempo) y = −1 (tipo espacio)
        ax.plot(np.sinh(h), sg * np.cosh(h), color="gray", lw=0.8, alpha=0.7)
        ax.plot(sg * np.cosh(h), np.sinh(h), color="gray", lw=0.8, alpha=0.7)
    if eventos is not None:
        for (ct, x), nombre, c in eventos:
            ctp, xp = boost(zeta) @ np.array([ct, x])
            ax.plot([x], [ct], "o", color=c, ms=8, zorder=5)
            P, Q = L @ np.array([ctp, -10.0]), L @ np.array([ctp, 10.0])        # la línea t' = cte que pasa por el evento
            ax.plot([P[1], Q[1]], [P[0], Q[0]], color=c, lw=1.2, ls=":")
            ax.text(x + 0.1, ct + 0.15, f"{nombre}: ct' = {ctp:+.2f}", color=c, fontsize=10)
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim), aspect="equal", xlabel="x", ylabel="ct")
    ax.grid(False)

RAYOS = [((0.0, 2.0), "adelante", COLORES[1]), ((0.0, -2.0), "atrás", COLORES[2])]

def mostrar_minkowski(zeta=0.5, exportar=False):
    fig, ax = plt.subplots(figsize=(6.6, 6.6))
    diagrama(ax, zeta, RAYOS)
    ax.set_title(f"S' con u = {np.tanh(zeta):.2f} c (rapidez ζ = {zeta:.2f}): ejes y grilla de S' en azul")
    if exportar:
        guardar(fig, "nb28_minkowski")
    plt.show()

interactuar(mostrar_minkowski, zeta=deslizador("rapidez ζ", 0.5, -1.5, 1.5, 0.05))
if CARPETA_FIGURAS:
    mostrar_minkowski(0.5, exportar=True)

# Comprobaciones
rng = np.random.default_rng(0)
ev = rng.normal(size=(2, 200)) * 3; zs = rng.uniform(-3, 3, 200)
err = max(abs(intervalo(boost(z) @ ev[:, i:i + 1])[0] - intervalo(ev[:, i:i + 1])[0]) / (abs(ev[:, i]).max()**2) for i, z in enumerate(zs))
verificar("el intervalo c²t² − x² no cambia (200 eventos y rapideces al azar)", err, 0.0, tol=1e-12)
verificar("componer dos boosts suma las rapideces: Λ(ζ₁)Λ(ζ₂) = Λ(ζ₁ + ζ₂)", np.abs(boost(0.7) @ boost(-1.9) - boost(0.7 - 1.9)).max(), 0.0, tol=1e-12)
eta = np.diag([1.0, -1, -1, -1])
def rot_z(t):
    R = np.eye(4); R[1:3, 1:3] = [[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]]; return R
B = np.eye(4); B[:2, :2] = boost(1.2)
Lam = rot_z(0.4) @ B @ rot_z(-1.1)
verificar("una rotación por un boost por otra rotación cumple Λᵀ η Λ = η", np.abs(Lam.T @ eta @ Lam - eta).max(), 0.0, tol=1e-12)
z = 0.5; g, b = np.cosh(z), np.tanh(z)
dt = (boost(z) @ np.array([0.0, 2.0]))[0] - (boost(z) @ np.array([0.0, -2.0]))[0]
verificar("tren: c(t'_adelante − t'_atrás) = −γβ Δx, con Δx = 4", dt, -g * b * 4.0, tol=1e-12)
b = 1e-4; x, ct = 3.0, 2.0
verificar("u ≪ c: Lorentz ≈ Galileo, x' ≈ x − ut (diferencia relativa de orden β²)", abs((boost(np.arctanh(b)) @ [ct, x])[1] - (x - b * ct)) / x, 0.0, tol=1e-7)

# %% [markdown]
# ### ¿Qué pasó?
# Para el andén, los dos rayos caen a la vez (los dos puntos están sobre $ct=0$). Pero las líneas de simultaneidad del tren, $t'=$ constante (las punteadas), están inclinadas: **el rayo de adelante cae antes** para los pasajeros, con $c\,\Delta t'=-\gamma\beta\,\Delta x$. Ninguno de los dos se equivoca: la simultaneidad de dos eventos separados depende del sistema. Cuando $\zeta$ crece, los ejes de $S'$ se cierran sobre la diagonal de la luz, que es la misma para todos. Las hipérbolas grises, $c^2t^2-x^2=\pm1$, cortan los ejes de $S'$ en su unidad: por eso la grilla de $S'$ se estira.

# %% [markdown]
# ## Experimento 2 — Perseguir la luz y sumar velocidades
#
# Un pulso de luz $f(x-ct)$ viaja hacia $+x$. Un observador lo persigue con velocidad $u$. Con Galileo ($x'=x-ut$, $t'=t$), el pico, que está en $x=ct$, queda en $x'=(c-u)t'$: el observador vería la luz más lenta, y si $u=c$, quieta. Con Lorentz transformamos los eventos de la línea de mundo del pico y medimos su velocidad en $S'$.
#
# Después, un cohete recibe muchos impulsos chicos: cada uno le suma $\Delta u=0.1\,c$ **en el sistema en el que el cohete está quieto en ese momento**. Con Galileo, después de $N$ impulsos va a $N\Delta u$. Con la suma relativista, $u\oplus\Delta u=\frac{u+\Delta u}{1+u\Delta u/c^2}$ (sección 2 de las notas), es decir, se suman las rapideces.
#
# ### Predecí
# 1. Con $u=0.9\,c$, ¿con qué velocidad ve pasar el pulso el observador? ¿Ve el pulso igual que en $S$?
# 2. ¿Cuántos impulsos de $0.1\,c$ hacen falta para llegar a $c$?

# %%
def mostrar_persecucion(beta=0.6):
    zeta = np.arctanh(beta)
    t = np.linspace(0, 6, 7)
    pico = np.vstack([t, t])                               # el pico del pulso: (ct, x) = (t, t)
    ctp, xp = boost(zeta) @ pico
    xs = np.linspace(-4, 12, 2000)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2))
    a1.plot(t, t, "o-", color=COLORES[3], label="en S: x = ct")
    a1.plot(t, (1 - beta) * t, "s--", color=COLORES[1], label=f"Galileo: x' = (c − u)t'")
    a1.plot(ctp, xp, "^-", color=COLORES[0], label="Lorentz: x' = ct'")
    a1.set(xlabel="ct (o ct')", ylabel="posición del pico", title=f"el pico del pulso visto desde S' (u = {beta:g} c)")
    a1.legend(fontsize=9)
    f = lambda u: np.exp(-u**2)                             # el pulso en S, en t = 0
    a2.plot(xs, f(xs), color=COLORES[3], label="en S")
    # en S', con t' = 0: x − ct = γ(1 − β)(x' − ct') (ver notas)
    a2.plot(xs, f(np.cosh(zeta) * (1 - beta) * xs), color=COLORES[0], label="en S' (Lorentz), t' = 0")
    a2.set(xlabel="x (o x')", title="el pulso: en S' es más ancho (efecto Doppler, Clase 30)", xlim=(-4, 8))
    a2.legend(fontsize=9)
    fig.tight_layout(); plt.show()

interactuar(mostrar_persecucion, beta=deslizador("u/c", 0.6, 0.0, 0.95, 0.05))

def mostrar_cohete(du=0.1):
    N = np.arange(0, 41)
    u = [0.0]
    for _ in N[1:]:
        u.append((u[-1] + du) / (1 + u[-1] * du))           # suma relativista de velocidades
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(N, du * N, "--", color=COLORES[1], label="Galileo: N Δu")
    ax.plot(N, u, "o", color=COLORES[0], ms=4, label="suma relativista, impulso por impulso")
    ax.plot(N, np.tanh(N * np.arctanh(du)), color=COLORES[0], lw=1, label="tanh(N ζ₁): las rapideces se suman")
    ax.axhline(1, color="k", lw=1)
    ax.set(xlabel="número de impulsos N", ylabel="u / c", ylim=(0, 2.2), title=f"un cohete que recibe impulsos de {du:g} c en su propio sistema")
    ax.legend(fontsize=9)
    plt.show()

interactuar(mostrar_cohete, du=deslizador("Δu/c", 0.1, 0.02, 0.5, 0.02))

# Comprobaciones
for beta in (0.6, 0.99):
    ctp, xp = boost(np.arctanh(beta)) @ np.vstack([np.linspace(0, 5, 6)] * 2)
    verificar(f"u = {beta} c: el pulso de luz viaja en S' con velocidad c (Lorentz)", np.polyfit(ctp, xp, 1)[0], 1.0, tol=1e-12)
u = [0.0]
for _ in range(25):
    u.append((u[-1] + 0.1) / (1 + u[-1] * 0.1))
verificar("25 impulsos de 0.1c: u = c tanh(25 ζ₁), con tanh ζ₁ = 0.1", u[-1], np.tanh(25 * np.arctanh(0.1)), tol=1e-12)
n, v = 1.33, 1e-4
exacta = (1 / n + v) / (1 + v / n)
verificar("Fizeau: la luz en agua que se mueve con v ≪ c va a c/n + v(1 − 1/n²)", (exacta - 1 / n) / v, 1 - 1 / n**2, tol=1e-3)

# %% [markdown]
# ### ¿Qué pasó?
# Con Galileo, el observador que persigue la luz la ve más lenta, y con $u=c$ vería una onda quieta: un campo que oscila en el espacio sin cambiar en el tiempo, que no es solución de las ecuaciones de Maxwell (sección 1 de las notas). Con Lorentz, **la luz va a $c$ en los dos sistemas**; lo que cambia es el pulso, que en $S'$ es más ancho: es el efecto Doppler, que veremos en la Clase 30.
#
# Los impulsos de $0.1\,c$ no llevan nunca el cohete a $c$: cada uno suma la misma rapidez $\zeta_1$, y $u=c\tanh(N\zeta_1)\to c$. Con Galileo, a los diez impulsos ya iría a $c$. Para $v\ll c$, la suma relativista da la corrección de primer orden que midió Fizeau en el agua en movimiento (nota histórica de las notas).

# %% [markdown]
# ## Experimento 3 — Relojes y reglas
#
# **El tiempo propio.** Un reloj que se mueve con velocidad $v(t)$ marca $d\tau=dt\sqrt{1-v^2/c^2}$ (sección 3 de las notas). Un viajero sale de la Tierra y vuelve en un tiempo $T$ (medido en la Tierra), con velocidad $v(t)=\beta_{\max}\,c\sin(2\pi t/T)$: va y vuelve. Integramos su tiempo propio.
#
# **Una fila de cargas.** Cargas quietas en $S'$, separadas $d_0$, tienen líneas de mundo verticales en $S'$. Las transformamos a $S$, donde se mueven con velocidad $u$, y medimos la separación entre sus posiciones **al mismo tiempo de $S$**.
#
# ### Predecí
# 1. ¿Quién envejece menos, el que viaja o el que se queda? ¿No es simétrico?
# 2. En $S$, ¿las cargas están más juntas o más separadas que en $S'$?

# %%
def tiempo_propio(beta_max=0.8, T=10.0, N=20001):
    t = np.linspace(0, T, N)
    v = beta_max * np.sin(2 * np.pi * t / T)
    x = cumulative_trapezoid(v, t, initial=0)
    tau = cumulative_trapezoid(np.sqrt(1 - v**2), t, initial=0)
    return t, x, tau

def fila_de_cargas(beta=0.6, d0=1.0, n=7):
    """Líneas de mundo de n cargas quietas en S' (x' = k d0), transformadas a S (que ve a S' moverse con +u)."""
    zeta = np.arctanh(beta)
    ctp = np.linspace(-8, 8, 401)
    lineas = [boost(-zeta) @ np.vstack([ctp, np.full_like(ctp, k * d0)]) for k in range(n)]
    x_en_t0 = np.array([np.interp(0.0, ct, x) for ct, x in lineas])       # posiciones en ct = 0 (de S)
    return lineas, x_en_t0

def mostrar_relojes(beta_max=0.8, beta=0.6, exportar=False):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5.6), gridspec_kw={"width_ratios": [1, 1.6]})
    t, x, tau = tiempo_propio(beta_max)
    a1.plot(x, t, color=COLORES[0], lw=2, label=f"viajero: τ = {tau[-1]:.2f}")
    a1.plot(0 * t, t, color=COLORES[1], lw=2, label=f"Tierra: τ = {t[-1]:.2f}")
    for k in range(0, len(t), 2000):
        a1.plot(x[k], t[k], "o", color=COLORES[0], ms=4)
        a1.text(x[k] + 0.08, t[k], f"{tau[k]:.1f}", fontsize=8, color=COLORES[0])
    a1.set(xlabel="x", ylabel="ct", aspect="equal", xlim=(-0.5, 4.5), ylim=(-0.3, 10.5), title="los mellizos: tiempo propio")
    a1.legend(fontsize=9, loc="lower right")
    lineas, x0 = fila_de_cargas(beta)
    for ct, xx in lineas:
        a2.plot(xx, ct, color=COLORES[2], lw=1.2)
    zeta = np.arctanh(beta); P, Q = boost(-zeta) @ [0, -3], boost(-zeta) @ [0, 9]
    a2.plot([P[1], Q[1]], [P[0], Q[0]], color=COLORES[0], lw=1.5, ls="--", label="t' = 0 (simultáneo en S')")
    a2.axhline(0, color=COLORES[1], lw=1.5, ls="--", label="t = 0 (simultáneo en S)")
    a2.plot(x0, 0 * x0, "o", color=COLORES[1], zorder=5)
    a2.set(xlim=(-2, 8), ylim=(-4, 6), aspect="equal", xlabel="x", ylabel="ct",
           title=f"cargas quietas en S' (separadas 1); en S: {x0[1] - x0[0]:.3f} = 1/γ")
    a2.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    if exportar:
        guardar(fig, "nb28_relojes")
    plt.show()

interactuar(mostrar_relojes, beta_max=deslizador("β máx del viajero", 0.8, 0.1, 0.99, 0.01), beta=deslizador("u/c de las cargas", 0.6, 0.0, 0.95, 0.05))
if CARPETA_FIGURAS:
    mostrar_relojes(0.8, 0.6, exportar=True)

# Comprobaciones
t = np.linspace(0, 10, 200001); b = 0.8
v = np.where(t < 5, b, -b)
verificar("viaje de ida y vuelta a 0.8c: τ_viajero / τ_Tierra = √(1 − β²) = 0.6", trapezoid(np.sqrt(1 - v**2), t) / 10, np.sqrt(1 - b**2), tol=1e-9)
from scipy.special import ellipe
t, x, tau = tiempo_propio(0.8)
verificar("el viajero vuelve al mismo lugar (∫v dt = 0)", abs(x[-1]), 0.0, tol=1e-9)
verificar("viaje con v = 0.8c sen(2πt/T): τ = (2T/π) E(β²), con E la integral elíptica completa", tau[-1], 2 * 10 / np.pi * ellipe(0.8**2), tol=1e-8)
for beta in (0.6, 0.95):
    _, x0 = fila_de_cargas(beta)
    verificar(f"u = {beta} c: la separación en S es d₀/γ (la densidad lineal crece en γ)", np.diff(x0).mean(), np.sqrt(1 - beta**2), tol=1e-9)

# %% [markdown]
# ### ¿Qué pasó?
# **El que viaja envejece menos.** No es simétrico: el viajero cambia de sistema inercial al dar la vuelta, y el de la Tierra no. Entre dos eventos, la línea de mundo recta (la del que no acelera) es la de **mayor** tiempo propio: en el espacio-tiempo, la recta es la más larga, por el signo menos del intervalo.
#
# En la fila de cargas, las líneas de mundo son paralelas e inclinadas. La separación en $S'$ se mide sobre $t'=0$, y la de $S$, sobre $t=0$: son cortes distintos de las mismas líneas, porque la simultaneidad cambia. En $S$ las cargas están **más juntas**, $d_0/\gamma$, y la densidad es $\lambda=\gamma\lambda_0$: es la contracción de Lorentz que usamos en la Clase 12 para ver la fuerza magnética como eléctrica.

# %% [markdown]
# ## Explorá
#
# 1. **Guía 12, problema 1.** Tomá un evento $(ct,x)$ tipo tiempo, uno tipo espacio y uno tipo luz, aplicales `boost(zeta)` para muchas rapideces $\zeta$ entre $-3$ y $3$, y graficá los puntos $(x',ct')$ que obtenés. ¿En qué casos cambia el signo de $t'$? Usá `diagrama` para ver los ejes de cada sistema.
# 2. **Guía 12, problema 2.** Deducí la dilatación del tiempo y la contracción de longitudes a partir de las transformaciones de Lorentz, y compará con `tiempo_propio` y `fila_de_cargas`. ¿Qué eventos tenés que comparar en cada caso?
# 3. **La garrocha y el granero.** Una garrocha de largo propio $1.5$ entra corriendo a $0.8\,c$ en un granero de largo $1$, con las dos puertas abiertas. Para el granero, la garrocha mide $0.9$ y cabe; para la garrocha, el granero mide $0.6$. Dibujá con `diagrama` las líneas de mundo de las puntas de la garrocha y de las puertas, y explicá qué eventos son simultáneos en cada sistema.
