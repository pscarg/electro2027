# %% [markdown]
# # Clase 14 — El dipolo magnético y el movimiento de cargas en campos
#
# **Objetivos**
# - Integrar el movimiento de una carga con el **método de Boris**: el ciclotrón, la deriva $\mathbf E\times\mathbf B$ y una **botella magnética** hecha con dos espiras.
# - Ver que, de lejos, una espira es un dipolo con $\mathbf m=IA\hat{\mathbf n}/c$.
# - Comprobar el término de contacto: el campo promedio dentro de una esfera que contiene una espira es $2\mathbf m/R^3$.
# - Calcular el torque y la fuerza sobre una espira en un campo externo, y compararlos con $\mathbf m\times\mathbf B$ y $\nabla(\mathbf m\cdot\mathbf B)$.
#
# **Material relacionado:** notas de la Clase 14. Guía 5: problemas 7, 9 y 10.
#
# **Unidades.** Gaussianas, adimensionales: $c=1$ en el código, $q/M=1$, longitudes en unidades del radio de la espira.

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

# %%
from scipy.integrate import trapezoid

# herramienta: boris v1 (NB14)
def boris(x0, v0, campos, q_M=1.0, c=1.0, dt=1e-2, pasos=1000):
    """Integra M dv/dt = q (E + v×B/c) con el método de Boris (no relativista).

    campos(x, t) devuelve (E, B) en la posición x (arreglos de 3 componentes).
    Las posiciones quedan en los tiempos n·dt y las velocidades en (n + ½)·dt.
    Con E = 0, el paso de Boris es una rotación exacta de v: conserva |v| a precisión de máquina.
    """
    x = np.array(x0, float); v = np.array(v0, float)
    X = np.empty((pasos + 1, 3)); V = np.empty((pasos, 3)); X[0] = x
    for n in range(pasos):
        E, B = campos(x, n * dt)
        v_menos = v + 0.5 * dt * q_M * np.asarray(E)
        t = 0.5 * dt * q_M * np.asarray(B) / c
        s = 2 * t / (1 + t @ t)
        v_prima = v_menos + np.cross(v_menos, t)
        v = v_menos + np.cross(v_prima, s) + 0.5 * dt * q_M * np.asarray(E)
        x = x + dt * v
        X[n + 1] = x; V[n] = v
    return X, V
# fin herramienta

# %% [markdown]
# ## Experimento 1 ★ — Cargas en campos
#
# ### (a) Ciclotrón
# Campo $\mathbf B=\hat{\mathbf z}$, carga con $q/M=1$ y velocidad inicial $(1,0,0.2)$. Las notas predicen una hélice con $\omega_c=qB/Mc=1$ y radio $r_L=v_\perp/\omega_c=1$.
#
# ### Predecí
# ¿Cambia la rapidez? ¿Cuánto tarda una vuelta?

# %%
uniforme = lambda x, t: (np.zeros(3), np.array([0, 0, 1.0]))
dt = 0.02
X, V = boris([0, 1, 0], [1, 0, 0.2], uniforme, dt=dt, pasos=3000)
rapidez = np.linalg.norm(V, axis=1)
cruces = np.where(np.diff(np.sign(X[:, 1])) > 0)[0]          # y pasa de − a + una vez por vuelta (centro en el origen)
periodo = np.mean(np.diff(cruces)) * dt
radio = 0.5 * (X[:, 0].max() - X[:, 0].min())

fig = plt.figure(figsize=(11, 4.4))
ax = fig.add_subplot(1, 2, 1, projection="3d")
ax.plot(X[:, 0], X[:, 1], X[:, 2], color=COLORES[0], lw=1)
ax.set(xlabel="x", ylabel="y", zlabel="z", title="ciclotrón: una hélice")
ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(np.arange(len(rapidez)) * dt, rapidez - rapidez[0], color=COLORES[1])
ax2.set(xlabel="t", ylabel="|v| − |v₀|", title="la rapidez no cambia")
plt.tight_layout(); plt.show()
verificar("|v| constante (máx. variación)", np.ptp(rapidez), 0.0, tol=1e-12)
verificar("período = 2π/ω_c", periodo, 2 * np.pi, tol=1e-3)
verificar("radio de Larmor = v⊥/ω_c", radio, 1.0, tol=1e-3)

# %% [markdown]
# ### (b) La deriva $\mathbf E\times\mathbf B$
# Ahora $\mathbf E=0.3\,\hat{\mathbf y}$ y $\mathbf B=\hat{\mathbf z}$. Las notas predicen que el centro de giro avanza con $\mathbf v_d=c\,\mathbf E\times\mathbf B/B^2=0.3\,\hat{\mathbf x}$, para cualquier carga. Probamos tres puntos de partida y una carga negativa.
#
# ### Predecí
# ¿Hacia dónde avanza una carga positiva que parte del reposo? ¿Y una negativa?

# %%
cruzados = lambda x, t: (np.array([0, 0.3, 0]), np.array([0, 0, 1.0]))
casos = [("q > 0, desde el reposo", 1.0, [0, 0, 0]), ("q > 0, v₀ = v_d", 1.0, [0.3, 0, 0]),
         ("q > 0, v₀ = (0, 0.6, 0)", 1.0, [0, 0.6, 0]), ("q < 0, desde el reposo", -1.0, [0, 0, 0])]
fig, ax = plt.subplots(figsize=(9, 4))
derivas = {}
for (nombre, qm, v0), c_ in zip(casos, COLORES):
    X, V = boris([0, 0, 0], v0, cruzados, q_M=qm, dt=0.01, pasos=6283 * 2)
    ax.plot(X[:1500, 0], X[:1500, 1], color=c_, lw=1.2, label=nombre)          # se dibujan los primeros 15 instantes
    T = 2 * np.pi; n_per = int(round(T / 0.01))
    derivas[nombre] = (X[-1, 0] - X[-1 - 10 * n_per, 0]) / (10 * n_per * 0.01)
ax.set(xlabel="x", ylabel="y", aspect="equal", title="E según +y, B según +z"); ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5))
plt.show()
for nombre, vd in derivas.items():
    print(f"{nombre:26s}: velocidad media en x = {vd:.5f}")
verificar("deriva de una carga positiva = cE×B/B²", derivas["q > 0, desde el reposo"], 0.3, tol=1e-3)
verificar("deriva de una carga negativa: la misma", derivas["q < 0, desde el reposo"], 0.3, tol=1e-3)

# %% [markdown]
# ### (c) Una botella magnética
# Dos espiras coaxiales de radio 1 en $z=\pm2$, con la misma corriente, producen un campo débil en el medio y fuerte cerca de las espiras (calculado con la herramienta de Biot–Savart de la Clase 12). Soltamos una carga cerca del eje, en $z=0$, con rapidez 1 y un ángulo $\alpha_0$ con el campo. Según las notas, $\mu=Mv_\perp^2/2B$ se conserva y la carga rebota donde $B=B_0/\sin^2\alpha_0$, si ese valor es menor que el máximo del campo.
#
# ### Predecí
# Con $\alpha_0=30^\circ$ y con $\alpha_0=20^\circ$, ¿queda atrapada? El campo máximo (en las espiras) es unas 5.7 veces el del centro.

# %%
I_bot = 8.0
bobinas = [espira(1.0, 60, -2.0), espira(1.0, 60, 2.0)]
def campo_botella(x, t):
    return np.zeros(3), sum(campo_poligonal([x], v, I_c=I_bot)[0] for v in bobinas)
B0 = np.linalg.norm(campo_botella(np.array([0.05, 0, 0]), 0)[1])
Bmax = np.linalg.norm(campo_botella(np.array([0, 0, 2.0]), 0)[1])
print(f"B en el centro = {B0:.3f},  B en las espiras (eje) = {Bmax:.3f},  cociente = {Bmax / B0:.2f}")

fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
resultado = {}
for alfa, c_ in [(30, COLORES[0]), (20, COLORES[1])]:
    a = np.radians(alfa)
    X, V = boris([0.05, 0, 0], [np.sin(a), 0, np.cos(a)], campo_botella, dt=0.005, pasos=5000)
    Bs = np.array([np.linalg.norm(campo_botella(x, 0)[1]) for x in X[:-1]])
    bvec = np.array([campo_botella(x, 0)[1] for x in X[:-1]])
    vpar = np.sum(V * bvec, axis=1) / Bs
    mu = (np.sum(V**2, axis=1) - vpar**2) / (2 * Bs)
    axs[0].plot(np.arange(len(X)) * 0.005, X[:, 2], color=c_, label=f"α₀ = {alfa}°")
    axs[1].plot(np.arange(len(mu)) * 0.005, mu / mu[0], color=c_, label=f"α₀ = {alfa}°")
    giro = np.where(np.diff(np.sign(vpar)) != 0)[0]
    resultado[alfa] = (Bs[giro[0]] / B0 if len(giro) else None, np.ptp(mu) / mu[0])
axs[0].axhline(2, color="0.6", ls=":"); axs[0].axhline(-2, color="0.6", ls=":")
axs[0].set(xlabel="t", ylabel="z", title="posición a lo largo del eje"); axs[0].legend()
axs[1].set(xlabel="t", ylabel="μ / μ₀", ylim=(0.8, 1.2), title="el momento magnético de la órbita"); axs[1].legend()
plt.show()
for alfa, (br, dmu) in resultado.items():
    print(f"α₀ = {alfa}°: " + (f"rebota con B/B₀ = {br:.3f} (predicción 1/sin²α₀ = {1 / np.sin(np.radians(alfa))**2:.3f})" if br else "no rebota: escapa")
          + f";  variación relativa de μ = {dmu:.3f}")
verificar("rebote en B = B₀/sin²α₀ (α₀ = 30°)", resultado[30][0], 1 / np.sin(np.radians(30))**2, tol=3e-2)
verificar("con α₀ = 20° escapa (1/sin²α₀ > B_max/B₀)", float(resultado[20][0] is None), 1.0, tol=1e-12)

# %% [markdown]
# ### ¿Qué pasó?
# En un campo uniforme la carga describe una hélice con $\omega_c=qB/Mc$ y su rapidez no cambia: el campo magnético no hace trabajo (y el método de Boris lo respeta exactamente). Con $\mathbf E$ y $\mathbf B$ cruzados, el centro de giro avanza con $c\mathbf E\times\mathbf B/B^2$, **perpendicular a $\mathbf E$**, sea cual sea la carga o el punto de partida: la carga negativa gira al revés, pero deriva igual. En la botella, $\mu$ se conserva bastante bien (el campo varía lento comparado con el giro), y la carga rebota donde predice la conservación de $\mu$; con un ángulo más chico, el campo máximo no alcanza y escapa por el "cono de pérdida".
#
# ## Experimento 2 — Una espira vista de lejos
#
# Comparamos el campo y el potencial de una espira cuadrada de lado 1 con los de un dipolo con $\mathbf m=IA\hat{\mathbf n}/c=\hat{\mathbf z}$, en direcciones al azar y a distancias crecientes.
#
# ### Predecí
# ¿Cómo decae el error relativo de la aproximación dipolar con la distancia?

# %%
cuadrada = [[0.5, 0.5, 0], [-0.5, 0.5, 0], [-0.5, -0.5, 0], [0.5, -0.5, 0]]
m = np.array([0, 0, 1.0])
rng = np.random.default_rng(3)
dirs = rng.normal(size=(50, 3)); dirs /= np.linalg.norm(dirs, axis=1)[:, None]
def dipolo_B(P):
    r = np.linalg.norm(P, axis=1)[:, None]; n = P / r
    return (3 * (n @ m)[:, None] * n - m) / r**3
def dipolo_A(P):
    r = np.linalg.norm(P, axis=1)[:, None]
    return np.cross(m, P) / r**3
rs = np.geomspace(2, 50, 12)
errB, errA = [], []
for r in rs:
    P = r * dirs
    errB.append(np.max(np.linalg.norm(campo_poligonal(P, cuadrada) - dipolo_B(P), axis=1) / np.linalg.norm(dipolo_B(P), axis=1)))
    errA.append(np.max(np.linalg.norm(potencial_poligonal(P, cuadrada) - dipolo_A(P), axis=1) / np.linalg.norm(dipolo_A(P), axis=1)))
fig, ax = plt.subplots(figsize=(6.6, 4.3))
ax.loglog(rs, errB, "o-", color=COLORES[0], label="B"); ax.loglog(rs, errA, "s-", color=COLORES[1], label="A")
ax.set(xlabel="distancia r (lado = 1)", ylabel="error relativo máximo", title="espira cuadrada frente a dipolo"); ax.legend()
ax.set_xticks([2, 5, 10, 20, 50], labels=["2", "5", "10", "20", "50"]); ax.set_xticks([], minor=True)
plt.show()
verificar("pendiente del error de B frente a dipolo = −2", np.polyfit(np.log(rs), np.log(errB), 1)[0], -2.0, tol=3e-2)

# %% [markdown]
# ### ¿Qué pasó?
# Lejos, la espira es un dipolo con $m=IA/c$: el error decae como $(a/r)^2$. (No hay término de orden $a/r$: el cuadrupolo magnético de una espira plana y simétrica es cero, y la primera corrección es la siguiente.)
#
# ## Experimento 3 — El campo promedio dentro de una esfera
#
# Una espira circular chica (radio $0.3$) dentro de una esfera de radio $R=1$. El promedio de $\mathbf B$ sobre la bola es $\frac1V\int\nabla\times\mathbf A\,d^3r=\frac1V\oint\hat{\mathbf n}\times\mathbf A\,da$: lo calculamos con el potencial sobre la superficie, que es suave.
#
# ### Predecí
# ¿Cuánto vale el promedio? Compará con el del dipolo eléctrico, $-\mathbf p/R^3$.

# %%
def direcciones(M=400):
    """M direcciones casi uniformes (espiral de Fibonacci)."""
    i = np.arange(M) + 0.5
    th = np.arccos(1 - 2 * i / M); ph = np.pi * (1 + 5**0.5) * i
    return np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], axis=1)

n_s = direcciones(20000)
for centro in ([0, 0, 0], [0.2, -0.3, 0.1]):
    esp = espira(0.3, 360) + np.array(centro)
    m_esp = np.pi * 0.3**2                      # I/c = 1
    A_s = potencial_poligonal(n_s, esp)
    promedio = np.mean(np.cross(n_s, A_s), axis=0) * 4 * np.pi / (4 * np.pi / 3)   # (1/V) ∮ n×A da, con R = 1
    verificar(f"⟨B_z⟩ en la bola (espira con centro en {centro}) = 2m/R³", promedio[2], 2 * m_esp, tol=1e-3)
u = np.linspace(-1, 1, 2001)                                    # u = cos θ
verificar("promedio angular de la parte 1/r³ del dipolo: ⟨3cos²θ − 1⟩", trapezoid(3 * u**2 - 1, u) / 2, 0.0, tol=1e-6)

# %% [markdown]
# ### ¿Qué pasó?
# El promedio es $2\mathbf m/R^3$, esté donde esté la espira dentro de la esfera, mientras que la fórmula $1/r^3$ del dipolo promedia cero. Lo que falta es el término de contacto, $+\frac{8\pi}{3}\mathbf m\,\delta^3$: positivo y el doble que el eléctrico. Adentro de una espira, el campo apunta según $\mathbf m$.
#
# ## Experimento 4 — Torque y fuerza
#
# Una espira rectangular de $1\times2$ en un campo uniforme inclinado: calculamos el torque $\sum\mathbf r\times\frac Ic(d\mathbf l\times\mathbf B)$. Después, una espira chica (radio 0.1) sobre el eje de una grande (radio 1), y la fuerza $\frac Ic\oint d\mathbf l\times\mathbf B$ sobre la chica.
#
# ### Predecí
# ¿Hacia dónde tira la espira grande a la chica si sus corrientes giran en el mismo sentido?

# %%
def fuerza_y_torque(vertices, B_en, I_c=1.0, n=400):
    V = np.vstack([vertices, vertices[:1]]); F = np.zeros(3); tau = np.zeros(3)
    for P, Q in zip(V[:-1], V[1:]):
        u = (np.arange(n) + 0.5) / n; r = P + np.outer(u, Q - P); dl = (Q - P) / n
        dF = I_c * np.cross(np.broadcast_to(dl, r.shape), B_en(r))
        F += dF.sum(axis=0); tau += np.cross(r, dF).sum(axis=0)
    return F, tau

rect = np.array([[0.5, 1, 0], [-0.5, 1, 0], [-0.5, -1, 0], [0.5, -1, 0]])
B_u = np.array([0.4, -0.2, 0.9])
F, tau = fuerza_y_torque(rect, lambda r: np.tile(B_u, (len(r), 1)))
m_rect = np.array([0, 0, 2.0])
verificar("torque en campo uniforme = m × B", np.linalg.norm(tau - np.cross(m_rect, B_u)), 0.0, tol=1e-10)
verificar("fuerza en campo uniforme = 0", np.linalg.norm(F), 0.0, tol=1e-10)

grande = espira(1.0, 720)
B_grande = lambda r: campo_poligonal(r, grande)
m1 = np.pi
for z0 in (1.5, 10.0, 30.0):
    chica = espira(0.1, 120, z0)
    F, _ = fuerza_y_torque(chica, B_grande, n=20)
    m2 = np.pi * 0.1**2
    h = 1e-4
    dBz = (B_grande([[0, 0, z0 + h]])[0, 2] - B_grande([[0, 0, z0 - h]])[0, 2]) / (2 * h)
    verificar(f"F_z sobre la espira chica en z = {z0} = m ∂B_z/∂z", F[2], m2 * dBz, tol=1e-2)
    if z0 > 5:
        print(f"z = {z0}: F_z / (−6 m₁m₂/z⁴) = {F[2] / (-6 * m1 * m2 / z0**4):.4f}")
verificar("lejos (z = 30): F_z = −6 m₁m₂/z⁴ (dos dipolos coaxiales)", F[2], -6 * m1 * m2 / 30**4, tol=1e-2)

# %% [markdown]
# ### ¿Qué pasó?
# En un campo uniforme la espira no siente fuerza, pero sí el torque $\mathbf m\times\mathbf B$, que tiende a alinear $\mathbf m$ con $\mathbf B$. En el campo no uniforme de otra espira, la fuerza es $\nabla(\mathbf m\cdot\mathbf B)$: con corrientes en el mismo sentido, la chica es atraída hacia la grande (donde el campo es más intenso), y lejos es la fuerza entre dos dipolos coaxiales, $-6m_1m_2/z^4$ (a $z=10$ todavía difiere un 2.5 %, la corrección de orden $R^2/z^2$ por el tamaño de la espira grande).
#
# ## Explorá
# 1. **Guía 5, P7.** Armá el disco que gira como espiras concéntricas con $dI=\sigma\omega s\,ds$, calculá su momento magnético y el campo en el eje, y compará con tus resultados (incluido el límite lejano).
# 2. **Guía 5, P9.** Calculá con `fuerza_y_torque` la fuerza sobre una espira cuadrada junto a un hilo largo, y compará con $\nabla(\mathbf m\cdot\mathbf B)$ cuando el lado es mucho menor que la distancia.
# 3. **Guía 5, P10.** Compará la trayectoria de Boris desde el reposo en campos cruzados con la solución analítica que encontraste. ¿Qué curva es? ¿Qué pasa si $E>B$?
# 4. **El cono de pérdida.** Barré el ángulo inicial en la botella y encontrá el ángulo crítico por debajo del cual la carga escapa. Compará con $\sin^2\alpha_c=B_0/B_{\max}$.
