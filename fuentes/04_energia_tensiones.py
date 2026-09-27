# %% [markdown]
# # Clase 4 — Energía electrostática y el tensor de tensiones de Maxwell
#
# **Objetivos**
# - Calcular la fuerza sobre una carga **integrando el tensor de tensiones** sobre distintas superficies, y ver que todas dan el mismo resultado.
# - Ver que el campo "tira" a lo largo de las líneas y "empuja" perpendicularmente a ellas.
# - Comprobar que la energía de interacción está **en el campo**: $\frac{1}{4\pi}\int \mathbf E_1\cdot\mathbf E_2\,d^3r = q_1q_2/d$.
# - Entender la diferencia entre la energía de un conjunto de cargas puntuales (solo interacción) y la de una distribución continua (que incluye autoenergía).
#
# **Material relacionado:** notas de la Clase 4. Guía 2: problemas 5 y 6.

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

from scipy.integrate import dblquad

# %% [markdown]
# ## Experimento 1 ★ — La fuerza entre dos cargas, leída en la superficie
#
# Dos cargas: $q_1$ en $z=+a$ y $q_2$ en $z=-a$. La fuerza sobre $q_1$ es
# $$F_j = \oint_S T_{ij}\,n_i\,da, \qquad T_{ij}=\frac{1}{4\pi}\Big(E_iE_j-\tfrac12\delta_{ij}E^2\Big),$$
# para **cualquier** superficie cerrada $S$ que encierre a $q_1$ y no a $q_2$. Aquí $\mathbf E$ es el campo **total**, incluido el de la propia $q_1$. Probamos tres superficies:
# 1. el plano $z=0$, cerrado por una semiesfera en el infinito;
# 2. una esfera de radio $b$ centrada en $q_1$;
# 3. un cilindro que rodea a $q_1$.
#
# ### Predecí
# 1. Con $q_1=+1$ y $q_2=-1$, sobre el plano $z=0$: ¿el campo es paralelo o perpendicular al plano? ¿El "vacío" del otro lado **tira** de la carga o la **empuja**?
# 2. ¿Y con dos cargas iguales?
# 3. ¿Contribuye el campo propio de $q_1$ a la integral sobre la esfera?

# %%
def campo(x, y, z, q1, q2, a):
    """Campo total de q1 en (0,0,a) y q2 en (0,0,-a)."""
    E = np.zeros((3,) + np.shape(x))
    for q, zq in ((q1, a), (q2, -a)):
        R = np.array([x, y, z - zq])
        E += q * R / np.sum(R**2, axis=0)**1.5
    return E

def traccion(E, n):
    """Fuerza por unidad de área que transmite el campo: t_j = T_ij n_i."""
    E2 = np.sum(E**2, axis=0)
    En = np.sum(E * n, axis=0)
    return (E * En - 0.5 * E2 * n) / (4 * np.pi)

def fuerza_plano(q1, q2, a, n=4000):
    """Plano z = 0 (normal hacia abajo, saliendo de la región que contiene a q1)."""
    t, w = np.polynomial.legendre.leggauss(n)
    u = 0.5 * (t + 1); s = u / (1 - u) * a; ds = a / (1 - u)**2 * 0.5 * w     # s en [0, ∞)
    E = campo(s, 0 * s, 0 * s, q1, q2, a)
    n_vec = np.array([0 * s, 0 * s, -np.ones_like(s)])
    return np.sum(traccion(E, n_vec)[2] * 2 * np.pi * s * ds)

def fuerza_esfera(q1, q2, a, b, n=400):
    """Esfera de radio b centrada en q1 (normal hacia afuera)."""
    c, w = np.polynomial.legendre.leggauss(n)          # c = cos θ
    st = np.sqrt(1 - c**2)
    n_vec = np.array([st, 0 * c, c])
    E = campo(b * st, 0 * c, a + b * c, q1, q2, a)
    return np.sum(traccion(E, n_vec)[2] * 2 * np.pi * b**2 * w)

def fuerza_cilindro(q1, q2, a, rc, z1, z2, n=2000):
    """Cilindro de radio rc entre z1 y z2 (tapas + lateral)."""
    t, w = np.polynomial.legendre.leggauss(n)
    s = 0.5 * rc * (t + 1); ds = 0.5 * rc * w
    F = 0.0
    for zt, nz in ((z2, 1.0), (z1, -1.0)):
        E = campo(s, 0 * s, zt + 0 * s, q1, q2, a)
        F += np.sum(traccion(E, np.array([0 * s, 0 * s, nz + 0 * s]))[2] * 2 * np.pi * s * ds)
    z = 0.5 * (z2 - z1) * (t + 1) + z1; dz = 0.5 * (z2 - z1) * w
    E = campo(rc + 0 * z, 0 * z, z, q1, q2, a)
    F += np.sum(traccion(E, np.array([1 + 0 * z, 0 * z, 0 * z]))[2] * 2 * np.pi * rc * dz)
    return F

q1, q2, a = 1.0, -1.0, 0.5
exacto = q1 * q2 / (2 * a)**2            # componente z de la fuerza sobre q1 (negativa: atracción)
print(f"Coulomb: F_z = {exacto:+.6f}")
verificar("plano z = 0", fuerza_plano(q1, q2, a), exacto, tol=1e-6)
verificar("esfera de radio 0.3 alrededor de q1", fuerza_esfera(q1, q2, a, 0.3), exacto, tol=1e-6)
verificar("esfera de radio 0.9 alrededor de q1", fuerza_esfera(q1, q2, a, 0.9), exacto, tol=1e-6)
verificar("cilindro r = 2, 0.2 < z < 3", fuerza_cilindro(q1, q2, a, 2.0, 0.2, 3.0), exacto, tol=1e-6)
verificar("dos cargas iguales, plano z = 0", fuerza_plano(1.0, 1.0, a), +1.0 / (2 * a)**2, tol=1e-6)

# %% [markdown]
# Miremos la **tracción** $\mathbf t = T\cdot\hat{\mathbf n}$ sobre una esfera alrededor de $q_1$: la fuerza por unidad de área que el campo de afuera ejerce, a través de esa superficie, sobre lo que está adentro.

# %%
def dibujar_traccion(q2=-1.0, b=0.45):
    th = np.linspace(0, np.pi, 13)[1:-1]
    fig, ax = plt.subplots(figsize=(6, 6))
    x = np.linspace(-1.5, 1.5, 300); z = np.linspace(-1.5, 1.5, 300)
    X, Z = np.meshgrid(x, z)
    Psi = 1.0 * (Z - a) / np.hypot(X, Z - a) + q2 * (Z + a) / np.hypot(X, Z + a)
    nq = 1 + abs(q2)
    ax.contour(X, Z, Psi, levels=np.linspace(-nq, nq, 25), colors="0.75", linewidths=0.7, linestyles="solid")
    for signo in (1, -1):
        px, pz = signo * b * np.sin(th), a + b * np.cos(th)
        n_vec = np.array([signo * np.sin(th), 0 * th, np.cos(th)])
        t = traccion(campo(px, 0 * px, pz, 1.0, q2, a), n_vec)
        ax.quiver(px, pz, t[0], t[2], color=COLORES[1], scale=4, width=0.006)
    circ = np.linspace(0, 2 * np.pi, 200)
    ax.plot(b * np.cos(circ), a + b * np.sin(circ), "k", lw=1)
    ax.plot(0, a, "o", color=COLORES[7], ms=10); ax.plot(0, -a, "o", color=COLORES[7] if q2 > 0 else COLORES[0], ms=10)
    ax.set(aspect="equal", xlim=(-1.5, 1.5), ylim=(-1.5, 1.5),
           title=f"tracción sobre una esfera alrededor de q1 (q2 = {q2:+.1f})\nF_z = {fuerza_esfera(1.0, q2, a, b):+.4f}")
    ax.grid(False); plt.show()

interactuar(dibujar_traccion, q2=deslizador("q2", -1.0, -2, 2, 0.25), b=deslizador("radio b", 0.45, 0.15, 0.9, 0.05))

# %% [markdown]
# ### ¿Qué pasó?
# - Las cuatro superficies dan **exactamente** la fuerza de Coulomb, $q_1q_2/(2a)^2$. La fuerza sobre las cargas encerradas se puede leer en *cualquier* superficie que las rodee: el campo transmite la fuerza como un medio elástico.
# - El campo **tira** a lo largo de sus líneas (tensión $E^2/8\pi$) y **empuja** perpendicularmente a ellas (presión $E^2/8\pi$). Con cargas opuestas, las líneas cruzan el plano medio y tiran de las cargas una hacia la otra. Con cargas iguales, las líneas son paralelas al plano y el campo las empuja hacia afuera.
# - En la esfera, el campo propio de $q_1$ produce tracciones grandes pero simétricas, que **no dan fuerza neta**: no hay autofuerza.
#
# ## Experimento 2 — La energía de interacción está en el campo
#
# Con $\mathbf E = \mathbf E_1+\mathbf E_2$, la energía del campo es $\frac{1}{8\pi}\int E_1^2 + \frac{1}{8\pi}\int E_2^2 + \frac{1}{4\pi}\int\mathbf E_1\cdot\mathbf E_2$. Los dos primeros términos son las autoenergías (infinitas para cargas puntuales). El tercero debería ser la energía de interacción $q_1q_2/d$.
#
# ### Predecí
# Para $q_1=+1$ y $q_2=-1$: ¿dónde es negativa la densidad $\mathbf E_1\cdot\mathbf E_2/4\pi$? ¿Dónde es positiva? ¿Cuál gana? (Pista: mirá hacia dónde apunta cada campo entre las cargas y afuera.)

# %%
def densidad_interaccion(s, z, q1, q2, a):
    R1 = np.sqrt(s**2 + (z - a)**2); R2 = np.sqrt(s**2 + (z + a)**2)
    return q1 * q2 * (s**2 + (z - a) * (z + a)) / (R1**3 * R2**3) / (4 * np.pi)

def energia_interaccion(q1, q2, a):
    """(1/4π) ∫ E1·E2 d³r en cilíndricas, partiendo la región en trozos para la cuadratura."""
    total = 0.0
    for s0, s1 in ((0, 1), (1, np.inf)):
        for z0, z1 in ((-np.inf, -a), (-a, 0), (0, a), (a, np.inf)):
            v, _ = dblquad(lambda z, s: 2 * np.pi * s * densidad_interaccion(s, z, q1, q2, a),
                           s0, s1, z0, z1, epsabs=1e-10, epsrel=1e-8)
            total += v
    return total

verificar("(1/4π)∫E1·E2 d³r = q1q2/d con cargas opuestas", energia_interaccion(1.0, -1.0, 0.5), -1.0, tol=1e-6)
verificar("(1/4π)∫E1·E2 d³r = q1q2/d con cargas iguales", energia_interaccion(1.0, 1.0, 0.5), 1.0, tol=1e-6)

x = np.linspace(-1.5, 1.5, 300); z = np.linspace(-1.5, 1.5, 300)   # grilla que no pasa exactamente por las cargas
X, Z = np.meshgrid(x, z)
fig, axs = plt.subplots(1, 2, figsize=(11, 5))
for ax, q2_ in zip(axs, (-1.0, 1.0)):
    u = densidad_interaccion(np.abs(X), Z, 1.0, q2_, 0.5)
    lim = np.percentile(np.abs(u), 95)
    im = ax.pcolormesh(X, Z, u, cmap="RdBu_r", vmin=-lim, vmax=lim, shading="auto")
    ax.plot(0, 0.5, "o", color=COLORES[7], ms=8); ax.plot(0, -0.5, "o", color=COLORES[7] if q2_ > 0 else COLORES[0], ms=8)
    ax.set(aspect="equal", title=f"E1·E2/4π   (q1 = +1, q2 = {q2_:+.0f})"); ax.grid(False)
fig.colorbar(im, ax=axs, shrink=0.8, label="rojo: positiva, azul: negativa")
guardar(fig, "nb04_interaccion"); plt.show()

# %% [markdown]
# ### ¿Qué pasó?
# La integral de $\mathbf E_1\cdot\mathbf E_2/4\pi$ sobre todo el espacio reproduce **exactamente** $q_1q_2/d$.
#
# Para cargas **opuestas**, entre las cargas los dos campos apuntan en el mismo sentido (de $+$ a $-$), así que la densidad es **positiva**. Afuera apuntan en sentidos opuestos y la densidad es **negativa**. La región negativa es mucho más grande y gana: $U_{\text{int}}=-1/d$. Para cargas **iguales** ocurre lo contrario. En ambos casos, el límite entre las dos regiones es la esfera que tiene a las cargas como extremos de un diámetro: allí $\mathbf E_1\perp\mathbf E_2$.
#
# La energía de interacción se puede pensar como localizada en el campo, con densidad $\mathbf E_1\cdot\mathbf E_2/4\pi$.
#
# ## Experimento 3 — Cargas puntuales frente a distribución continua
#
# Repartimos una carga $Q=1$ en $N$ cargas iguales sobre una esfera de radio $a=1$ (puntos distribuidos de manera casi uniforme). Su energía es la suma sobre pares, $U_N=\sum_{i<j} q_iq_j/r_{ij}$, que **no incluye** la autoenergía de cada carga. La cáscara continua tiene $U = Q^2/2a$.
#
# ### Predecí
# ¿$U_N$ se acerca a $Q^2/2a$ desde arriba o desde abajo? ¿Qué le falta a la suma sobre pares?

# %%
def puntos_esfera(N):
    """Puntos casi uniformes sobre la esfera unitaria (espiral de Fibonacci)."""
    i = np.arange(N) + 0.5
    th = np.arccos(1 - 2 * i / N); ph = np.pi * (1 + 5**0.5) * i
    return np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], axis=1)

def energia_pares(N, Q=1.0, a=1.0):
    P = a * puntos_esfera(N); suma = 0.0
    for k in range(0, N, 500):                          # por bloques, para no usar mucha memoria
        D = np.linalg.norm(P[k:k + 500, None, :] - P[None, :, :], axis=2)
        mascara = np.arange(k, min(k + 500, N))[:, None] < np.arange(N)[None, :]
        suma += np.sum(1 / D[mascara])
    return (Q / N)**2 * suma

Ns = np.array([100, 250, 500, 1000, 2000, 4000])
Us = np.array([energia_pares(n) for n in Ns])
A_, B_ = np.linalg.lstsq(np.vstack([np.ones(len(Ns)), 1 / np.sqrt(Ns)]).T, Us, rcond=None)[0]

fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.plot(1 / np.sqrt(Ns), Us, "o", label="suma sobre pares $U_N$")
xx = np.linspace(0, 0.11, 50); ax.plot(xx, A_ + B_ * xx, "--", lw=1, label=f"ajuste {A_:.4f} {B_:+.3f}/√N")
ax.axhline(0.5, color="0.5", lw=1); ax.text(0.06, 0.4965, "Q²/2a (cáscara continua)", color="0.4")
ax.set(xlabel="1/√N", ylabel="energía", title="N cargas sobre una esfera"); ax.legend(loc="lower left")
guardar(fig, "nb04_esfera_N"); plt.show()
verificar("U_N extrapolada a N → ∞ = Q²/2a", A_, 0.5, tol=1e-3)

# %% [markdown]
# ### ¿Qué pasó?
# $U_N$ se acerca a $Q^2/2a$ **desde abajo**, con una diferencia que decae como $1/\sqrt N$. Lo que le falta a la suma sobre pares es la **autoenergía de cada trozo**. Si pensamos cada carga $Q/N$ como una manchita de tamaño comparable a la separación, $\sim a/\sqrt N$, su autoenergía es $\sim (Q/N)^2/(a/\sqrt N)$. Sumadas las $N$ manchitas dan $\sim Q^2/(a\sqrt N)$, que es exactamente el término que falta y que tiende a cero.
#
# La integral continua $\frac12\int\rho\phi$ incluye las autoenergías; la suma sobre pares, no. Para cargas puntuales la diferencia es infinita, porque la autoenergía de un punto diverge.
#
# ## Explorá
# 1. **Guía 2, P6.** Deducí analíticamente la fuerza entre dos cargas **iguales** integrando $T_{ij}$ sobre el plano medio. ¿Por qué ahí el campo es tangente al plano? Compará con `fuerza_plano(1, 1, a)`.
# 2. **Otras superficies.** Probá `fuerza_esfera` con un radio mayor que $2a$, que encierra a las dos cargas. ¿Qué obtenés y por qué?
# 3. **Guía 2, P5.** Calculá numéricamente $\frac{1}{8\pi}\int E^2\,d^3r$ para una esfera maciza uniformemente cargada y compará con el resultado de construirla capa por capa.
# 4. **El torque.** Con el tensor de tensiones, calculá el torque $\oint \mathbf r\times(T\cdot\hat{\mathbf n})\,da$ sobre un dipolo en un campo uniforme y compará con $\mathbf p\times\mathbf E$ (Clase 6).
