# %% [markdown]
# # Clase 18 — Las ecuaciones de Maxwell: corriente de desplazamiento, ondas y potenciales
#
# **Objetivos**
# - Integrar las ecuaciones de Maxwell en una dimensión, paso a paso, y ver aparecer las ondas: un pulso que se parte en dos y una lámina de corriente que emite.
# - Comprobar que, con la corriente de desplazamiento, la circulación de $\mathbf B$ no depende de la superficie elegida.
# - Ver cuándo deja de valer la aproximación cuasiestática en un capacitor.
# - Hacer un cambio de calibre numérico: potenciales distintos, los mismos campos.
#
# **Material relacionado:** notas de la Clase 18. Guía 7: problemas 2 y 3.
#
# **Unidades.** Gaussianas, adimensionales, con $c=1$.

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
# herramienta: fdtd1d v1 (NB18)
def yee1d(E, B, pasos, S=1.0, dx=1.0, J=None, cada=1, absorbente=True):
    """Ecuaciones de Maxwell en 1D con el esquema de Yee, en unidades con c = 1.

    E: E_y en los nodos x_i = i dx, en t = 0 (N valores).
    B: B_z en los puntos medios x_{i+1/2}, en t = −dt/2 (N − 1 valores).
    dt = S dx, con S ≤ 1 (número de Courant). J(n): J_y en los nodos en t = (n + 1/2) dt (opcional).
    Actualiza ∂B_z/∂t = −∂E_y/∂x y ∂E_y/∂t = −∂B_z/∂x − 4π J_y, con bordes absorbentes de Mur (exactos si S = 1).
    Devuelve las historias (Es, Bs, ts), guardadas cada `cada` pasos (t es el tiempo de E; B va medio paso atrás).
    """
    E = np.array(E, float); B = np.array(B, float); dt = S * dx
    k = (S - 1) / (S + 1)
    Es, Bs, ts = [E.copy()], [B.copy()], [0.0]
    for n in range(pasos):
        B -= S * (E[1:] - E[:-1])
        E0, E1, Ef, Ef1 = E[0], E[1], E[-1], E[-2]
        E[1:-1] -= S * (B[1:] - B[:-1])
        if J is not None:
            E -= 4 * np.pi * dt * J(n)
        if absorbente:
            E[0] = E1 + k * (E[1] - E0)
            E[-1] = Ef1 + k * (E[-2] - Ef)
        if (n + 1) % cada == 0:
            Es.append(E.copy()); Bs.append(B.copy()); ts.append((n + 1) * dt)
    return np.array(Es), np.array(Bs), np.array(ts)
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
from scipy.integrate import quad
from scipy.special import j0, j1, erf
import math
import sympy as sp

# %% [markdown]
# ## Experimento 1 ★ — La luz sale de las ecuaciones de Maxwell
#
# Integramos las dos ecuaciones de rotor en una dimensión, $\partial_tB_z=-\partial_xE_y$ y $\partial_tE_y=-\partial_xB_z-4\pi J_y$ (con $c=1$), con el esquema de Yee de la herramienta `fdtd1d`: $E_y$ en los nodos, $B_z$ en los puntos medios y medio paso de tiempo atrás. El paso de tiempo es $dt=S\,dx$, con $S\le1$. Nada en el código "sabe" que hay ondas.
#
# Primero, en $t=0$, un pulso $E_y=e^{-x^2}$ con $B_z=0$.
#
# ### Predecí
# ¿Qué hace el pulso? ¿Con qué velocidad? ¿Y si en $t=0$ también hay un campo magnético $B_z=E_y$?

# %%
dx = 0.05
x = np.arange(-25, 25 + dx / 2, dx); xm = 0.5 * (x[1:] + x[:-1])
gauss = lambda u: np.exp(-u**2)

def B_inicial(F, S, sentido=0):
    """B_z en t = −dt/2. sentido = 0: B nulo en t = 0 (d'Alembert con f = g = F/2); +1: pulso que va hacia +x (B = E)."""
    dt = S * dx
    if sentido == 0:
        return 0.5 * (F(xm + dt / 2) - F(xm - dt / 2))
    return F(xm + dt / 2)

T = 15.0
pasos = int(round(T / dx))
Es, Bs, ts = yee1d(gauss(x), B_inicial(gauss, 1.0), pasos, S=1.0, dx=dx, cada=5)
Ed, Bd, td = yee1d(gauss(x), B_inicial(gauss, 1.0, +1), pasos, S=1.0, dx=dx, cada=5)

fig, axs = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
for ax, H, tit in ((axs[0], Es, "B = 0 en t = 0"), (axs[1], Ed, "B = E en t = 0")):
    ax.imshow(H, extent=(x[0], x[-1], ts[-1], 0), aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.plot([0, T], [0, T], "k:", lw=0.8); ax.plot([0, -T], [0, T], "k:", lw=0.8)
    ax.set(xlabel="x", title=tit, xlim=(-20, 20)); ax.grid(False)
axs[0].set_ylabel("t")
plt.show()

# %% [markdown]
# Un pulso cuadrado, con bordes abruptos, a $t=15$: con $S=1$ la grilla lo propaga exactamente; con $S<1$, mirá qué pasa.

# %%
cuadrado = lambda u: (np.abs(u) < 1).astype(float)
def mostrar(S=1.0):
    n = int(round(T / (S * dx)))
    E, B, t = yee1d(cuadrado(x), B_inicial(cuadrado, S), n, S=S, dx=dx, cada=n)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(x, 0.5 * (cuadrado(x - t[-1]) + cuadrado(x + t[-1])), color="0.6", lw=3, label="d'Alembert")
    ax.plot(x, E[-1], color=COLORES[0], lw=1.2, label=f"Yee, S = {S:g}")
    ax.set(xlabel="x", ylabel="E_y", title=f"t = {t[-1]:.2f}", xlim=(-20, 20)); ax.legend()
    plt.show()

interactuar(mostrar, S=deslizador("S", 1.0, 0.3, 1.0, 0.05))

exacta = 0.5 * (gauss(x - ts[-1]) + gauss(x + ts[-1]))
verificar("pulso con B = 0: E = [F(x − ct) + F(x + ct)]/2 (S = 1)", np.max(np.abs(Es[-1] - exacta)), 0.0, tol=1e-10)
verificar("con B = E solo hay pulso hacia +x", np.max(np.abs(Ed[-1][x < 0])), 0.0, tol=1e-10)
n = int(round(T / (0.5 * dx)))
E5, B5, t5 = yee1d(gauss(x), B_inicial(gauss, 0.5), n, S=0.5, dx=dx, cada=n)
der = x > 0
verificar("velocidad = c también con S = 0.5 (centro del pulso)", np.sum(x[der] * E5[-1][der]**2) / np.sum(E5[-1][der]**2) / t5[-1], 1.0, tol=1e-3)

# %% [markdown]
# **Una lámina que emite.** Ahora los campos empiezan nulos, y en $x=0$ hay una lámina con corriente $K(t)=e^{-(t-4)^2}$ (en la grilla, $J_y=K/dx$ en un nodo). Las notas predicen $E_y=-\frac{2\pi}{c}K\!\left(t-\frac{|x|}{c}\right)$.

# %%
i0 = np.argmin(np.abs(x))
K = lambda t: np.exp(-(t - 4.0)**2)
def J_lamina(n, S=1.0):
    j = np.zeros_like(x); j[i0] = K((n + 0.5) * S * dx) / dx
    return j
El, Bl, tl = yee1d(0 * x, 0 * xm, pasos, S=1.0, dx=dx, J=J_lamina, cada=5)
fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
axs[0].imshow(El, extent=(x[0], x[-1], tl[-1], 0), aspect="auto", cmap="RdBu_r", vmin=-2 * np.pi, vmax=2 * np.pi)
axs[0].set(xlabel="x", ylabel="t", title="E_y emitido por la lámina", xlim=(-20, 20)); axs[0].grid(False)
k = len(tl) - 1
axs[1].plot(x, -2 * np.pi * K(tl[k] - np.abs(x)), color="0.6", lw=3, label="−2πK(t − |x|/c)")
axs[1].plot(x, El[k], color=COLORES[1], lw=1.2, label="Yee")
axs[1].set(xlabel="x", ylabel="E_y", title=f"t = {tl[k]:.1f}", xlim=(-20, 20)); axs[1].legend(fontsize=9)
plt.show()
verificar("lámina: E_y = −2πK(t − |x|/c)  (error / 2π)", np.max(np.abs(El[k] + 2 * np.pi * K(tl[k] - np.abs(x)))) / (2 * np.pi), 0.0, tol=2e-3)

# %% [markdown]
# ### ¿Qué pasó?
# El pulso con $B_z=0$ se parte en dos mitades que se alejan con velocidad $c$ (la "V" en el mapa $x$–$t$): es la solución de d'Alembert con $f=g=F/2$. Si el pulso empieza con $B_z=E_y$, viaja entero hacia $+x$. La lámina emite hacia los dos lados un campo que repite, con retardo $|x|/c$, la forma de $K(t)$ y con signo opuesto. Las ondas no se pusieron a mano: salen de las dos ecuaciones de rotor.
#
# Con $S=1$ el esquema de Yee es exacto en una dimensión. Con $S<1$ el pulso cuadrado deja ondulaciones: las componentes de longitud de onda corta viajan más despacio en la grilla (dispersión numérica). Un pulso suave casi no las tiene.
#
# ## Experimento 2 — Dos superficies, la misma circulación
#
# Un hilo de $z=-1$ a $z=1$ con corriente $I=1$ carga una esfera chica arriba ($+Q$) y otra abajo ($-Q$), con $\dot Q=I$. Calculamos la circulación de $\mathbf B$ sobre un círculo de radio $s$ a altura $z_0$ de tres maneras:
# - con la ley de Biot–Savart del hilo (la herramienta `biot_savart`, con una poligonal abierta);
# - con Ampère–Maxwell y el **disco plano**: corriente de conducción (si el hilo lo atraviesa) más corriente de desplazamiento;
# - con Ampère–Maxwell y una **taza**: la pared del cilindro de radio $s$ entre $z_0$ y $h=4$, y la tapa en $z=h$. El hilo no la atraviesa: solo hay corriente de desplazamiento.
#
# ### Predecí
# Con $z_0=0$, ¿la circulación es $\frac{4\pi}{c}I$? ¿Y con $z_0=1.5$, por encima del extremo del hilo?

# %%
d, I = 1.0, 1.0
hilo = np.array([[0, 0, -d], [0, 0, d]])
def circulacion_BS(s, z0, N=400):
    V = espira(s, N, z0); W = np.roll(V, -1, axis=0)
    return np.sum(campo_poligonal(0.5 * (V + W), hilo, I_c=I, cerrada=False) * (W - V))

def dE(sp_, z, eje):
    """∂E/∂t de las dos cargas (por unidad de İ = I): componente 'z' o 's' en (s, z)."""
    tot = 0.0
    for zq, q in ((d, 1.0), (-d, -1.0)):
        r3 = (sp_**2 + (z - zq)**2)**1.5
        tot = tot + q * ((z - zq) if eje == "z" else sp_) / r3
    return I * tot

def disco(s, z0):
    """(4π/c) × (conducción + desplazamiento) por el disco plano."""
    cond = I if abs(z0) < d else 0.0
    desp = quad(lambda r: dE(r, z0, "z") * 2 * np.pi * r, 0, s, limit=200)[0] / (4 * np.pi)
    return 4 * np.pi * (cond + desp)

def taza(s, z0, h=4.0):
    """(4π/c) × desplazamiento por la pared (normal ŝ) y la tapa (normal ẑ)."""
    pared = quad(lambda z: dE(s, z, "s") * 2 * np.pi * s, z0, h, limit=200, points=[d])[0]
    tapa = quad(lambda r: dE(r, h, "z") * 2 * np.pi * r, 0, s, limit=200)[0]
    return pared + tapa

def mostrar_superficies(z0=0.0):
    ss = np.linspace(0.1, 4, 25)
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.plot(ss, [circulacion_BS(s, z0) for s in ss], "o", color=COLORES[0], label="Biot–Savart del hilo")
    ax.plot(ss, [disco(s, z0) for s in ss], color=COLORES[1], label="disco: conducción + desplazamiento")
    ax.plot(ss, [taza(s, z0) for s in ss], "--", color=COLORES[2], label="taza: solo desplazamiento")
    ax.axhline(4 * np.pi * I if abs(z0) < d else 0, color="0.5", ls=":", label="Ampère sin corregir")
    ax.set(xlabel="radio s del círculo", ylabel="∮B·dl", title=f"círculo a altura z₀ = {z0:g}"); ax.legend(fontsize=8)
    plt.show()

interactuar(mostrar_superficies, z0=deslizador("z₀", 0.0, -2.0, 2.0, 0.25))

exacta = lambda s, z0: 2 * np.pi * I * ((d - z0) / np.hypot(d - z0, s) + (d + z0) / np.hypot(d + z0, s))   # Clase 12
verificar("Biot–Savart = (4π/c) I d/√(d² + s²)  (z₀ = 0, s = 1)", circulacion_BS(1.0, 0.0), exacta(1.0, 0.0), tol=1e-4)
verificar("disco plano, conducción + desplazamiento", disco(1.0, 0.0), exacta(1.0, 0.0), tol=1e-8)
verificar("taza, solo desplazamiento", taza(1.0, 0.0), exacta(1.0, 0.0), tol=1e-8)
verificar("por encima del hilo (z₀ = 1.5): Biot–Savart = desplazamiento", circulacion_BS(1.0, 1.5), disco(1.0, 1.5), tol=1e-4)

# %% [markdown]
# ### ¿Qué pasó?
# La circulación es menor que $\frac{4\pi}{c}I$, y las tres cuentas coinciden: con el disco plano, la corriente de desplazamiento resta (el campo de las cargas cruza el disco hacia abajo y crece); con la taza, el hilo no la atraviesa y todo es desplazamiento. Por encima del hilo ($z_0>1$) no hay corriente de conducción por ningún lado y el campo magnético existe igual: lo produce la corriente de desplazamiento. Que Biot–Savart aplicada solo al hilo dé el resultado correcto se debe a que la corriente de desplazamiento de cada carga es radial y simétrica, y no produce campo.
#
# ## Experimento 3 — El capacitor a alta frecuencia
#
# Placas circulares de radio $a$, campo que oscila con frecuencia $\omega$. Las notas dan $E(s)=E_0J_0(\omega s/c)$ y $\mathcal B(s)=-iE_0J_1(\omega s/c)$, y la primera corrección a lo cuasiestático, $E\simeq E_0\left[1-(\omega s/2c)^2\right]$.
#
# ### Predecí
# ¿Hasta qué valor de $\omega a/c$ el campo es casi uniforme entre las placas? ¿Qué pasa cerca de $\omega a/c\simeq2.4$?

# %%
def mostrar_capacitor(wa=1.0):
    s = np.linspace(0, 1, 300)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(s, j0(wa * s), color=COLORES[0], label="E/E₀ = J₀(ωs/c)")
    ax.plot(s, 1 - (wa * s / 2)**2, "--", color=COLORES[0], lw=1, label="1 − (ωs/2c)²")
    ax.plot(s, j1(wa * s), color=COLORES[1], label=r"$|\mathcal{B}|/E_0=J_1(\omega s/c)$")
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set(xlabel="s / a", ylabel="amplitud", title=f"ωa/c = {wa:g}", ylim=(-0.5, 1.1)); ax.legend(fontsize=9)
    plt.show()

interactuar(mostrar_capacitor, wa=deslizador("ωa/c", 1.0, 0.1, 4.0, 0.1))

serie = lambda z, n=25: sum((-1)**m / math.factorial(m)**2 * (z / 2)**(2 * m) for m in range(n))
verificar("serie de J₀ (Clase 10) = scipy, en x = 2", serie(2.0), j0(2.0), tol=1e-12)
kk = 2.0; s = np.linspace(0.01, 1, 4001); h = s[1] - s[0]
E = j0(kk * s); dEds = np.gradient(E, h); Bc = -1j * j1(kk * s)
verificar("Faraday: dE/ds = −(iω/c)𝓑", np.max(np.abs(dEds[5:-5] - (-1j * kk * Bc[5:-5]).real)), 0.0, tol=1e-6)
resto = np.gradient(s * dEds, h) / s + kk**2 * E
verificar("Bessel: (1/s)(sE')' + (ω/c)²E = 0", np.max(np.abs(resto[10:-10])) / kk**2, 0.0, tol=1e-5)
verificar("cuasiestático: E(a)/E₀ ≃ 1 − (ωa/2c)² con ωa/c = 0.1", j0(0.1), 1 - 0.05**2, tol=1e-5)

# %% [markdown]
# ### ¿Qué pasó?
# Con $\omega a/c\lesssim0.3$ el campo es uniforme a menos de un 2%, y la primera corrección, $-(\omega s/2c)^2$, lo describe bien: es el campo inducido por el $\mathcal B$ que produce la corriente de desplazamiento. Cuando $\omega a/c$ se acerca a $2.405$, el primer cero de $J_0$, el campo cambia de signo dentro de las placas: el capacitor ya no es un capacitor sino una cavidad.
#
# ## Experimento 4 — Un cambio de calibre
#
# La lámina del Experimento 1, ahora con potenciales. Con $\phi=0$, las notas dan $A_y=2\pi\int_{-\infty}^{t-|x|/c}K(t')\,dt'$; con $K(t)=e^{-(t-4)^2}$, $\int_{-\infty}^\tau K=\frac{\sqrt\pi}{2}\left[1+\operatorname{erf}(\tau-4)\right]$. Hacemos un cambio de calibre con una función $f(x,t)$ que no tiene nada que ver con la lámina: $\phi'=-\partial_tf$, $A'_x=\partial_xf$, $A'_y=A_y$. Calculamos $\mathbf E$ y $\mathbf B$ en una grilla $x$–$t$ con derivadas centradas, en los dos calibres.
#
# ### Predecí
# Los potenciales nuevos tienen un "bulto" donde está $f$. ¿Aparece algo en los campos ahí?

# %%
h = 0.04
xg = np.arange(-10, 10 + h / 2, h); tg = np.arange(0, 20 + h / 2, h)
X, Tt = np.meshgrid(xg, tg)                       # eje 0: t, eje 1: x
Ay = 2 * np.pi * np.sqrt(np.pi) / 2 * (1 + erf(Tt - np.abs(X) - 4.0))
dt_ = lambda F: np.gradient(F, h, axis=0)
dx_ = lambda F: np.gradient(F, h, axis=1)

def campos(phi, Ax, Ay):
    """E_x, E_y, B_z (con c = 1; nada depende de y)."""
    return -dx_(phi) - dt_(Ax), -dt_(Ay), dx_(Ay)

f = 5 * np.exp(-((X - 2)**2 + (Tt - 8)**2) / 2)
Ex1, Ey1, Bz1 = campos(0 * X, 0 * X, Ay)
Ex2, Ey2, Bz2 = campos(-dt_(f), dx_(f), Ay)

fig, axs = plt.subplots(1, 3, figsize=(13, 4))
for ax, F, tit in ((axs[0], -dt_(f), "φ' (calibre nuevo)"), (axs[1], dx_(f), "A'_x (calibre nuevo)"), (axs[2], Ey2, "E_y (los dos calibres)")):
    m = np.max(np.abs(F))
    ax.imshow(F, extent=(xg[0], xg[-1], tg[-1], tg[0]), aspect="auto", cmap="RdBu_r", vmin=-m, vmax=m)
    ax.set(xlabel="x", title=tit); ax.grid(False)
axs[0].set_ylabel("t")
plt.show()

verificar("E_x en el calibre nuevo = 0", np.max(np.abs(Ex2)), 0.0, tol=1e-10)
verificar("E_y igual en los dos calibres", np.max(np.abs(Ey2 - Ey1)), 0.0, tol=1e-10)
verificar("B_z igual en los dos calibres", np.max(np.abs(Bz2 - Bz1)), 0.0, tol=1e-10)
lejos = np.abs(X) > 0.2
verificar("E_y = −2πK(t − |x|/c) (error / 2π)", np.max(np.abs(Ey1[lejos] + 2 * np.pi * K(Tt[lejos] - np.abs(X[lejos])))) / (2 * np.pi), 0.0, tol=2e-3)
g = np.exp(-(X - Tt + 5)**2)                       # □g = 0: mantiene el calibre de Lorenz
lorenz = lambda ff: (dt_(-dt_(ff)) + dx_(dx_(ff)))[3:-3, 3:-3]    # ∂_t φ' + ∂_x A'_x, lejos de los bordes de la grilla
print(f"condición de Lorenz con f gaussiana: máx |∂_tφ' + ∂_xA'_x| = {np.max(np.abs(lorenz(f))):.3f} (se rompe)")
verificar("con f = g(x − ct) la condición de Lorenz se mantiene", np.max(np.abs(lorenz(g))), 0.0, tol=1e-8)

# %% [markdown]
# ### ¿Qué pasó?
# Los potenciales nuevos tienen un bulto grande donde está $f$, pero los campos son idénticos, hasta el redondeo: $E_x=-\partial_x\phi'-\partial_tA'_x=\partial_x\partial_tf-\partial_t\partial_xf=0$. Lo que está en el calibre es una descripción; lo físico son $\mathbf E$ y $\mathbf B$. Una $f$ cualquiera rompe la condición de Lorenz, que en este caso valía ($\phi=0$, $\nabla\cdot\mathbf A=0$); una $f$ que cumple la ecuación de ondas, como $g(x-ct)$, la mantiene: es la libertad residual de las notas.
#
# ## Explorá
#
# 1. **Guía 7, P2.** En el Experimento 3, el campo del capacitor y su corrección salen de usar alternadamente Ampère–Maxwell y Faraday. En el solenoide del problema 2 los papeles se invierten: partí de $\mathbf B$ uniforme adentro, obtené $\mathbf E$ con Faraday y la corrección de $\mathbf B$ con la corriente de desplazamiento. Graficá el campo corregido para varios valores de $\omega a/c$.
# 2. **Guía 7, P3.** La celda de abajo define la divergencia y el rotor en cilíndricas con `sympy`, comprobados con un caso conocido. Usalos para obtener las cargas y corrientes del problema 3 a partir de los campos dados, y comprobá la ecuación de continuidad.
# 3. Con `yee1d`, poné dos láminas separadas un cuarto de longitud de onda, con corrientes $K_0\sin\omega t$ y $K_0\cos\omega t$. ¿Hacia qué lado emiten? Explicalo con la solución de las notas y la superposición.
# 4. Con $S<1$, mandá trenes de ondas sinusoidales de distintas longitudes de onda (medidas en celdas) y medí su velocidad en la grilla. ¿Desde cuántas celdas por longitud de onda la dispersión numérica es menor que un 1%?

# %%
s_, phi_, z_, t_ = sp.symbols("s varphi z t", positive=True)

def div_cil(Fs, Fphi, Fz):
    """Divergencia en cilíndricas (s, φ, z)."""
    return sp.simplify(sp.diff(s_ * Fs, s_) / s_ + sp.diff(Fphi, phi_) / s_ + sp.diff(Fz, z_))

def rot_cil(Fs, Fphi, Fz):
    """Rotor en cilíndricas: componentes (s, φ, z)."""
    return (sp.simplify(sp.diff(Fz, phi_) / s_ - sp.diff(Fphi, z_)),
            sp.simplify(sp.diff(Fs, z_) - sp.diff(Fz, s_)),
            sp.simplify(sp.diff(s_ * Fphi, s_) / s_ - sp.diff(Fs, phi_) / s_))

# comprobación con el campo de un hilo (Clase 12) y con el de un cilindro con corriente uniforme (Clase 13), con I/c = 1
print("hilo: div =", div_cil(0, 2 / s_, 0), " rot =", rot_cil(0, 2 / s_, 0))
print("adentro de un cilindro de radio 1: rot =", rot_cil(0, 2 * s_, 0), " (= 4πJ/c con J = I/π)")
verificar("rotor de B = 2Is/c (cilindro, I/c = 1) = 4πJ/c", float(rot_cil(0, 2 * s_, 0)[2]), 4.0, tol=1e-12)
