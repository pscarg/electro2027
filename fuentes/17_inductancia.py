# %% [markdown]
# # Clase 17 — Energía magnética, inductancia y circuitos
#
# **Objetivos**
# - Comprobar la **fórmula de Neumann** y la simetría $M_{12}=M_{21}$ para dos espiras de formas muy distintas.
# - Calcular la autoinductancia de una espira de alambre delgado, y ver que diverge cuando el alambre se hace infinitamente delgado.
# - Obtener la fuerza entre dos espiras a partir de la energía, a corriente constante y a flujo constante, y compararla con la fuerza de Lorentz.
# - Integrar circuitos RL y RLC, libres y forzados, y dos circuitos acoplados.
#
# **Material relacionado:** notas de la Clase 17. Guía 6: problemas 5 a 7.
#
# **Unidades.** Gaussianas, adimensionales: $c=1$, corrientes $I/c=1$ en las espiras, longitudes en unidades del radio de la espira grande.

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
from scipy.integrate import quad, solve_ivp, trapezoid

def tramos(V):
    """Puntos medios y vectores dl de los lados de una poligonal cerrada."""
    W = np.roll(V, -1, axis=0)
    return 0.5 * (V + W), W - V

def mutua(V1, V2, cerrada=True):
    """Inductancia mutua de Neumann como circulación: M = ∮₂ A₁·dl₂, con A₁ el potencial de la poligonal V1 para I₁/c = 1.
    V2 es siempre cerrada; V1 puede ser abierta (cerrada=False), por ejemplo un hilo recto largo."""
    P2, dl2 = tramos(V2)
    return np.sum(potencial_poligonal(P2, V1, cerrada=cerrada) * dl2)

def cuadrado(lado, centro, angulo, n=100):
    """Vértices de un cuadrado de lado `lado` (n por lado), girado `angulo` alrededor del eje x y centrado en `centro`."""
    esq = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1], [1, 1]]) * lado / 2
    u = np.arange(n) / n
    P = np.vstack([p + np.outer(u, q - p) for p, q in zip(esq[:-1], esq[1:])])
    P = np.column_stack([P, np.zeros(len(P))])
    ca, sa = np.cos(angulo), np.sin(angulo)
    giro = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
    return P @ giro.T + np.asarray(centro, float)

# %% [markdown]
# ## Experimento 1 ★ — La fórmula de Neumann: $M_{12}=M_{21}$
#
# Una espira circular de radio 1 en el plano $z=0$, y un cuadrado de lado $0.6$ centrado en $(0.5,\,0.2,\,0.7)$ y girado un ángulo $\theta$ alrededor del eje $x$. Calculamos dos flujos que no tienen nada que ver entre sí:
# - el del campo de la espira a través del cuadrado, $\Phi_{\square}=\oint_\square\mathbf A_\circ\cdot d\mathbf l$;
# - el del campo del cuadrado a través de la espira, $\Phi_\circ=\oint_\circ\mathbf A_\square\cdot d\mathbf l$.
#
# Con $I/c=1$, cada flujo es una inductancia mutua. Los dos campos son muy distintos, y las superficies también.
#
# ### Predecí
# 1. ¿Cuál de los dos flujos es mayor?
# 2. ¿Para qué ángulo se anulan?

# %%
circulo = espira(1.0, 600)
angulos = np.radians(np.arange(0, 181, 5))
M_cuadrado = np.array([mutua(circulo, cuadrado(0.6, (0.5, 0.2, 0.7), t)) for t in angulos])   # flujo del círculo por el cuadrado
M_circulo = np.array([mutua(cuadrado(0.6, (0.5, 0.2, 0.7), t), circulo) for t in angulos])    # flujo del cuadrado por el círculo

def mostrar(theta=30.0):
    t = np.radians(theta); sq = cuadrado(0.6, (0.5, 0.2, 0.7), t)
    fig = plt.figure(figsize=(12, 4.6))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    for V, col in ((circulo, COLORES[0]), (sq, COLORES[1])):
        W = np.vstack([V, V[:1]]); ax.plot(W[:, 0], W[:, 1], W[:, 2], color=col)
    ax.set(xlim=(-1, 1), ylim=(-1, 1), zlim=(-0.3, 1.2), xlabel="x", ylabel="y", zlabel="z", title=f"θ = {theta:.0f}°")
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(np.degrees(angulos), M_cuadrado, "o", color=COLORES[0], label="campo del círculo por el cuadrado")
    ax2.plot(np.degrees(angulos), M_circulo, "-", color=COLORES[1], label="campo del cuadrado por el círculo")
    ax2.axvline(theta, color="0.6", ls=":"); ax2.axhline(0, color="0.6", lw=0.8)
    ax2.set(xlabel="θ (grados)", ylabel="M", title="los dos flujos"); ax2.legend(fontsize=8)
    plt.show()

interactuar(mostrar, theta=deslizador("θ", 30.0, 0.0, 180.0, 5.0))

verificar("M₁₂ = M₂₁ en θ = 30°", M_cuadrado[6], M_circulo[6], tol=1e-4)
verificar("máx |M₁₂ − M₂₁| / máx |M| en todos los ángulos", np.max(np.abs(M_cuadrado - M_circulo)) / np.max(np.abs(M_circulo)), 0.0, tol=1e-4)

# %% [markdown]
# **Dos espiras coaxiales.** Radios $a=1$ y $b=0.5$, separadas $z$. Las notas reducen Neumann a una integral,
# $$M(z)=2\pi ab\int_0^{2\pi}\frac{\cos\psi\,d\psi}{\sqrt{a^2+b^2+z^2-2ab\cos\psi}},$$
# y lejos, con $r_0^2=a^2+b^2+z^2$, $M\simeq\dfrac{2\pi^2a^2b^2}{r_0^3}\simeq\dfrac{2\pi^2a^2b^2}{z^3}$.

# %%
a1, b1 = 1.0, 0.5
def M_coaxial(a, b, z):
    """La integral de las notas (sección 3.2)."""
    f = lambda p: np.cos(p) / np.sqrt(a * a + b * b + z * z - 2 * a * b * np.cos(p))
    return 2 * np.pi * a * b * quad(f, 0, 2 * np.pi, limit=200)[0]

zs = np.geomspace(0.05, 20, 30)
M_pol = np.array([mutua(espira(a1, 400), espira(b1, 400, z)) for z in zs])
M_int = np.array([M_coaxial(a1, b1, z) for z in zs])
fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.loglog(zs, M_int, color=COLORES[0], label="integral de las notas")
ax.loglog(zs, M_pol, "o", ms=4, color=COLORES[1], label="polígonos (Neumann)")
ax.loglog(zs, 2 * np.pi**2 * a1**2 * b1**2 / zs**3, "--", color="0.5", label="2π²a²b²/z³")
ax.set(xlabel="z", ylabel="M", title="espiras coaxiales, a = 1, b = 0.5", ylim=(1e-5, 20)); ax.legend()
plt.show()
verificar("polígonos = integral (z = 0.5)", mutua(espira(a1, 400), espira(b1, 400, 0.5)), M_coaxial(a1, b1, 0.5), tol=1e-3)
r0 = np.sqrt(a1**2 + b1**2 + 100)
verificar("lejos: M = 2π²a²b²/r₀³ (z = 10)", M_coaxial(a1, b1, 10.0), 2 * np.pi**2 * a1**2 * b1**2 / r0**3, tol=1e-3)

# %% [markdown]
# ### ¿Qué pasó?
# Los dos flujos coinciden para todos los ángulos (a $10^{-5}$, el error de discretizar las curvas), aunque se calculan con campos distintos sobre superficies distintas: la fórmula de Neumann es simétrica, $\oint\oint\frac{d\mathbf l_1\cdot d\mathbf l_2}{|\mathbf r_1-\mathbf r_2|}$. Se anulan entre $80^\circ$ y $85^\circ$: ahí las líneas del campo de la espira que entran al cuadrado vuelven a salir por él, y las del cuadrado que atraviesan la espira hacia arriba se compensan con las que la atraviesan hacia abajo. Para las espiras coaxiales, la cuenta con polígonos reproduce la integral de las notas, y lejos $M\propto z^{-3}$, como el flujo de un dipolo.
#
# ## Experimento 2 — La autoinductancia de una espira de alambre delgado
#
# Una espira de radio $R=1$ hecha con alambre de radio $a$. Las notas muestran que $L$ es la inductancia mutua entre el hilo central y el círculo de radio $R-a$, y que para $a\ll R$
# $$L\simeq4\pi R\left[\ln\frac{8R}{a}-2\right].$$
#
# ### Predecí
# Si el alambre se hace diez veces más fino, ¿cuánto cambia $L$? ¿Tiene $L$ un límite cuando $a\to0$?

# %%
R = 1.0
radios = np.geomspace(1e-3, 0.3, 12)
L_pol = np.array([mutua(espira(R, 1000), espira(R - a, 1000)) for a in radios])
L_int = np.array([M_coaxial(R, R - a, 0.0) for a in radios])
L_asin = 4 * np.pi * R * (np.log(8 * R / radios) - 2)
fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
x = np.log(R / radios)
axs[0].plot(x, L_int / (4 * np.pi * R), color=COLORES[0], label="integral de Neumann")
axs[0].plot(x, L_pol / (4 * np.pi * R), "o", ms=4, color=COLORES[1], label="polígonos")
axs[0].plot(x, L_asin / (4 * np.pi * R), "--", color="0.4", label="ln(8R/a) − 2")
axs[0].set(xlabel="ln(R/a)", ylabel="L / 4πR", title="L crece sin límite cuando a → 0"); axs[0].legend()
axs[1].loglog(radios, np.abs(L_asin - L_int) / L_int, "o-", color=COLORES[2])
axs[1].set(xlabel="a / R", ylabel="error relativo", title="la fórmula asintótica: error de orden a/R")
plt.show()

verificar("polígonos = integral de Neumann (a = 0.01)", mutua(espira(R, 1000), espira(R - 0.01, 1000)), M_coaxial(R, R - 0.01, 0.0), tol=1e-3)
verificar("L ≃ 4πR[ln(8R/a) − 2] (a = 10⁻³)", M_coaxial(R, R - 1e-3, 0.0), 4 * np.pi * R * (np.log(8000) - 2), tol=1e-3)
verificar("L(a/10) − L(a) = 4πR ln 10 (a = 10⁻³)", M_coaxial(R, R - 1e-4, 0.0) - M_coaxial(R, R - 1e-3, 0.0), 4 * np.pi * R * np.log(10), tol=2e-3)

# %% [markdown]
# ### ¿Qué pasó?
# $L/4\pi R$ en función de $\ln(R/a)$ es una recta de pendiente 1: cada vez que el alambre se hace diez veces más fino, $L$ aumenta en $4\pi R\ln10\simeq29R$, sin límite. Un hilo infinitamente delgado tendría autoinductancia (y energía) infinita, por la contribución del campo $2I/cs$ muy cerca del alambre. La fórmula asintótica tiene un error relativo proporcional a $a/R$: con $a/R=10^{-3}$ es de $6\times10^{-4}$.
#
# ## Experimento 3 — La fuerza a partir de la energía
#
# Dos espiras coaxiales, de radios $a=1$ (en $z=0$) y $b=0.5$ (a altura $z$), de alambre de radio $0.01$, con corrientes $I/c=1$ del mismo sentido. Calculamos:
# - la energía a corriente constante, $U_I(z)=\frac{1}{2c^2}\left(L_1I_1^2+2M(z)I_1I_2+L_2I_2^2\right)$;
# - la energía a flujo constante, $U_\Phi(z)=\frac12\boldsymbol\Phi^T\mathsf L(z)^{-1}\boldsymbol\Phi$, con los flujos fijados en $z_0=0.8$;
# - la fuerza de Lorentz sobre la espira de arriba, $\mathbf F=\frac{I_2}{c}\oint d\mathbf l_2\times\mathbf B_1$, con la herramienta de Biot–Savart.
#
# ### Predecí
# Las espiras se atraen. Si la de arriba baja un poco con las corrientes fijas (pilas), ¿la energía del campo sube o baja? ¿Y si las espiras son superconductoras (flujos fijos)?

# %%
alambre = 0.01
L1 = 4 * np.pi * a1 * (np.log(8 * a1 / alambre) - 2)
L2 = 4 * np.pi * b1 * (np.log(8 * b1 / alambre) - 2)
M_de = lambda z: mutua(espira(a1, 400), espira(b1, 400, z))
def matriz(z):
    m = M_de(z); return np.array([[L1, m], [m, L2]])

z0 = 0.8
I0 = np.array([1.0, 1.0])                              # I/c
Phi0 = matriz(z0) @ I0                                 # Φ_i = Σ L_ij I_j / c
def U_corriente(z):
    return 0.5 * I0 @ matriz(z) @ I0
def U_flujo(z):
    return 0.5 * Phi0 @ np.linalg.solve(matriz(z), Phi0)

zz = np.linspace(0.4, 1.4, 21)
UI = np.array([U_corriente(z) for z in zz]) - U_corriente(z0)
UF = np.array([U_flujo(z) for z in zz]) - U_flujo(z0)
P2, dl2 = tramos(espira(b1, 400, z0))
F_lorentz = np.sum(np.cross(dl2, campo_poligonal(P2, espira(a1, 400))), axis=0)[2]

fig, ax = plt.subplots(figsize=(6.6, 4.4))
ax.plot(zz, UI, color=COLORES[0], label="corriente constante")
ax.plot(zz, UF, color=COLORES[1], label="flujo constante")
ax.plot(zz, F_lorentz * (zz - z0), "k:", lw=1, label="pendiente ±F de Lorentz")
ax.plot(zz, -F_lorentz * (zz - z0), "k:", lw=1)
ax.axvline(z0, color="0.7", lw=0.8)
ax.set(xlabel="z de la espira de arriba", ylabel="U − U(z₀)", title="la misma fuerza, dos energías"); ax.legend(fontsize=9)
plt.show()

h = 1e-3
dUI = (U_corriente(z0 + h) - U_corriente(z0 - h)) / (2 * h)
dUF = (U_flujo(z0 + h) - U_flujo(z0 - h)) / (2 * h)
print(f"fuerza de Lorentz F_z = {F_lorentz:.6f} (negativa: atracción)")
verificar("F = +(∂U/∂z) a corriente constante", dUI, F_lorentz, tol=1e-4)
verificar("F = −(∂U/∂z) a flujo constante", -dUF, F_lorentz, tol=1e-4)
zl = 30.0
dM_lejos = (M_de(zl + 0.01) - M_de(zl - 0.01)) / 0.02
m1, m2 = np.pi * a1**2, np.pi * b1**2
verificar("lejos: F = −6m₁m₂/z⁴ (z = 30, Clase 14)", dM_lejos, -6 * m1 * m2 / zl**4, tol=5e-3)

# %% [markdown]
# ### ¿Qué pasó?
# A corriente constante, la energía del campo **sube** cuando la espira baja (porque $M$ crece), y la pendiente es $+F$; a flujo constante, la energía **baja**, con pendiente $-F$. Las dos dan la fuerza de Lorentz, que depende solo de las corrientes y las posiciones en ese instante. Con pilas, cada vez que la espira baja $\delta z$, las pilas entregan $2\,\delta U$: la mitad queda en el campo y la otra mitad es el trabajo de la fuerza. Lejos, la fuerza es la de dos dipolos, $-6m_1m_2/z^4$.
#
# ## Experimento 4 — Circuitos
#
# La ecuación de las notas, con $c=1$: $L\dot I+RI+Q/C=\mathcal E(t)$, con $I=\dot Q$. Para un circuito sin capacitor ponemos $C=\infty$.
#
# ### Predecí
# 1. Una bobina con corriente $I_0$ se descarga en una resistencia $R$. Si duplicamos $R$, ¿qué pasa con el calor total disipado?
# 2. En el circuito RLC forzado, ¿a qué frecuencia la corriente es máxima, y qué tan ancha es la resonancia?

# %%
def circuito(L, R, C, fem=lambda t: 0.0, Q0=0.0, I0=0.0, t_fin=20.0, n=2001):
    """Integra L dI/dt + R I + Q/C = fem(t). Devuelve t, Q, I."""
    invC = 0.0 if np.isinf(C) else 1.0 / C
    f = lambda t, y: [y[1], (fem(t) - R * y[1] - y[0] * invC) / L]
    t = np.linspace(0, t_fin, n)
    s = solve_ivp(f, (0, t_fin), [Q0, I0], t_eval=t, rtol=1e-10, atol=1e-12)
    return s.t, s.y[0], s.y[1]

def mostrar_rl(R=1.0):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for r, col in ((1.0, "0.6"), (R, COLORES[0])):
        t, Q, I = circuito(1.0, r, np.inf, I0=1.0, t_fin=10.0)
        calor = np.concatenate([[0], np.cumsum(r * 0.5 * (I[1:]**2 + I[:-1]**2) * np.diff(t))])
        axs[0].plot(t, I, color=col, label=f"R = {r:g}")
        axs[1].plot(t, calor, color=col, label=f"R = {r:g}")
    axs[1].axhline(0.5, color="k", ls="--", lw=1, label="LI₀²/2c²")
    axs[0].set(xlabel="t", ylabel="I", title="la corriente decae con τ = L/c²R"); axs[0].legend()
    axs[1].set(xlabel="t", ylabel="calor acumulado", title="el calor total no depende de R"); axs[1].legend()
    plt.show()

interactuar(mostrar_rl, R=deslizador("R", 2.0, 0.2, 5.0, 0.1))

for r in (0.5, 2.0):
    t, Q, I = circuito(1.0, r, np.inf, I0=1.0, t_fin=60.0 / r, n=20001)
    verificar(f"calor = LI₀²/2c² con R = {r:g}", trapezoid(r * I**2, t), 0.5, tol=1e-5)

# %% [markdown]
# **RLC libre y forzado.** Con $L=1$, $C=1$ ($\omega_0=1$) y $R$ en el deslizador, $\gamma=R$. A la izquierda, la descarga libre del capacitor y las dos energías; a la derecha, la amplitud de la corriente con una fuente $\cos\omega t$: la curva es $1/|Z|$, y los puntos salen de integrar la ecuación hasta que el transitorio desaparece.

# %%
def amplitud_estacionaria(R, w, L=1.0, C=1.0):
    gamma = R / L; t_fin = 40.0 / gamma + 20 * 2 * np.pi / w
    t, Q, I = circuito(L, R, C, fem=lambda t: np.cos(w * t), t_fin=t_fin, n=40001)
    ultimos = t > t_fin - 3 * 2 * np.pi / w
    return np.max(np.abs(I[ultimos]))

def mostrar_rlc(R=0.2):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    t, Q, I = circuito(1.0, R, 1.0, Q0=1.0, t_fin=40.0)
    axs[0].plot(t, 0.5 * Q**2, color=COLORES[3], label="eléctrica Q²/2C")
    axs[0].plot(t, 0.5 * I**2, color=COLORES[0], label="magnética LI²/2c²")
    axs[0].plot(t, 0.5 * Q**2 + 0.5 * I**2, "k", lw=1, label="total")
    axs[0].set(xlabel="t", ylabel="energía", title=r"la energía va y viene, y decae como $e^{-\gamma t}$"); axs[0].legend(fontsize=8)
    w = np.linspace(0.3, 1.7, 400)
    Z = R - 1j * (w - 1 / w)
    axs[1].plot(w, 1 / np.abs(Z), color=COLORES[0], label="1/|Z|")
    wp = np.array([0.6, 0.9, 1.0, 1.1, 1.4])
    axs[1].plot(wp, [amplitud_estacionaria(R, x) for x in wp], "o", color=COLORES[1], label="integrando")
    axs[1].axhline(1 / (R * np.sqrt(2)), color="0.6", ls=":", lw=1)
    axs[1].set(xlabel="ω", ylabel="amplitud de I", title=f"resonancia: ancho γ = {R:g}"); axs[1].legend()
    plt.show()

interactuar(mostrar_rlc, R=deslizador("R", 0.2, 0.05, 1.0, 0.05))

R_ = 0.2
t, Q, I = circuito(1.0, R_, 1.0, Q0=1.0, t_fin=60.0, n=60001)
cruces = t[1:][np.diff(np.sign(Q)) != 0]
verificar("ω_d = √(ω₀² − γ²/4) (ceros de Q)", np.pi / np.mean(np.diff(cruces)), np.sqrt(1 - R_**2 / 4), tol=1e-4)
verificar("amplitud en resonancia = 𝓔₀/R", amplitud_estacionaria(R_, 1.0), 1 / R_, tol=1e-3)
w_mas = np.sqrt(1 + R_**2 / 4) + R_ / 2
verificar("en ω₊ = √(ω₀² + γ²/4) + γ/2 la amplitud es 𝓔₀/(R√2)", amplitud_estacionaria(R_, w_mas), 1 / (R_ * np.sqrt(2)), tol=1e-3)

# %% [markdown]
# **Dos circuitos acoplados: el anillo de Faraday.** Primario con $L_1=1$, $R_1=1$; secundario con $L_2$ y $R_2=1$; $M=0.8$. En $t=0$ conectamos una pila $\mathcal E=1$ al primario, y en $t=50$ la sacamos (dejando el circuito cerrado). Las notas predicen que por el secundario pasa una carga $Q_2=-M\,\Delta I_1/c^2R_2$ en cada cambio, sin importar $L_2$: al conectar, $-M\mathcal E/c^2R_1R_2$.

# %%
def acoplados(L2, M=0.8, R1=1.0, R2=1.0, L1=1.0, t_cambio=50.0, t_fin=100.0):
    """Primario con una pila 𝓔 = 1 entre t = 0 y t_cambio; secundario sin fuente. Devuelve t, (I₁, I₂, Q₂) y M."""
    Lm = np.array([[L1, M], [M, L2]])
    fem = lambda t: np.array([1.0 if t < t_cambio else 0.0, 0.0])
    f = lambda t, y: np.r_[np.linalg.solve(Lm, fem(t) - np.array([R1, R2]) * y[:2]), y[1]]   # (I₁, I₂, Q₂)
    t = np.linspace(0, t_fin, 5001)
    s = solve_ivp(f, (0, t_fin), [0, 0, 0], t_eval=t, rtol=1e-10, atol=1e-12, max_step=0.02)
    return s.t, s.y, M

fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for L2_, col in ((1.0, COLORES[0]), (4.0, COLORES[2])):
    t, (I1, I2, Q2), M = acoplados(L2_)
    for ax in axs:
        ax.plot(t, I1, color=col, lw=1, ls="--", label=f"I₁ (L₂ = {L2_:g})")
        ax.plot(t, I2, color=col, label=f"I₂ (L₂ = {L2_:g})")
for ax, (t0, tit) in zip(axs, ((0.0, "al conectar la pila"), (50.0, "al sacarla"))):
    ax.axhline(0, color="0.6", lw=0.8); ax.set(xlim=(t0 - 1, t0 + 15), xlabel="t", title=tit)
axs[0].set(ylabel="corriente"); axs[1].legend(fontsize=8)
plt.show()

for L2_ in (1.0, 4.0):
    t, (I1, I2, Q2), M = acoplados(L2_)
    verificar(f"Q₂ al conectar = −M𝓔/c²R₁R₂ (L₂ = {L2_:g})", Q2[np.searchsorted(t, 50.0)], -M, tol=1e-3)
    verificar(f"Q₂ al desconectar = +M𝓔/c²R₁R₂ (L₂ = {L2_:g})", Q2[-1] - Q2[np.searchsorted(t, 50.0)], M, tol=1e-3)

# %% [markdown]
# ### ¿Qué pasó?
# - **RL:** con $R$ mayor la corriente cae más rápido, pero el calor total es siempre $LI_0^2/2c^2$, la energía que había en el campo.
# - **RLC:** la energía pasa del capacitor a la bobina y vuelve, dos veces por período, y decae como $e^{-\gamma t}$. La frecuencia de la oscilación libre es $\omega_d=\sqrt{\omega_0^2-\gamma^2/4}$. Forzado, la amplitud es máxima en $\omega_0$, vale $\mathcal E_0/R$ ahí, y la integración en el tiempo cae sobre la curva $1/|Z|$ que da el método complejo: el ancho a media potencia es exactamente $\gamma$.
# - **Acoplados:** la corriente del secundario es un pulso al conectar y otro opuesto al desconectar, como en el experimento de Faraday. La forma del pulso depende de $L_2$; la carga total no.
#
# ## Explorá
#
# 1. **Guía 6, P5.** Con `circuito`, poné en la espira la fem que produce un flujo externo $\Phi(t)=At$ desde $t=0$ (pensá cuál es) y mirá la corriente para valores de $R$ cada vez más chicos. Graficá el flujo total, el externo más el propio $LI/c$. ¿Qué pasa con él cuando $R\to0$?
# 2. **Guía 6, P6.** Con `mutua(V1, V2, cerrada=False)`, calculá la inductancia mutua entre un hilo recto largo (una poligonal abierta, por ejemplo de $y=-200$ a $y=200$ con muchos puntos) y una espira cuadrada de lado $a$ en el mismo plano, con el lado más cercano a distancia $d$. Compará con tu resultado, y obtené la fuerza derivando $M$ respecto de $d$; comparala con $\frac{I_2}{c}\oint d\mathbf l\times\mathbf B$ calculada con `campo_poligonal`.
# 3. **Guía 6, P7.** Armá dos bobinas como pilas de espiras (`espira(R, N, z)` para varios `z`), una encima de la otra, y calculá su inductancia mutua sumando `mutua` entre todos los pares. ¿Cómo depende del número de vueltas de cada una? Después escribí las ecuaciones acopladas de las notas (sección 5.5) con una fuente $V_0\sin\omega t$ en el primario y una resistencia $R$ en el secundario, integralas como en `acoplados`, y mirá la amplitud en el secundario para $R\to0$ y $R\to\infty$.
# 4. Repetí el Experimento 3 con las corrientes en sentidos opuestos, y con la espira chica corrida del eje: ¿sigue valiendo $\mathbf F=\frac{I_1I_2}{c^2}\nabla M$ en las tres componentes?
