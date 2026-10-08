# %% [markdown]
# # Clase 30 — Formulación covariante: E y B en otro sistema, Maxwell en una línea
#
# **Objetivos**
# - Transformar $\mathbf E$ y $\mathbf B$ como las componentes de un tensor, $F'=\Lambda F\Lambda^T$, y ver que $E^2-B^2$ y $\mathbf E\cdot\mathbf B$ no cambian.
# - Encontrar el sistema en el que los campos cruzados de la deriva $\mathbf E\times\mathbf B$ son solo magnéticos.
# - Obtener el campo de una carga en movimiento transformando el de Coulomb, y calcular la energía y el momento de su campo: el problema de los $\frac43$.
# - Ver el efecto Doppler en los campos de una onda plana y en los frentes de una fuente que se mueve, y la concentración hacia adelante de la luz de una fuente rápida.
#
# **Material relacionado:** notas de la Clase 30. Guía 12: problemas 5 a 9.
#
# **Unidades.** Gaussianas con $c=1$; cargas y radios de orden 1.

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
from numpy.polynomial.legendre import leggauss

# %% [markdown]
# ## Experimento 1 ★ — Los campos vistos desde otro sistema
#
# $\mathbf E$ y $\mathbf B$ son las componentes del tensor $F^{\mu\nu}$ (Clase 29), con $F^{i0}=E^i$ y $F^{ij}=-\epsilon_{ijk}B^k$. En otro sistema, $F'^{\mu\nu}=\Lambda^\mu{}_\alpha\Lambda^\nu{}_\beta F^{\alpha\beta}$, es decir $F'=\Lambda F\Lambda^T$ como matrices. Para un boost con velocidad $\boldsymbol\beta c$ en una dirección cualquiera, $\Lambda^0{}_0=\gamma$, $\Lambda^0{}_i=\Lambda^i{}_0=-\gamma\beta_i$ y $\Lambda^i{}_j=\delta_{ij}+(\gamma-1)\beta_i\beta_j/\beta^2$. Las notas (sección 1) dan el resultado en forma vectorial: $\mathbf E'_\parallel=\mathbf E_\parallel$, $\mathbf E'_\perp=\gamma(\mathbf E+\boldsymbol\beta\times\mathbf B)_\perp$, $\mathbf B'_\parallel=\mathbf B_\parallel$, $\mathbf B'_\perp=\gamma(\mathbf B-\boldsymbol\beta\times\mathbf E)_\perp$.
#
# Tomamos los campos cruzados de la deriva de las Clases 14 y 29, $\mathbf E=E\hat{\mathbf y}$ y $\mathbf B=B\hat{\mathbf z}$ con $E<B$, y nos movemos según $\hat{\mathbf x}$ (la dirección de $\mathbf E\times\mathbf B$) con velocidad $\beta c$.
#
# ### Predecí
# ¿Hay una velocidad para la que el campo eléctrico desaparece? ¿Cuánto vale ahí el magnético? ¿Qué cantidades no cambian con $\beta$?

# %%
def tensor_F(E, B):
    """F^{μν} con F^{i0} = E^i y F^{ij} = −ε_ijk B^k (Clase 29)."""
    Ex, Ey, Ez = E; Bx, By, Bz = B
    return np.array([[0, -Ex, -Ey, -Ez], [Ex, 0, -Bz, By], [Ey, Bz, 0, -Bx], [Ez, -By, Bx, 0]], float)

def campos_de_F(F):
    return F[1:, 0].copy(), np.array([F[3, 2], F[1, 3], F[2, 1]])

def boost(beta):
    """Λ^μ_ν al sistema que se mueve con velocidad beta·c (un vector de 3 componentes)."""
    beta = np.asarray(beta, float); b2 = beta @ beta
    g = 1 / np.sqrt(1 - b2); L = np.eye(4)
    L[0, 0] = g; L[0, 1:] = L[1:, 0] = -g * beta
    if b2 > 0:
        L[1:, 1:] += (g - 1) * np.outer(beta, beta) / b2
    return L

def transformar_campos(E, B, beta):
    L = boost(beta)
    return campos_de_F(L @ tensor_F(E, B) @ L.T)

def mostrar_cruzados(e=0.5, beta=0.3, exportar=False):
    bs = np.linspace(-0.98, 0.98, 400)
    Ep = np.array([transformar_campos([0, e, 0], [0, 0, 1], [b, 0, 0])[0] for b in bs])
    Bp = np.array([transformar_campos([0, e, 0], [0, 0, 1], [b, 0, 0])[1] for b in bs])
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.plot(bs, Ep[:, 1], color=COLORES[1], label="E'_y")
    ax.plot(bs, Bp[:, 2], color=COLORES[0], label="B'_z")
    ax.plot(bs, Ep[:, 1]**2 - Bp[:, 2]**2, "--", color="k", lw=1.2, label="E'² − B'²  (invariante)")
    ax.axhline(0, color="gray", lw=0.8)
    Eb, Bb = transformar_campos([0, e, 0], [0, 0, 1], [beta, 0, 0])
    ax.plot([beta], [Eb[1]], "o", color=COLORES[1]); ax.plot([beta], [Bb[2]], "o", color=COLORES[0])
    ax.axvline(e, color=COLORES[2], lw=1.2, label=f"β = E/B = {e:g}: E' = 0")
    ax.set(xlabel="β (velocidad de S' según E×B)", ylabel="campos en S'", ylim=(-3, 4),
           title=f"E = {e:g} ŷ, B = ẑ en S;  en β = {beta:g}: E'·B' = {Eb @ Bb:.3g}, E'² − B'² = {Eb @ Eb - Bb @ Bb:.3f}")
    ax.legend(fontsize=9, loc="upper left")
    if exportar:
        guardar(fig, "nb30_cruzados")
    plt.show()

interactuar(mostrar_cruzados, e=deslizador("E/B", 0.5, 0.0, 0.95, 0.05), beta=deslizador("β", 0.3, -0.95, 0.95, 0.05))
if CARPETA_FIGURAS:
    mostrar_cruzados(0.5, 0.3, exportar=True)

# Comprobaciones
rng = np.random.default_rng(7)
err_f, err_i = 0.0, 0.0
for _ in range(100):
    E, B = rng.normal(size=3), rng.normal(size=3)
    beta = rng.normal(size=3); beta *= rng.uniform(0, 0.95) / np.linalg.norm(beta)
    Ep, Bp = transformar_campos(E, B, beta)
    g = 1 / np.sqrt(1 - beta @ beta); n = beta / np.linalg.norm(beta)
    par = lambda V: (V @ n) * n
    Ef = par(E) + g * ((E + np.cross(beta, B)) - par(E + np.cross(beta, B)))
    Bf = par(B) + g * ((B - np.cross(beta, E)) - par(B - np.cross(beta, E)))
    err_f = max(err_f, np.abs(Ep - Ef).max(), np.abs(Bp - Bf).max())
    err_i = max(err_i, abs((Ep @ Ep - Bp @ Bp) - (E @ E - B @ B)), abs(Ep @ Bp - E @ B))
verificar("F' = ΛFΛᵀ coincide con las fórmulas vectoriales de E' y B' (100 casos al azar)", err_f, 0.0, tol=1e-12)
verificar("E² − B² y E·B no cambian (100 casos al azar)", err_i, 0.0, tol=1e-12)
Ep, Bp = transformar_campos([0, 0.5, 0], [0, 0, 1], [0.5, 0, 0])
verificar("campos cruzados con E = B/2: en el sistema que se mueve con cE/B, E' = 0", np.linalg.norm(Ep), 0.0, tol=1e-12)
verificar("y ahí B' = B/γ", Bp[2], np.sqrt(1 - 0.25), tol=1e-12)
sig, b = 1.0, 0.6; g = 1 / np.sqrt(1 - b**2)
Ep, Bp = transformar_campos([0, 4 * np.pi * sig, 0], [0, 0, 0], [b, 0, 0])
verificar("capacitor que se mueve: E' = 4π(γσ), con la densidad contraída", Ep[1], 4 * np.pi * g * sig, tol=1e-12)
verificar("y |B'| = (4π/c)K', con la corriente superficial K' = γσv", abs(Bp[2]), 4 * np.pi * g * sig * b, tol=1e-12)

# %% [markdown]
# ### ¿Qué pasó?
# Al moverse según $\mathbf E\times\mathbf B$, $E'_y=\gamma(E-\beta B)$ baja y se anula en $\beta=E/B$: **en el sistema que se mueve con la deriva, solo hay campo magnético**, $B'=B/\gamma$. Ahí la carga gira en círculos sin avanzar; vista desde el laboratorio, su centro avanza con $cE/B$, exactamente, a cualquier velocidad. Es la deriva de la Clase 14 y la de la Clase 29. Mientras los campos cambian, $E'^2-B'^2$ (la línea negra punteada) queda fija, y también $\mathbf E'\cdot\mathbf B'=0$. El capacitor que se mueve tiene un campo eléctrico mayor, por la contracción de la densidad de carga, y un campo magnético, el de sus dos placas convertidas en láminas de corriente.

# %% [markdown]
# ## Experimento 2 — Una carga en movimiento y los $\frac43$
#
# En el sistema de la carga, el campo es el de Coulomb, $\mathbf E_0=q\mathbf r_0/r_0^3$, sin $\mathbf B$. En el laboratorio la carga se mueve con $\beta c\,\hat{\mathbf x}$ y está en el origen en $t=0$. Un evento del laboratorio $(t=0,\mathbf r)$ tiene en el sistema de la carga $x_0=\gamma x$, $y_0=y$, $z_0=z$. Transformamos el campo de Coulomb en cada punto con `transformar_campos` y lo comparamos con el de la Clase 24, que salió de los potenciales de Liénard–Wiechert:
# $$\mathbf E=\frac{q(1-\beta^2)\,\mathbf R}{R^3\left(1-\beta^2\sin^2\psi\right)^{3/2}},\qquad\mathbf B=\boldsymbol\beta\times\mathbf E ,$$
# con $\mathbf R$ desde la posición presente y $\psi$ el ángulo entre $\mathbf R$ y la velocidad.
#
# Después calculamos la energía y el momento del campo de una cáscara esférica de carga $q$ y radio $a$ que se mueve (en el laboratorio es un elipsoide achatado, y el campo afuera es el de una carga puntual). Según las notas (sección 5), $\mathcal E_{\text{campo}}=\gamma U\left(1+\frac{\beta^2}{3}\right)$ y $cP_{\text{campo}}=\frac43\gamma\beta U$, con $U=q^2/2a$.
#
# ### Predecí
# ¿El campo de una carga que se mueve es más intenso adelante o al costado? ¿Forman $(\mathcal E_{\text{campo}},cP_{\text{campo}})$ un cuadrivector?

# %%
def campo_carga_movil(x, y, z, beta, q=1.0):
    """E y B en t = 0 de una carga en el origen que se mueve con β c x̂ (fórmula de la Clase 24)."""
    R = np.sqrt(x**2 + y**2 + z**2); s2 = (y**2 + z**2) / R**2
    f = q * (1 - beta**2) / (R**3 * (1 - beta**2 * s2)**1.5)
    E = np.stack([f * x, f * y, f * z]); B = np.stack([0 * x, -beta * E[2], beta * E[1]])   # β × E con β = β x̂
    return E, B

def coulomb_transformado(x, y, z, beta, q=1.0):
    """El campo de Coulomb en el sistema de la carga, transformado al laboratorio en el evento (t = 0, x, y, z)."""
    g = 1 / np.sqrt(1 - beta**2)
    x0 = g * x; r0 = np.sqrt(x0**2 + y**2 + z**2)
    E0 = q * np.array([x0, y, z]) / r0**3
    return transformar_campos(E0, [0, 0, 0], [-beta, 0, 0])      # el laboratorio se mueve con −β respecto de la carga

def mostrar_carga(beta=0.8, exportar=False):
    th = np.linspace(0, 2 * np.pi, 33)[:-1]
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 5))
    for ax, b, tit in ((axs[0], 0.0, "en reposo"), (axs[1], beta, f"β = {beta:g}")):
        for r in (1.0, 1.6):
            x, y = r * np.cos(th), r * np.sin(th)
            E, _ = campo_carga_movil(x, y, 0 * x, b)
            ax.quiver(x, y, E[0], E[1], color=COLORES[1], scale=12, width=0.006)
        rr = np.linspace(0.3, 3, 200)
        for t0 in th[::2]:
            ax.plot(rr * np.cos(t0), rr * np.sin(t0), color="gray", lw=0.4, alpha=0.5)
        ax.plot(0, 0, "o", color="k")
        if b > 0:
            ax.annotate("", xy=(0.6, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="->", lw=2))
        ax.set(aspect="equal", xlim=(-2.2, 2.2), ylim=(-2.2, 2.2), title=f"campo eléctrico de una carga {tit}")
        ax.grid(False)
    if exportar:
        guardar(fig, "nb30_carga")
    plt.show()

interactuar(mostrar_carga, beta=deslizador("β", 0.8, 0.0, 0.95, 0.05))
if CARPETA_FIGURAS:
    mostrar_carga(0.8, exportar=True)

def energia_momento_cascara(beta, a=1.0, q=1.0, n=200):
    """Integra (E² + B²)/8π y (E × B)_x/4π fuera del elipsoide γ²x² + ρ² > a² (laboratorio, t = 0), con c = 1.
    Con ξ = γx, la región es r' > a en (ξ, ρ), y d³x = (2π/γ) r'² sen θ dr' dθ."""
    g = 1 / np.sqrt(1 - beta**2)
    u, wu = leggauss(n); s = 0.5 * (u + 1) / a; ws = 0.5 * wu / a           # s = 1/r' en (0, 1/a)
    v, wv = leggauss(n); th = 0.5 * np.pi * (v + 1); wth = 0.5 * np.pi * wv
    S, TH = np.meshgrid(s, th, indexing="ij"); W = np.outer(ws, wth)
    rp = 1 / S; xi, rho = rp * np.cos(TH), rp * np.sin(TH)
    E, B = campo_carga_movil(xi / g, rho, 0 * rho, beta, q)
    jac = 2 * np.pi / g * rp**2 * np.sin(TH) * rp**2                       # dr' = ds/s²
    energia = np.sum(W * jac * ((E**2).sum(0) + (B**2).sum(0)) / (8 * np.pi))
    momento = np.sum(W * jac * (E[1] * B[2] - E[2] * B[1]) / (4 * np.pi))
    return energia, momento

# Comprobaciones
x = rng.uniform(-3, 3, 50); y = rng.uniform(-3, 3, 50); z = rng.uniform(-3, 3, 50)
err = 0.0
for xi, yi, zi in zip(x, y, z):
    Et, Bt = coulomb_transformado(xi, yi, zi, 0.8)
    Ec, Bc = campo_carga_movil(np.array(xi), np.array(yi), np.array(zi), 0.8)
    err = max(err, np.abs(Et - Ec).max() / np.abs(Ec).max(), np.abs(Bt - Bc).max() / np.abs(Ec).max())
verificar("transformar el campo de Coulomb da el campo de la Clase 24 (Liénard–Wiechert), con B = β × E", err, 0.0, tol=1e-12)
U = 0.5
verificar("β = 0: la energía del campo de la cáscara es U = q²/2a", energia_momento_cascara(0.0)[0], U, tol=1e-8)
for b in (0.3, 0.8):
    g = 1 / np.sqrt(1 - b**2); en, p = energia_momento_cascara(b)
    verificar(f"β = {b}: 𝓔_campo = γU(1 + β²/3)", en, g * U * (1 + b**2 / 3), tol=1e-6)
    verificar(f"β = {b}: cP_campo = (4/3)γβU, no γβU", p, 4 / 3 * g * b * U, tol=1e-6)

# %% [markdown]
# ### ¿Qué pasó?
# Transformar el campo de Coulomb da exactamente el campo de la Clase 24: la componente transversal crece en $\gamma$ y la longitudinal no, y además las distancias en la dirección del movimiento se contraen. El resultado es un campo achatado, más intenso al costado y más débil adelante, y con un campo magnético $\boldsymbol\beta\times\mathbf E$. Ese es el $\gamma$ que apareció en la Clase 24.
#
# La energía y el momento del campo de la cáscara no forman un cuadrivector: para eso harían falta $\mathcal E=\gamma Mc^2$ y $cP=\gamma\beta Mc^2$ con la misma $M$, y en cambio $\mathcal E/\gamma$ depende de $\beta$ y $cP/\gamma\beta=\frac43U$. El campo solo no es un sistema cerrado: la cáscara se sostiene con las tensiones de Poincaré, y la cuenta cierra cuando se las incluye (sección 5 de las notas).

# %% [markdown]
# ## Experimento 3 — Doppler y aberración
#
# **Una onda plana.** En $S$, $E_y=B_z=\cos(kx-\omega t)$, con $\omega=ck$. Un observador se mueve según $+\hat{\mathbf x}$ con $\beta c$. Transformamos los campos en los eventos de su línea de mundo, $x'=0$, y medimos la frecuencia y la amplitud que ve. Las notas (sección 4) dan las dos multiplicadas por $D=\sqrt{\frac{1-\beta}{1+\beta}}$.
#
# **Una fuente que se mueve.** Una fuente emite un frente de onda esférico cada $T_0$ de su tiempo propio mientras se mueve con $\beta c\,\hat{\mathbf x}$. Dibujamos los frentes en un instante.
#
# **El efecto faro.** La fuente emite fotones en todas las direcciones por igual en su sistema. Transformamos sus cuadrimomentos (Clase 29) al laboratorio y miramos hacia dónde van.
#
# ### Predecí
# 1. Si el observador se aleja de la fuente de la onda plana, ¿la ve más débil, más fuerte o igual?
# 2. De los fotones que la fuente emite hacia adelante en su sistema (la mitad), ¿dónde quedan en el laboratorio?

# %%
def onda_vista(beta, ts):
    """E'_y y B'_z de la onda plana sobre la línea de mundo x' = 0 del observador, en sus tiempos t'."""
    g = 1 / np.sqrt(1 - beta**2)
    t = g * ts; x = g * beta * ts                                   # el evento (t', x' = 0) en S
    fase = np.cos(x - t)
    out = np.array([transformar_campos([0, f, 0], [0, 0, f], [beta, 0, 0]) for f in fase])
    return out[:, 0, 1], out[:, 1, 2]

def frentes(ax, beta, T0=1.0, n=8):
    g = 1 / np.sqrt(1 - beta**2)
    te = g * T0 * np.arange(n); xe = beta * te; t_obs = te[-1] + 0.01
    for k in range(n):
        r = t_obs - te[k]
        ax.add_patch(plt.Circle((xe[k], 0), r, fill=False, color=COLORES[0], lw=1.2))
    ax.plot(xe, 0 * xe, ".", color="gray", ms=4)                     # dónde se emitió cada frente
    ax.plot(xe[-1], 0, "o", color="k")
    ax.set(aspect="equal", xlim=(-t_obs - 0.5, t_obs + 0.5), ylim=(-t_obs - 0.5, t_obs + 0.5),
           title=f"frentes de una fuente con β = {beta:g}")
    ax.grid(False)

def fotones(beta, N=20000, semilla=2):
    rng_f = np.random.default_rng(semilla)
    ct = rng_f.uniform(-1, 1, N); ph = rng_f.uniform(0, 2 * np.pi, N); st = np.sqrt(1 - ct**2)
    k = np.stack([np.ones(N), ct, st * np.cos(ph), st * np.sin(ph)])  # (ω/c, k) con |k| = 1, x según la velocidad
    kl = boost([-beta, 0, 0]) @ k                                      # al laboratorio
    return ct, kl[1] / kl[0]                                           # cos θ' en la fuente y cos θ en el laboratorio

def mostrar_doppler(beta=0.6, exportar=False):
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.4))
    ts = np.linspace(0, 20, 2000); Ey, _ = onda_vista(beta, ts)
    axs[0].plot(ts, np.cos(-ts), color="gray", lw=1, label="β = 0")
    axs[0].plot(ts, Ey, color=COLORES[0], label=f"β = {beta:g}")
    axs[0].set(xlabel="t' (tiempo del observador)", ylabel="E'_y", xlim=(0, 20), title="una onda plana vista desde S'")
    axs[0].legend(fontsize=9, loc="lower right")
    frentes(axs[1], beta)
    ct, cl = fotones(beta)
    axs[2].hist(np.degrees(np.arccos(cl)), bins=60, range=(0, 180), color=COLORES[2], alpha=0.8, label="laboratorio")
    axs[2].hist(np.degrees(np.arccos(ct)), bins=60, range=(0, 180), histtype="step", color="k", label="sistema de la fuente")
    axs[2].axvline(np.degrees(np.arccos(beta)), color=COLORES[1], ls="--", lw=1.2, label="cos θ = β")
    axs[2].set(xlabel="ángulo con la velocidad (grados)", ylabel="fotones", title="efecto faro")
    axs[2].legend(fontsize=8)
    fig.tight_layout()
    if exportar:
        guardar(fig, "nb30_doppler")
    plt.show()

interactuar(mostrar_doppler, beta=deslizador("β", 0.6, 0.0, 0.95, 0.05))
if CARPETA_FIGURAS:
    mostrar_doppler(0.6, exportar=True)

# Comprobaciones
b = 0.6; D = np.sqrt((1 - b) / (1 + b))
ts = np.linspace(0, 40 / D, 40001); Ey, Bz = onda_vista(b, ts)
cruces = ts[1:][np.diff(np.sign(Ey)) > 0]
verificar("el observador que se aleja mide la frecuencia ω' = Dω, con D = √((1 − β)/(1 + β))", 2 * np.pi / np.diff(cruces).mean(), D, tol=1e-4)
verificar("y la amplitud del campo E' = D E", np.abs(Ey).max(), D, tol=1e-6)
verificar("en S' sigue siendo una onda plana: E'_y = B'_z", np.abs(Ey - Bz).max(), 0.0, tol=1e-12)
g = 1 / np.sqrt(1 - b**2); T0 = 1.0
te = g * T0 * np.arange(8); t_obs = te[-1] + 0.01
adelante = b * te + (t_obs - te); atras = b * te - (t_obs - te)      # dónde cortan los frentes al eje x
verificar("frentes de la fuente: adelante están separados cT₀ √((1 − β)/(1 + β))", np.abs(np.diff(adelante)).mean(), T0 * D, tol=1e-12)
verificar("y atrás, cT₀ √((1 + β)/(1 − β))", np.abs(np.diff(atras)).mean(), T0 / D, tol=1e-12)
ct, cl = fotones(0.8)
verificar("efecto faro, β = 0.8: los fotones que van hacia adelante en la fuente quedan en cos θ > β", np.mean((cl > 0.8) == (ct > 0)), 1.0, tol=1e-12)

# %% [markdown]
# ### ¿Qué pasó?
# El observador que se aleja de la onda ve una frecuencia menor y también **campos más débiles**, los dos multiplicados por $D=\sqrt{(1-\beta)/(1+\beta)}$: la intensidad cae como $D^2$. Sigue siendo una onda plana, con $E'=B'$. Los frentes de una fuente que se mueve se juntan adelante y se separan atrás. A diferencia del Doppler del sonido, el relativista depende solo de la velocidad relativa: el factor $\gamma$ de la dilatación del tiempo de la fuente compensa exactamente lo que en el sonido depende de quién se mueve respecto del aire.
#
# Los fotones que la fuente emite hacia adelante en su sistema, la mitad, quedan en el laboratorio dentro del cono $\cos\theta>\beta$, de semiángulo $\sim1/\gamma$ para $\beta\to1$: la luz de una fuente rápida sale concentrada hacia adelante. Es el haz angosto de la radiación de sincrotrón de la Clase 24.

# %% [markdown]
# ## Explorá
#
# 1. **Guía 12, problema 5.** Con `transformar_campos`, probá campos $\mathbf E\perp\mathbf B$ con $|\mathbf E|=|\mathbf B|$, campos con $\mathbf E\cdot\mathbf B=0$ y $E\neq B$, y campos cualesquiera, y buscá numéricamente los sistemas que piden los tres incisos. Después demostralo con los invariantes.
# 2. **Guía 12, problema 6.** Para una línea de carga sobre el eje $x$, escribí $\mathbf E$ en $S$, transformalo con `transformar_campos` en eventos de $S'$ (cuidado con las coordenadas: $x=\gamma(x'+\beta ct')$) y comparalo con tus $\mathbf E'$ y $\mathbf B'$ escritos con la carga y la corriente de $S'$.
# 3. **Guía 12, problema 7.** Con `campo_carga_movil`, calculá numéricamente el flujo de $\mathbf E$ por esferas de distintos radios centradas en la carga y el vector de Poynting, y comparalos con tus resultados.
# 4. **Guía 12, problemas 8 y 9.** La onda plana de `onda_vista` va según $x$. Hacé una que vaya en otra dirección, transformala, y compará la frecuencia y la dirección que medís con las que da tu cuadrivector $k^\mu$.
