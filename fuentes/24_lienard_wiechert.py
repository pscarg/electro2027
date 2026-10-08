# %% [markdown]
# # Clase 24 — Liénard–Wiechert: los campos de una carga en movimiento
#
# **Objetivos**
# - Calcular el campo exacto de una carga con cualquier movimiento, resolviendo el tiempo retardado en toda una grilla.
# - Ver el ``quiebre'' de las líneas de campo cuando una carga arranca: el pulso de radiación.
# - Medir el haz de una carga relativista acelerada y la potencia de Liénard.
# - Ver cómo una carga que gira rápido radía en armónicos: el camino a la radiación de sincrotrón.
#
# **Material relacionado:** notas de la Clase 24. Guía 9: problemas 4 y 7.
#
# **Unidades.** Gaussianas, adimensionales, con $c=1$ y $q=1$.

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
from matplotlib import animation

# %% [markdown]
# ## Las herramientas
#
# Una trayectoria es una función que devuelve la posición, la velocidad y la aceleración en el instante $t'$ (arrays con la forma de $t'$, en el plano $xy$). `t_retardado` resuelve $t-t_r=|\mathbf r-\mathbf r_q(t_r)|$ (con $c=1$) en todos los puntos a la vez por bisección: la función $t-t'-|\mathbf r-\mathbf r_q(t')|$ decrece con $t'$ porque $v<c$, así que tiene una sola raíz (sección 1 de las notas). `campos_LW` evalúa los campos de las notas,
# $$\mathbf E=q\left[\frac{(\hat{\mathbf n}-\boldsymbol\beta)(1-\beta^2)}{\kappa^3R^2}+\frac{\hat{\mathbf n}\times\left((\hat{\mathbf n}-\boldsymbol\beta)\times\dot{\boldsymbol\beta}\right)}{c\,\kappa^3R}\right]_{t_r},\qquad\kappa=1-\hat{\mathbf n}\cdot\boldsymbol\beta,\qquad\mathbf B=\hat{\mathbf n}\times\mathbf E .$$

# %%
def t_retardado(X, Y, t, trayectoria, T_max=100.0, iteraciones=70):
    """Tiempo retardado en cada punto (X, Y) del plano z = 0, por bisección (c = 1). t puede ser un número o un array con la forma de X."""
    hi = np.asarray(t, float) + np.zeros(np.shape(X)); lo = hi - T_max
    for _ in range(iteraciones):
        mid = 0.5 * (lo + hi)
        (xq, yq), _, _ = trayectoria(mid)
        f = t - mid - np.hypot(X - xq, Y - yq)
        lo = np.where(f > 0, mid, lo); hi = np.where(f > 0, hi, mid)
    return 0.5 * (lo + hi)

def campos_LW(X, Y, t, trayectoria, separar=False, T_max=100.0):
    """E (Ex, Ey) y B_z de Liénard–Wiechert en el plano z = 0 (q = c = 1). Con separar=True, devuelve también la parte de aceleración."""
    tr = t_retardado(X, Y, t, trayectoria, T_max=T_max)
    (xq, yq), (vx, vy), (ax, ay) = trayectoria(tr)
    Rx, Ry = X - xq, Y - yq; R = np.hypot(Rx, Ry)
    nx, ny = Rx / R, Ry / R
    kappa = 1 - nx * vx - ny * vy
    b2 = vx**2 + vy**2
    Evx = (nx - vx) * (1 - b2) / (kappa**3 * R**2); Evy = (ny - vy) * (1 - b2) / (kappa**3 * R**2)
    # n × ((n − β) × a) = (n − β)(n·a) − a (n·(n − β)) = (n − β)(n·a) − κ a
    na = nx * ax + ny * ay
    Eax = ((nx - vx) * na - kappa * ax) / (kappa**3 * R); Eay = ((ny - vy) * na - kappa * ay) / (kappa**3 * R)
    Ex, Ey = Evx + Eax, Evy + Eay
    Bz = nx * Ey - ny * Ex
    return (Ex, Ey, Bz, (Eax, Eay)) if separar else (Ex, Ey, Bz)

def lineas_axiales(rs, ths, t, trayectoria):
    """Función de flujo Ψ para un movimiento sobre el eje x (Clase 3, con el eje x en lugar de z), en una grilla polar
    (r, θ) centrada en el origen, con 0 ≤ θ ≤ π:  ∂Ψ/∂θ = −E_r r² sinθ.
    Sobre el eje, Ψ = +q delante de la carga y −q detrás; para una carga quieta en el origen, Ψ = q cos θ.
    Las líneas de E son sus curvas de nivel. Devuelve X, Y (mitad superior) y Ψ."""
    Rg, Tg = np.meshgrid(rs, ths, indexing="ij")
    X, Y = Rg * np.cos(Tg), Rg * np.sin(Tg)
    Ex, Ey, _ = campos_LW(X, Y, t, trayectoria)
    Er = Ex * np.cos(Tg) + Ey * np.sin(Tg)
    (xq, _), _, _ = trayectoria(np.array(t, float))
    G = cumulative_trapezoid(Er * Rg**2 * np.sin(Tg), ths, axis=1, initial=0)   # ∫₀^θ
    # Integramos desde el lado del eje opuesto a la carga, para no pasar cerca de ella:
    # si x_q ≥ 0, desde θ = π (donde Ψ = −q): Ψ = −q + ∫_θ^π; si no, desde θ = 0 (donde Ψ = +q).
    Psi = (-1 + G[:, -1:] - G) if xq >= 0 else (1 - G)
    return X, Y, Psi

def arranque(v0, tau):
    """Una carga quieta en el origen que acelera uniformemente sobre el eje x entre 0 y tau, hasta v0, y sigue a v0."""
    a = v0 / tau
    def tray(tp):
        tp = np.asarray(tp, float)
        x = np.where(tp < 0, 0.0, np.where(tp < tau, 0.5 * a * tp**2, 0.5 * a * tau**2 + v0 * (tp - tau)))
        v = np.where(tp < 0, 0.0, np.where(tp < tau, a * tp, v0))
        ac = np.where((tp >= 0) & (tp < tau), a, 0.0)
        z = np.zeros_like(tp)
        return (x, z), (v, z), (ac, z)
    return tray

# %% [markdown]
# ## Experimento 1 ★ — El quiebre de las líneas de campo
#
# Una carga quieta en el origen arranca en $t=0$: acelera uniformemente durante $\tau=0.5$ hasta $v_0$, y después sigue con velocidad constante. Dibujamos las líneas de $\mathbf E$ en el plano que contiene la trayectoria, como curvas de nivel de la función de flujo de la Clase 3 (el movimiento es sobre el eje $x$, así que hay simetría de revolución alrededor de él).
#
# ### Predecí
# En el instante $t=8$, ¿qué campo hay a distancia $r>8$ del origen? ¿Y a $r<7.5$? ¿Qué une a las dos regiones?

# %%
v0, tau = 0.5, 0.5
tray = arranque(v0, tau)
rs = np.linspace(0.05, 17.0, 340); ths = np.linspace(0, np.pi, 721)
niveles = np.linspace(-0.95, 0.95, 20)

def dibujar_quiebre(ax, t):
    X, Y, Psi = lineas_axiales(rs, ths, t, tray)
    (xq, _), _, _ = tray(np.array(t))
    Psi = np.ma.array(Psi, mask=np.hypot(X - xq, Y) < 0.3)
    ax.clear()
    for signo in (1, -1):
        ax.contour(X, signo * Y, Psi, levels=niveles, colors=[COLORES[0]], linewidths=0.9, linestyles="solid")
    ax.plot([xq], [0], "o", color=COLORES[1])
    for rad in (t, t - tau):
        if rad > 0:
            ax.add_patch(plt.Circle((0, 0), rad, fill=False, ls=":", color="0.6", lw=0.8))
    ax.set(aspect="equal", xlim=(-12, 12), ylim=(-12, 12), xlabel="x", ylabel="y", title=f"t = {t:.1f},  v₀ = {v0} c")
    ax.grid(False)

fig, ax = plt.subplots(figsize=(6.2, 6.2))
tiempos = np.linspace(1.0, 11.0, 3 if PRUEBA else 26)
anim = animation.FuncAnimation(fig, lambda n: dibujar_quiebre(ax, tiempos[n]), frames=len(tiempos), interval=150)
plt.close(fig)
display(HTML(anim.to_jshtml()))
fig, ax = plt.subplots(figsize=(6.2, 6.2)); dibujar_quiebre(ax, 8.0); guardar(fig, "nb24_quiebre"); plt.close(fig)

# Comprobaciones
t = 8.0
Xc = np.array([3.0, -2.0, 9.5]); Yc = np.array([1.0, 4.0, 2.0])
tr = t_retardado(Xc, Yc, t, tray)
(xq, yq), _, _ = tray(tr)
verificar("el tiempo retardado cumple t − t_r = |r − r_q(t_r)|", np.max(np.abs(t - tr - np.hypot(Xc - xq, Yc - yq))), 0.0, tol=1e-10)

def flujo_esfera(Rs, t, tray, nth=2001):
    """Flujo de E a través de una esfera de radio Rs centrada en el origen (simetría de revolución alrededor de x)."""
    th = np.linspace(1e-6, np.pi - 1e-6, nth)
    X, Y = Rs * np.cos(th), Rs * np.sin(th)
    Ex, Ey, _ = campos_LW(X, Y, t, tray)
    Er = Ex * np.cos(th) + Ey * np.sin(th)
    return trapezoid(Er * 2 * np.pi * Rs**2 * np.sin(th), th)

verificar("Gauss con la carga adentro y el quiebre afuera (esfera r = 5)", flujo_esfera(5.0, t, tray), 4 * np.pi, tol=1e-4)
verificar("Gauss con la esfera en medio del quiebre (r = 7.75)", flujo_esfera(7.75, t, tray), 4 * np.pi, tol=1e-4)
# Adentro: el campo de una carga con velocidad constante, que apunta desde la posición presente (sección 3)
P = (np.array([2.0]), np.array([3.0]))
Ex, Ey, _ = campos_LW(*P, t, tray)
(xp, _), _, _ = tray(np.array(t))
Rpx, Rpy = P[0] - xp, P[1]; Rp = np.hypot(Rpx, Rpy); sin2 = (Rpy / Rp)**2
fac = (1 - v0**2) / (Rp**3 * (1 - v0**2 * sin2)**1.5)
verificar("adentro: E = q(1−β²) R_p / [R_p³(1−β² sin²ψ)^{3/2}] desde la posición presente", np.hypot(Ex[0] - fac[0] * Rpx[0], Ey[0] - fac[0] * Rpy[0]) / (fac[0] * Rp[0]), 0.0, tol=1e-8)
# El quiebre para v₀ ≪ c y lejos (cτ ≪ R): E transversal en la cáscara = q a sinθ / c²R (sección 4)
v0c, tauc, tk = 0.02, 0.5, 40.0; trayc = arranque(v0c, tauc)
Rm = tk - tauc / 2
Ex, Ey, _ = campos_LW(np.array([0.0]), np.array([Rm]), tk, trayc)
verificar("quiebre (v₀ ≪ c, cτ ≪ R): E_θ = q a sinθ/c²R a 90°", abs(Ex[0]), (v0c / tauc) / Rm, tol=2e-2)

# %% [markdown]
# ### ¿Qué pasó?
# Afuera de la esfera de radio $ct$, la noticia del arranque no llegó: las líneas son las de una carga quieta en el origen. Adentro de la esfera de radio $c(t-\tau)$, las líneas son las de una carga con velocidad constante y salen de la posición **presente** de la carga, aunque la información venga del pasado; con $v_0=0.5c$ ya se ve que se aprietan hacia el plano perpendicular a la velocidad. En la cáscara entre las dos esferas, de espesor $c\tau$, las líneas se quiebran para empalmar: ese tramo casi transversal, que se aleja a velocidad $c$, es el pulso de radiación. El flujo de $\mathbf E$ por cualquier esfera que rodee la carga sigue siendo $4\pi q$, también en medio del quiebre.
#
# ## Experimento 2 — El haz de una carga relativista
#
# Para una aceleración paralela a la velocidad, las notas dan $\frac{dP}{d\Omega}\propto\frac{\sin^2\theta}{(1-\beta\cos\theta)^5}$ (potencia por unidad de tiempo de la carga), con el máximo en $\cos\theta_{\max}=\frac{\sqrt{1+15\beta^2}-1}{3\beta}$, $\theta_{\max}\simeq\frac{1}{2\gamma}$ para $\gamma\gg1$, y la potencia total $\frac{2q^2a^2}{3c^3}\gamma^6$. Para una órbita circular ($\dot{\boldsymbol\beta}\perp\boldsymbol\beta$), $\frac{2q^2a^2}{3c^3}\gamma^4$.
#
# ### Predecí
# Con $\beta=0.9$, ¿hacia dónde sale la radiación? El diagrama no relativista ($\sin^2\theta$) es nulo justo hacia adelante.

# %%
def dPdOmega_rel(beta_v, adot, n):
    """dP/dΩ por unidad de tiempo de la carga (q = c = 1): |n × ((n − β) × β̇)|²/(4π κ⁵), n con forma (..., 3)."""
    beta_v, adot = np.asarray(beta_v, float), np.asarray(adot, float)
    nb = n - beta_v
    u = np.cross(n, np.cross(nb, adot))
    kappa = 1 - n @ beta_v
    return np.sum(u**2, axis=-1) / (4 * np.pi * kappa**5)

def esfera(nth=1201, nph=240):
    th = np.linspace(0, np.pi, nth); ph = np.linspace(0, 2 * np.pi, nph, endpoint=False)
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    return th, np.stack([np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH), np.cos(TH)], axis=-1)

def P_total(beta_v, adot, nth=1201, nph=240):
    th, n = esfera(nth, nph)
    dP = dPdOmega_rel(beta_v, adot, n)
    return trapezoid(np.mean(dP, axis=1) * 2 * np.pi * np.sin(th), th)

def mostrar_haz(beta=0.5):
    th = np.linspace(-np.pi, np.pi, 2001)
    n = np.stack([np.sin(th), 0 * th, np.cos(th)], axis=-1)
    dP = dPdOmega_rel([0, 0, beta], [0, 0, 1], n)
    fig = plt.figure(figsize=(6, 5.2))
    ax = fig.add_subplot(1, 1, 1, projection="polar")
    ax.plot(th, dP / dP.max(), color=COLORES[0])
    ax.set_theta_zero_location("N"); ax.set_title(f"aceleración ∥ velocidad (hacia arriba), β = {beta:g}")
    plt.show()

interactuar(mostrar_haz, beta=deslizador("β", 0.5, 0.0, 0.99, 0.01))

for b in (0.5, 0.9):
    th = np.linspace(1e-4, np.pi / 2, 400001)
    dP = dPdOmega_rel([0, 0, b], [0, 0, 1], np.stack([np.sin(th), 0 * th, np.cos(th)], axis=-1))
    verificar(f"β = {b}: cos θ_max = (√(1+15β²) − 1)/3β", np.cos(th[np.argmax(dP)]), (np.sqrt(1 + 15 * b**2) - 1) / (3 * b), tol=1e-5)
g = 10.0; b = np.sqrt(1 - 1 / g**2)
th = np.linspace(1e-5, 0.3, 300001)
dP = dPdOmega_rel([0, 0, b], [0, 0, 1], np.stack([np.sin(th), 0 * th, np.cos(th)], axis=-1))
verificar("γ = 10: θ_max ≃ 1/2γ", th[np.argmax(dP)], 1 / (2 * g), tol=2e-2)
b = 0.6; gam = 1 / np.sqrt(1 - b**2)
verificar("lineal: P = (2/3) q²a² γ⁶/c³", P_total([0, 0, b], [0, 0, 1.0]), 2 / 3 * gam**6, tol=1e-4)
verificar("circular: P = (2/3) q²a² γ⁴/c³", P_total([0, 0, b], [1.0, 0, 0]), 2 / 3 * gam**4, tol=1e-4)

# %% [markdown]
# ### ¿Qué pasó?
# Para $\beta\ll1$ el diagrama es el $\sin^2\theta$ del dipolo, simétrico entre adelante y atrás. Al crecer $\beta$, el factor $(1-\beta\cos\theta)^{-5}$ inclina los lóbulos hacia adelante, y para $\beta\to1$ la radiación sale en un cono angosto de ángulo $\sim1/2\gamma$ alrededor de la velocidad (aunque justo en $\theta=0$ sigue siendo nula). La potencia total crece como $\gamma^6$ para una aceleración paralela y como $\gamma^4$ para una circular: por eso los aceleradores circulares de electrones pierden tanta energía por radiación (radiación de sincrotrón).
#
# ## Experimento 3 — Una carga que gira rápido radía armónicos
#
# Una carga gira en una circunferencia de radio $A$ con velocidad angular $\omega=1$, así que $\beta=A\omega/c$. Un observador lejano en el plano de la órbita (sobre el eje $x$) recibe el campo $E_\perp(t)$, que calculamos con Liénard–Wiechert y descomponemos con Fourier (Clase 21). Para $\beta\ll1$ es la radiación de un dipolo que gira (Clase 22), a la frecuencia $\omega$. Las notas (sección 6) dan el campo exacto en el plano,
# $$rE_\perp=\frac{q\beta\omega}{c}\,\frac{\sin\omega t_r+\beta}{(1+\beta\sin\omega t_r)^3},\qquad t-\frac rc=t_r-\frac Ac\cos\omega t_r .$$
#
# ### Predecí
# Si $\beta$ crece, ¿el campo que llega sigue siendo una sinusoide? ¿De qué parte de la órbita llega la mayor parte?

# %%
def circular(A, w=1.0):
    """Una carga que gira en una circunferencia de radio A centrada en el origen, con velocidad angular w (β = A w)."""
    def tray(tp):
        tp = np.asarray(tp, float)
        c, s = np.cos(w * tp), np.sin(w * tp)
        return (A * c, A * s), (-A * w * s, A * w * c), (-A * w**2 * c, -A * w**2 * s)
    return tray

def campo_lejano(tray, theta, Rlejos=1e4, periodos=4, puntos=512):
    """R E⊥ en un punto lejano del plano, en la dirección θ (medida desde el eje x), como función del tiempo
    del observador menos R/c, para un movimiento de período 2π."""
    t = Rlejos + np.linspace(0, 2 * np.pi * periodos, periodos * puntos, endpoint=False)
    X = np.full_like(t, Rlejos * np.cos(theta)); Y = np.full_like(t, Rlejos * np.sin(theta))
    Ex, Ey, _ = campos_LW(X, Y, t, tray, T_max=2 * Rlejos)
    return t - Rlejos, (-Ex * np.sin(theta) + Ey * np.cos(theta)) * Rlejos

def armonicos(tray, theta, n_arm=6, periodos=4, puntos=512):
    """Amplitudes de los armónicos n = 1 … n_arm del campo lejano (movimiento de período 2π), y el campo."""
    t, E = campo_lejano(tray, theta, periodos=periodos, puntos=puntos)
    F = np.abs(np.fft.rfft(E)) / len(E) * 2
    return F[periodos * np.arange(1, n_arm + 1)], t, E

def mostrar_armonicos(beta=0.3):
    amp, t, E = armonicos(circular(beta), 0.0, n_arm=40, periodos=2, puntos=1024)
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.8))
    axs[0].plot(t, E / np.abs(E).max(), color=COLORES[0])
    axs[0].set(xlabel="ωt (tiempo del observador, menos R/c)", ylabel="R E⊥ (normalizado)", title=f"campo radiado en el plano, β = {beta:g}")
    axs[1].bar(np.arange(1, len(amp) + 1), amp / amp.max(), color=COLORES[1])
    axs[1].set(yscale="log", ylim=(1e-6, 2), xticks=[1, 10, 20, 30, 40], xlabel="armónico n (frecuencia nω)", ylabel="amplitud / la mayor", title="espectro")
    plt.tight_layout(); plt.show()

interactuar(mostrar_armonicos, beta=deslizador("β", 0.3, 0.01, 0.95, 0.01))

b = 0.9; trc = circular(b)
t, E = campo_lejano(trc, 0.0)
s = np.sin(t_retardado(np.full_like(t, 1e4), 0 * t, t + 1e4, trc, T_max=2e4))
verificar("β = 0.9, en el plano: R E⊥ = qβω(sin ωt_r + β)/c(1 + β sin ωt_r)³", np.max(np.abs(E - b * (s + b) / (1 + b * s)**3)) / np.abs(E).max(), 0.0, tol=1e-3)
amp = armonicos(circular(0.004), 0.0)[0]
verificar("β ≪ 1: primer armónico = q A ω²/c² (dipolo que gira, Clase 22)", amp[0], 0.004, tol=1e-3)
verificar("β ≪ 1: segundo armónico / primero = 2β (cuadrupolo, Clase 23)", amp[1] / amp[0], 2 * 0.004, tol=1e-2)

# %% [markdown]
# ### ¿Qué pasó?
# Para $\beta\ll1$ el campo es una sinusoide de frecuencia $\omega$, con la amplitud del dipolo que gira de la Clase 22. Al crecer $\beta$ se deforma: el tramo de la órbita en que la carga se acerca al observador ($\sin\omega t_r=-1$) llega reforzado por $1/\kappa^3$ y comprimido en el tiempo del observador ($dt=\kappa\,dt_r$), y el tramo en que se aleja llega debilitado y estirado. Aparecen armónicos $2\omega,3\omega,\ldots$; el segundo vale $2\beta$ veces el primero, la radiación cuadrupolar de la Clase 23, que también tenía frecuencia $2\omega$. Para $\beta\to1$ el observador recibe un destello por vuelta, como de un faro, de duración $\sim1/\gamma^3\omega$, y el espectro llega hasta armónicos $n\sim\gamma^3$: es la radiación de sincrotrón.
#
# ## Explorá
#
# 1. **Guía 9, P7.** Armá con `arranque` como modelo una trayectoria que **frena**: una carga que viene con velocidad $v_0$ y se detiene en un tiempo $\tau$. Dibujá las líneas con `lineas_axiales`, medí el campo transversal en la cáscara y comparalo con tu resultado no relativista. Después probá con $v_0$ cerca de $c$ y buscá el ángulo del cono.
# 2. **Guía 9, P4.** Escribí, como `circular`, una trayectoria que oscile sobre el eje $x$, $x=A\cos\omega t$. Con `armonicos(tray, theta)` medí la razón entre el segundo y el primer armónico en función de $\theta$ y de $\beta_{\max}=A\omega/c$, y comparala con lo que te dio el desarrollo multipolar.
# 3. **Radiación de sincrotrón.** Para $\beta=0.8,\,0.9,\,0.95,\,0.98$, buscá con `armonicos(circular(beta), 0.0, n_arm=400, puntos=4096)` el armónico de mayor amplitud. ¿Crece como $\gamma^3$ (sección 6 de las notas)?
# 4. **El campo ``aplastado''.** Con `lineas_axiales` (en una grilla polar) y una carga que se mueve siempre con $\beta=0.9$, dibujá las líneas y medí cuánto mayor es el campo perpendicular a la velocidad que el paralelo, a la misma distancia de la posición presente. Compará con la sección 3 de las notas.
