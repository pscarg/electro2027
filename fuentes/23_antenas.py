# %% [markdown]
# # Clase 23 — Antenas, dipolo magnético, cuadrupolo y dispersión de Rayleigh
#
# **Objetivos**
# - Ver cómo cambia el diagrama de una antena cuando su largo se acerca a la longitud de onda, y calcular su resistencia de radiación.
# - Dirigir el haz de un arreglo de antenas cambiando las fases.
# - Comparar los diagramas dipolar eléctrico, dipolar magnético y cuadrupolar, y comprobar que sus potencias se suman.
# - Entender por qué el cielo es azul y su luz está polarizada.
#
# **Material relacionado:** notas de la Clase 23. Guía 9: problemas 4 a 6.
#
# **Unidades.** Gaussianas, con $c=1$. Para las antenas, las longitudes van en unidades del largo $d$ y las corrientes en unidades de $I_0$; la resistencia en unidades de $1/c$, que equivale a $30\ \Omega$.

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
from scipy.integrate import trapezoid

OHM_POR_1_SOBRE_C = 29.98      # 1/c (en s/cm) equivale a 29.98 Ω

# %% [markdown]
# ## Experimento 1 ★ — El diagrama de una antena según su largo
#
# Para una antena recta de largo $d$ según $\hat{\mathbf z}$ con corriente $I(z)e^{-i\omega t}$, las notas (sección 4) dan, en la zona de radiación,
# $$\frac{dP}{d\Omega}=\frac{k^2\sin^2\theta}{8\pi c}\left|F(\theta)\right|^2,\qquad F(\theta)=\int_{-d/2}^{d/2}I(z')\,e^{-ikz'\cos\theta}\,dz' ,$$
# y la resistencia de radiación $R=2P/I_{\text{alim}}^2$. Para una corriente uniforme, $F=I_0d\,\frac{\sin u}{u}$ con $u=\frac{kd}{2}\cos\theta$. La función `F_antena` calcula $F$ **numéricamente** para cualquier $I(z)$.
#
# ### Predecí
# Si la antena es mucho más larga que la longitud de onda, ¿hacia dónde se concentra la radiación? ¿Qué aparece entre medio?

# %%
def F_antena(I_de_z, kd, cos_th, n=1001):
    """F(θ) = ∫ I(z') e^{−ik z' cosθ} dz' sobre |z'| ≤ 1/2 (longitudes en unidades de d, k = kd)."""
    z = np.linspace(-0.5, 0.5, n)
    fase = np.exp(-1j * kd * np.outer(np.atleast_1d(cos_th), z))
    return trapezoid(I_de_z(z)[None, :] * fase, z, axis=1)

def patron_antena(I_de_z, kd, nth=361):
    """θ y dP/dΩ (en unidades de I0²/c) de una antena lineal."""
    th = np.linspace(0, np.pi, nth)
    F = F_antena(I_de_z, kd, np.cos(th))
    return th, kd**2 * np.sin(th)**2 * np.abs(F)**2 / (8 * np.pi)

def resistencia(I_de_z, kd, I_alim=1.0):
    """R = 2P/I_alim², en Ω."""
    th, dP = patron_antena(I_de_z, kd)
    P = trapezoid(dP * 2 * np.pi * np.sin(th), th)
    return 2 * P / I_alim**2 * OHM_POR_1_SOBRE_C

uniforme = lambda z: np.ones_like(z)
triangular = lambda z: 1 - 2 * np.abs(z)

def mostrar_antena(d_sobre_lambda=0.5):
    kd = 2 * np.pi * d_sobre_lambda
    th, dP = patron_antena(uniforme, kd)
    fig = plt.figure(figsize=(12, 4.6))
    ax = fig.add_subplot(1, 2, 1, projection="polar")
    ax.plot(th, dP / dP.max(), color=COLORES[0]); ax.plot(-th, dP / dP.max(), color=COLORES[0])
    ax.set_theta_zero_location("N"); ax.set_title(f"corriente uniforme, d = {d_sobre_lambda:g} λ  (eje z hacia arriba)")
    ax2 = fig.add_subplot(1, 2, 2)
    ds = np.linspace(0.02, 3, 60)
    ax2.plot(ds, [resistencia(uniforme, 2 * np.pi * x) for x in ds], color=COLORES[0], label="uniforme")
    ax2.plot(ds, 80 * np.pi**2 * ds**2, "k--", lw=1, label="80π²(d/λ)²: dipolo corto")
    ax2.plot(d_sobre_lambda, resistencia(uniforme, kd), "o", color=COLORES[1])
    ax2.set(xlabel="d/λ", ylabel="R (Ω)", ylim=(0, 1500), title="resistencia de radiación"); ax2.legend(fontsize=9)
    plt.tight_layout(); plt.show()

interactuar(mostrar_antena, d_sobre_lambda=deslizador("d/λ", 0.5, 0.05, 4.0, 0.05))

cos_th = np.linspace(-1, 1, 7); kd = 9.0
u = kd / 2 * cos_th
verificar("F numérico = d sin(u)/u (corriente uniforme)", np.max(np.abs(F_antena(uniforme, kd, cos_th, n=8001) - np.sinc(u / np.pi))), 0.0, tol=1e-6)
verificar("antena corta uniforme: R = (2/3)(kd)²/c ≈ 80π²(d/λ)² Ω", resistencia(uniforme, 0.01), 2 / 3 * 0.01**2 * OHM_POR_1_SOBRE_C, tol=1e-4)
verificar("antena corta alimentada en el centro: R = (kd)²/6c ≈ 20π²(d/λ)² Ω", resistencia(triangular, 0.01), 0.01**2 / 6 * OHM_POR_1_SOBRE_C, tol=1e-4)
th, dP = patron_antena(uniforme, 2 * np.pi * 3)
verificar("d = 3λ: el máximo está en θ = 90° (perpendicular a la antena)", np.degrees(th[np.argmax(dP)]), 90.0, tol=1e-6)

# %% [markdown]
# ### ¿Qué pasó?
# Mientras $d\ll\lambda$, el diagrama es el $\sin^2\theta$ del dipolo, y $R$ crece como $(d/\lambda)^2$: una antena corta radía muy poco para la corriente que lleva. Cuando $d$ pasa de $\lambda$, aparecen lóbulos: las contribuciones de los distintos tramos llegan con fases distintas, y se anulan en las direcciones con $\frac{kd}{2}\cos\theta=n\pi$, es decir $\cos\theta=n\lambda/d$. El lóbulo principal queda perpendicular a la antena y se angosta como $\lambda/d$: una antena larga es direccional. (Una corriente uniforme a lo largo de varias longitudes de onda se logra con una fila de antenas cortas alimentadas en fase: el Experimento 2.)
#
# ## Experimento 2 — Un arreglo de antenas en fase
#
# $N$ dipolos cortos verticales en $x_n=na$ ($n=0,\ldots,N-1$), alimentados con la misma amplitud y fases $e^{-in\delta}$. En el plano horizontal, en la dirección de azimut $\varphi$, la onda de cada uno llega con la fase extra $e^{-ikx_n\cos\varphi}$ (sección 4 de las notas), y el campo total es el de uno multiplicado por $\sum_ne^{-in\psi}$ con $\psi=ka\cos\varphi+\delta$. Es una suma geométrica:
# $$\left|\sum_{n=0}^{N-1}e^{-in\psi}\right|^2=\left|\frac{1-e^{-iN\psi}}{1-e^{-i\psi}}\right|^2=\frac{\sin^2(N\psi/2)}{\sin^2(\psi/2)},$$
# con máximo $N^2$ en $\psi=0$: el haz apunta a $\cos\varphi_0=-\delta/ka$. **Cambiando las fases se dirige el haz sin mover nada.**
#
# ### Predecí
# Con 8 antenas separadas $\lambda/2$, ¿qué fase $\delta$ entre vecinas hace que el haz apunte a $60^\circ$ del eje del arreglo?

# %%
def factor_arreglo(N, ka, delta, phi):
    psi = ka * np.cos(phi) + delta
    return np.abs(np.sum(np.exp(-1j * np.outer(psi, np.arange(N))), axis=1))**2

def mostrar_arreglo(N=8, a_sobre_lambda=0.5, delta_grados=0.0):
    N = int(N); ka = 2 * np.pi * a_sobre_lambda
    phi = np.linspace(0, 2 * np.pi, 1441)
    AF = factor_arreglo(N, ka, np.radians(delta_grados), phi)
    fig = plt.figure(figsize=(6, 5.2))
    ax = fig.add_subplot(1, 1, 1, projection="polar")
    ax.plot(phi, AF / N**2, color=COLORES[0])
    ax.set_title(f"{N} antenas, a = {a_sobre_lambda:g} λ, δ = {delta_grados:g}°  (eje del arreglo: φ = 0)")
    plt.show()

interactuar(mostrar_arreglo, N=deslizador("N", 8, 2, 20, 1), a_sobre_lambda=deslizador("a/λ", 0.5, 0.1, 1.5, 0.05),
            delta_grados=deslizador("δ (°)", 0.0, -180.0, 180.0, 5.0))

N, ka = 8, np.pi
delta = -ka * np.cos(np.radians(60))
phi = np.linspace(0, np.pi, 180001)
AF = factor_arreglo(N, ka, delta, phi)
verificar("el haz apunta donde ψ = 0: φ₀ = 60°", np.degrees(phi[np.argmax(AF)]), 60.0, tol=1e-4)
verificar("en el máximo, la intensidad es N² veces la de una antena", AF.max(), N**2, tol=1e-6)
psi = np.linspace(0, 2 * np.pi, 20001)
verificar("promedio en ψ del factor = N (las intensidades se suman en promedio)", trapezoid(np.sin(N * psi / 2)**2 / np.maximum(np.sin(psi / 2)**2, 1e-30), psi) / (2 * np.pi), N, tol=1e-3)

# %% [markdown]
# ### ¿Qué pasó?
# Con $\delta=0$ el haz sale perpendicular al arreglo (en $\varphi=90^\circ$), y su ancho es del orden de $\lambda/Na$. Con $\delta=-ka\cos\varphi_0$ apunta a $\varphi_0$. En el máximo la intensidad es $N^2$ veces la de una sola antena, pero en promedio es $N$ veces: la energía no aparece de la nada, se redistribuye. Si $a>\lambda/2$, aparecen otros máximos iguales (lóbulos de rejilla), como en una red de difracción.
#
# ## Experimento 3 — Dipolo eléctrico, dipolo magnético y cuadrupolo
#
# Con $c=1$ y la distancia $r$ factorizada, los campos de radiación de las notas son
# $$r\mathbf E_{E1}=-\ddot{\mathbf p}_\perp,\qquad r\mathbf E_{M1}=\hat{\mathbf r}\times\ddot{\mathbf m},\qquad r\mathbf E_{E2}=-\tfrac16\left[\dddot{\mathsf Q}\hat{\mathbf r}-\hat{\mathbf r}\,(\hat{\mathbf r}\cdot\dddot{\mathsf Q}\hat{\mathbf r})\right],$$
# y $\frac{dP}{d\Omega}=\frac{|r\mathbf E|^2}{4\pi}$. Las notas dan $P_{M1}=\frac23|\ddot{\mathbf m}|^2$ y $P_{E2}=\frac{1}{180}\dddot Q_{ij}\dddot Q_{ij}$, y muestran que, aunque los campos interfieren en cada dirección, **las potencias totales se suman**.
#
# ### Predecí
# Un cuadrupolo con simetría axial que oscila: ¿en qué dirección radía más?

# %%
def direcciones(nth=181, nph=360):
    th = np.linspace(0, np.pi, nth); ph = np.linspace(0, 2 * np.pi, nph, endpoint=False)
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    n = np.stack([np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH), np.cos(TH)], axis=-1)
    return th, ph, n

def campo_multipolar(n, pdd=(0, 0, 0), mdd=(0, 0, 0), Qddd=np.zeros((3, 3))):
    """r E de radiación (c = 1) en las direcciones n, para p̈, m̈ y Q⃛ dados."""
    pdd, mdd, Qddd = np.asarray(pdd, float), np.asarray(mdd, float), np.asarray(Qddd, float)
    E1 = -(pdd - (n @ pdd)[..., None] * n)
    M1 = np.cross(n, mdd)
    Qn = n @ Qddd.T
    E2 = -(Qn - np.sum(n * Qn, axis=-1)[..., None] * n) / 6
    return E1 + M1 + E2

th, ph, n = direcciones()
def P_num(**kw):
    E = campo_multipolar(n, **kw)
    dP = np.sum(E**2, axis=-1) / (4 * np.pi)
    return trapezoid(np.mean(dP, axis=1) * 2 * np.pi * np.sin(th), th)

Qax = np.diag([-0.5, -0.5, 1.0])
fig, axs = plt.subplots(1, 3, figsize=(13, 4.2), subplot_kw={"projection": "polar"})
corte = np.stack([np.sin(th), 0 * th, np.cos(th)], axis=-1)
for ax, kw, tit in zip(axs, ({"pdd": (0, 0, 1)}, {"mdd": (0, 0, 1)}, {"Qddd": Qax}), ("E1: p̈ ∥ z", "M1: m̈ ∥ z", "E2 axial")):
    dP = np.sum(campo_multipolar(corte, **kw)**2, axis=-1)
    for s in (1, -1):
        ax.plot(s * th, dP / dP.max(), color=COLORES[0])
    ax.set_theta_zero_location("N"); ax.set_title(tit)
plt.tight_layout(); plt.show()

rng = np.random.default_rng(1)
A = rng.normal(size=(3, 3)); Qr = A + A.T; Qr -= np.trace(Qr) / 3 * np.eye(3)
pr, mr = rng.normal(size=3), rng.normal(size=3)
verificar("P_M1 = 2|m̈|²/3c³", P_num(mdd=mr), 2 * mr @ mr / 3, tol=1e-4)
verificar("P_E2 = Q⃛ᵢⱼQ⃛ᵢⱼ/180c⁵ (Q⃛ al azar, simétrico y sin traza)", P_num(Qddd=Qr), np.sum(Qr**2) / 180, tol=1e-4)
verificar("las potencias se suman: P(E1+M1+E2) = P_E1 + P_M1 + P_E2", P_num(pdd=pr, mdd=mr, Qddd=Qr),
          2 * pr @ pr / 3 + 2 * mr @ mr / 3 + np.sum(Qr**2) / 180, tol=1e-4)
dPax = np.sum(campo_multipolar(corte, Qddd=Qax)**2, axis=-1)
verificar("E2 axial: el máximo está en θ = 45°", np.degrees(th[np.argmax(dPax)]), 45.0, tol=1e-6)

# El oscilador cuadrupolar lineal: cargas q en z = ±A cos ωt (p = 0). Las notas dan ⟨P⟩ = 16q²A⁴ω⁶/15c⁵, a frecuencia 2ω.
q, Aamp, w = 1.0, 0.1, 1.0
t = np.linspace(0, 2 * np.pi / w, 4000, endpoint=False); dt = t[1] - t[0]
z = Aamp * np.cos(w * t)
Qzz = 2 * q * (2 * z**2)                       # Σ q(3z² − z²) para las dos cargas
deriv = lambda f: (np.roll(f, -1) - np.roll(f, 1)) / (2 * dt)      # derivada centrada, periódica
Q3 = deriv(deriv(deriv(Qzz)))
Pt = (Q3**2 * (1 + 0.25 + 0.25)) / 180         # Q⃛ᵢⱼQ⃛ᵢⱼ = (3/2) Q⃛_zz²
verificar("oscilador cuadrupolar: ⟨P⟩ = 16q²A⁴ω⁶/15c⁵", np.mean(Pt), 16 * q**2 * Aamp**4 * w**6 / 15, tol=1e-4)

# %% [markdown]
# ### ¿Qué pasó?
# El dipolo magnético tiene el mismo diagrama que el eléctrico, $\sin^2\theta$ (los campos intercambian papeles: $\mathbf E\to\mathbf B$, $\mathbf B\to-\mathbf E$). El cuadrupolo axial radía en cuatro lóbulos, máximos a $45^\circ$ y nulos sobre el eje y en el plano ecuatorial: es la dependencia $\sin^2\theta\cos^2\theta$, la de $l=2$. Con los tres juntos, el diagrama cambia (interfieren), pero la potencia total es la suma de las tres. El oscilador cuadrupolar radía al doble de la frecuencia de las cargas, porque $Q\propto\cos^2\omega t$.
#
# ## Experimento 4 — El cielo azul
#
# Una molécula en la luz del Sol se polariza, $\mathbf p=\alpha\mathbf E$, y radía como un dipolo: la sección eficaz es $\sigma=\frac{8\pi}{3}k^4\alpha^2$ (sección 5 de las notas). Multiplicamos el espectro del Sol (un cuerpo negro a $5800$ K, en función de la longitud de onda) por $k^4\propto\lambda^{-4}$.
#
# ### Predecí
# ¿Dónde queda el máximo del espectro de la luz dispersada? ¿Y cómo está polarizada la luz del cielo a $90^\circ$ del Sol?

# %%
lam = np.linspace(300, 800, 501) * 1e-7                       # cm
h, kB, cc = 6.626e-27, 1.381e-16, 2.998e10
planck = lambda l, T: 1 / l**5 / (np.exp(h * cc / (l * kB * T)) - 1)
sol = planck(lam, 5800); disp = sol / lam**4
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
axs[0].plot(lam * 1e7, sol / sol.max(), color=COLORES[3], label="Sol (5800 K)")
axs[0].plot(lam * 1e7, disp / disp.max(), color=COLORES[0], label="dispersada ∝ λ⁻⁴ × Sol")
axs[0].axvspan(380, 750, color="0.92"); axs[0].set(xlabel="λ (nm)", title="el espectro de la luz del cielo"); axs[0].legend(fontsize=9)
Th = np.linspace(0, np.pi, 361)
I_perp, I_par = np.ones_like(Th), np.cos(Th)**2           # p ⟂ al plano de dispersión, y en el plano
axs[1].plot(np.degrees(Th), (I_perp + I_par) / 2, color=COLORES[0], label="intensidad (luz no polarizada)")
axs[1].plot(np.degrees(Th), (I_perp - I_par) / (I_perp + I_par), color=COLORES[1], label="grado de polarización")
axs[1].set(xlabel="ángulo de dispersión (°)", title="polarización de la luz dispersada"); axs[1].legend(fontsize=9)
plt.tight_layout(); plt.show()

# Comprobaciones
kk, alfa = 2 * np.pi / 500e-7, 1.74e-24
th1 = np.linspace(0, np.pi, 20001)
sigma_num = trapezoid(kk**4 * alfa**2 * np.sin(th1)**2 * 2 * np.pi * np.sin(th1), th1)
verificar("σ = ∫ k⁴α² sin²α dΩ = (8π/3)k⁴α²", sigma_num, 8 * np.pi / 3 * kk**4 * alfa**2, tol=1e-6)
re = (4.803e-10)**2 / (9.109e-28 * cc**2)
verificar("Thomson: σ_T = (8π/3) r_e² = 6.65×10⁻²⁵ cm²", 8 * np.pi / 3 * re**2, 6.652e-25, tol=1e-3)
verificar("a 90°, la luz dispersada está totalmente polarizada", ((I_perp - I_par) / (I_perp + I_par))[180], 1.0, tol=1e-12)
L_aten = 1 / (2.69e19 * sigma_num)
print(f"longitud de atenuación de la luz de 500 nm en aire: {L_aten / 1e5:.0f} km")

# %% [markdown]
# ### ¿Qué pasó?
# El espectro dispersado tiene su máximo en el violeta y el azul, aunque el Sol emite más en el verde: el factor $\lambda^{-4}$ favorece a las longitudes de onda cortas (entre $450$ y $700$ nm, por $(700/450)^4\simeq6$). Que el cielo se vea azul y no violeta tiene que ver con el ojo y con la absorción del violeta en la alta atmósfera. A $90^\circ$ del Sol la luz dispersada está totalmente polarizada: el dipolo inducido no radía en su propia dirección. Con la polarizabilidad del nitrógeno, la luz de $500$ nm recorre unos $60$ km de aire antes de dispersarse.
#
# ## Explorá
#
# 1. **Guía 9, P4.** `momentos(t, trayectorias, cargas)`, en la celda de abajo, calcula $\mathbf p(t)$, $\mathbf m(t)$ y $Q_{ij}(t)$ de cargas puntuales con trayectorias dadas, y sus derivadas. Usalo con $z=A\cos\omega t$ para ver qué términos se anulan, y `campo_multipolar` para comparar sus diagramas.
# 2. **Guía 9, P5.** `patron_fuente(puntos, J, k)` calcula $\frac{dP}{d\Omega}$ en la zona de radiación para cualquier conjunto de elementos de corriente $\mathbf J_n\,dV$ en posiciones $\mathbf r_n$ (sin desarrollo multipolar). Armá la placa que oscila como una grilla de elementos con $\mathbf J=\sigma_0\dot z_0\hat{\mathbf z}$, compará con la aproximación dipolar y buscá los ceros fuera del eje.
# 3. **Guía 9, P6.** Con `patron_antena`, usá la corriente $\cos(\pi z/d)$ y el $kd$ que corresponda, y compará el diagrama con el del dipolo.
# 4. **Púlsares.** Integrá $\dot\Omega=-K\Omega^3$ (la ecuación de frenado de las notas) y comprobá que el índice de frenado $\Omega\ddot\Omega/\dot\Omega^2$ vale 3 y que la edad es $P/2\dot P$ si el período inicial es chico. Con los datos del Cangrejo ($P=33$ ms, $\dot P=4.2\times10^{-13}$), ¿qué edad da? Se vio explotar en 1054.

# %%
def momentos(t, trayectorias, cargas):
    """p, m (c = 1) y Q de cargas puntuales. trayectorias: lista de arrays (len(t), 3). Devuelve p, m, Q con sus derivadas por diferencias finitas."""
    p = sum(q * r for q, r in zip(cargas, trayectorias))
    v = [np.gradient(r, t, axis=0) for r in trayectorias]
    m = sum(q * np.cross(r, vv) for q, r, vv in zip(cargas, trayectorias, v)) / 2
    Q = sum(q * (3 * np.einsum("ti,tj->tij", r, r) - np.einsum("ti,ti->t", r, r)[:, None, None] * np.eye(3)) for q, r in zip(cargas, trayectorias))
    d = lambda f, n: f if n == 0 else d(np.gradient(f, t, axis=0), n - 1)
    return {"p": p, "pdd": d(p, 2), "m": m, "mdd": d(m, 2), "Q": Q, "Qddd": d(Q, 3)}

def patron_fuente(puntos, J, k, nth=91, nph=72):
    """dP/dΩ promediado en el tiempo (c = 1) de elementos de corriente complejos J (N, 3) en puntos (N, 3): (k²/8π)|n × Σ J e^{−ik n·r}|²."""
    th, ph, n = direcciones(nth, nph)
    fase = np.exp(-1j * k * np.einsum("...i,ni->...n", n, puntos))
    F = np.einsum("...n,ni->...i", fase, J)
    return th, ph, k**2 / (8 * np.pi) * np.sum(np.abs(np.cross(n, F))**2, axis=-1)

# Las herramientas, comprobadas: una carga en un círculo (m = q a v/2 ẑ) y la antena uniforme como suma de elementos
t = np.linspace(0, 2 * np.pi, 2001)
r = np.stack([np.cos(t), np.sin(t), 0 * t], axis=1)
mom = momentos(t, [r], [1.0])
verificar("momentos: carga en un círculo, m_z = q a v/2c", mom["m"][1000, 2], 0.5, tol=1e-5)
dz = 1 / 400; zs = (np.arange(400) + 0.5) * dz - 0.5             # puntos medios de 400 tramos
puntos = np.stack([0 * zs, 0 * zs, zs], axis=1); J = np.zeros((len(zs), 3), complex); J[:, 2] = dz
th2, ph2, dP2 = patron_fuente(puntos, J, 4.0, nth=181, nph=4)
_, dPa = patron_antena(uniforme, 4.0, nth=181)
verificar("patron_fuente reproduce la antena uniforme", np.max(np.abs(dP2[:, 0] - dPa)) / dPa.max(), 0.0, tol=1e-4)
