# %% [markdown]
# # Clase 11 — Coordenadas esféricas y polinomios de Legendre
#
# **Objetivos**
# - Continuar desde el eje: comparar la **serie de Legendre** del potencial de un disco cargado con el potencial exacto, y ver dónde converge y dónde no.
# - Construir los polinomios de Legendre con la recurrencia de las notas, y comprobar su ortogonalidad, su norma y la función generatriz; ver por qué la serie tiene que terminar.
# - Dibujar la **esfera conductora en un campo uniforme** y verificar $\sigma=\frac{3E_0}{4\pi}\cos\theta$ y el campo uniforme de una cáscara con $\sigma_0\cos\theta$.
# - Resolver una carga dentro de una esfera a tierra con Legendre y reencontrar la **imagen** de la Clase 8.
#
# **Material relacionado:** notas de la Clase 11. Guía 4: problemas 8 a 10.
#
# **Unidades.** Gaussianas, adimensionales: longitudes en unidades del radio $R$, cargas en unidades de $q$ (o de $Q$ para el disco).

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
from scipy.special import ellipk, eval_legendre, binom


# %% [markdown]
# ## Experimento 1 ★ — El disco cargado, desde el eje
#
# Un disco de radio $R=1$ con densidad $\sigma$ tal que la carga total es $Q=\pi R^2\sigma=1$. Sobre el eje, $\phi=2\pi\sigma(\sqrt{R^2+z^2}-|z|)$. Las notas lo desarrollan para $r>R$ y lo continúan:
# $$\phi(r,\theta)=2\pi\sigma\sum_{k\ge1}\binom{1/2}{k}\frac{R^{2k}}{r^{2k-1}}\,P_{2k-2}(\cos\theta)=\frac Qr-\frac{QR^2}{4r^3}P_2+\frac{QR^4}{8r^5}P_4-\cdots$$
# El potencial exacto fuera del eje lo calculamos sumando anillos, con la integral elíptica de la Clase 3.
#
# ### Predecí
# 1. ¿Para qué distancias converge la serie? ¿Qué pasa con $r$ apenas mayor que $R$? ¿Y con $r<R$?
# 2. A igual $r$, ¿el potencial es mayor sobre el eje o en el plano del disco?

# %%
sigma = 1 / np.pi                                    # Q = πR²σ = 1

def potencial_disco(s, z, R=1.0, n=2000):
    """Potencial exacto de un disco uniforme, sumando anillos (Clase 3)."""
    x, w = np.polynomial.legendre.leggauss(n)
    a = 0.5 * R * (x + 1); pesos = 0.5 * R * w
    s = np.asarray(s, float)[..., None]; z = np.asarray(z, float)[..., None]
    Qa = 2 * np.pi * sigma * a
    D2 = (a + s)**2 + z**2
    return np.sum(pesos * (2 * Qa / np.pi) * ellipk(4 * a * s / D2) / np.sqrt(D2), axis=-1)

def serie_disco(r, theta, kmax, R=1.0):
    """Suma parcial de la serie de Legendre del disco, con k = 1 … kmax."""
    k = np.arange(1, kmax + 1)
    c = 2 * np.pi * sigma * binom(0.5, k) * R**(2 * k)
    r = np.asarray(r, float)[..., None]; x = np.cos(np.asarray(theta, float))[..., None]
    return np.sum(c / r**(2 * k - 1) * eval_legendre(2 * k - 2, x), axis=-1)

th = np.linspace(0, np.pi / 2, 91)
fig, axs = plt.subplots(1, 2, figsize=(12, 4.4))
for r, c in zip([3.0, 1.5, 1.1], RAMPA[1:]):
    exacto = potencial_disco(r * np.sin(th), r * np.cos(th))
    axs[0].plot(np.degrees(th), exacto * r, color=c, lw=3, alpha=0.4)
    axs[0].plot(np.degrees(th), serie_disco(r, th, 30) * r, "--", color=c, lw=1.2, label=f"r = {r}R")
axs[0].set(xlabel="θ (grados)", ylabel="r φ / Q", title="exacto (grueso) y serie con 30 términos (trazos)"); axs[0].legend()

ks = np.arange(1, 41)
for r, c in zip([3.0, 1.5, 1.1, 1.02, 0.9], [RAMPA[1], RAMPA[2], RAMPA[3], COLORES[1], COLORES[4]]):
    ex = potencial_disco(r * np.sin(np.pi / 3), r * np.cos(np.pi / 3))
    err = [abs(serie_disco(r, np.pi / 3, k) - ex) for k in ks]
    axs[1].semilogy(ks, err, "o-", ms=3, color=c, label=f"r = {r}R")
axs[1].set(xlabel="número de términos", ylabel="|serie − exacto| en θ = 60°", ylim=(1e-16, 1e3), title="convergencia"); axs[1].legend()
guardar(fig, "nb11_disco"); plt.show()

ex = potencial_disco(1.5 * np.sin(np.pi / 3), 1.5 * np.cos(np.pi / 3))
verificar("serie (40 términos) = exacto en r = 1.5R, θ = 60°", serie_disco(1.5, np.pi / 3, 40), ex, tol=1e-10)
# sobre el eje la comparación es con la fórmula cerrada
eje_exacto = lambda z: 2 * np.pi * sigma * (np.sqrt(1 + z**2) - z)
verificar("sobre el eje, r = 1.1R: serie (200 términos) = 2πσ(√(R²+z²) − z)", serie_disco(1.1, 0.0, 200), eje_exacto(1.1), tol=1e-10)
salto = abs(serie_disco(0.9, 0.0, 60) - serie_disco(0.9, 0.0, 40))
print(f"r = 0.9R: |S₆₀ − S₄₀| = {salto:.3g}  (las sumas parciales crecen: la serie diverge)")
verificar("r = 0.9R: la serie diverge (|S₆₀ − S₄₀| > 1)", float(salto > 1), 1.0, tol=1e-12)
# eje frente a plano, a r = 3R
r = 3.0
eje, plano = potencial_disco(0.0, r), potencial_disco(r, 0.0)
print(f"r = 3R: φ(eje) = {eje:.6f}   φ(plano) = {plano:.6f}   Q/r = {1 / r:.6f}")
verificar("φ(plano) − φ(eje) a r = 3R ≈ (3/8) Q R²/r³ (cuadrupolo)", plano - eje, 3 / 8 / r**3, tol=5e-2)

# %% [markdown]
# ### ¿Qué pasó?
# Para $r>R$ la serie converge al potencial exacto, con un error que baja aproximadamente en un factor $(R/r)^2$ por término: rapidísimo a $r=3R$, lentísimo a $r=1.02R$. Para $r<R$ la serie **diverge**: la bola $r<R$ contiene parte del disco, y ahí no hay desarrollo en Legendre que valga. A igual distancia, el potencial es mayor en el plano que en el eje: es el cuadrupolo negativo de un objeto achatado, $-\frac{QR^2}{4r^3}P_2$.
#
# ## Experimento 2 — Los polinomios de Legendre
#
# Construimos los $P_l$ con la recurrencia de las notas, $a_{j+2}=\frac{j(j+1)-l(l+1)}{(j+1)(j+2)}a_j$, normalizados con $P_l(1)=1$, y los comparamos con los de `scipy`. Después miramos qué pasa si $l$ **no** es entero.
#
# ### Predecí
# Con $l=2.5$ la serie no termina. ¿Cómo se comporta cerca de $x=1$?

# %%
def legendre_serie(l, x, jmax=4000):
    """Serie de la ecuación de Legendre con la paridad de round(l); normalizada con P(1)=1 si termina."""
    j0 = int(round(l)) % 2
    a = np.zeros(jmax + 2); a[j0] = 1.0
    for j in range(j0, jmax, 2):
        a[j + 2] = (j * (j + 1) - l * (l + 1)) / ((j + 1) * (j + 2)) * a[j]
    val = np.polynomial.polynomial.polyval(x, a)
    return val / np.polynomial.polynomial.polyval(1.0, a) if float(l).is_integer() else val

x = np.linspace(-1, 1, 401)
fig, axs = plt.subplots(1, 2, figsize=(12, 4.3))
for l, c in zip(range(5), COLORES):
    axs[0].plot(x, legendre_serie(l, x), color=c, label=f"P{l}")
axs[0].set(xlabel="x = cos θ", title="polinomios de Legendre"); axs[0].legend(ncol=5, fontsize=9)
uno_menos_x = np.logspace(-4, 0, 200)
xx = 1 - uno_menos_x
axs[1].semilogx(uno_menos_x, legendre_serie(2.5, xx, jmax=60000), color=COLORES[1], label="l = 2.5 (serie par, sin terminar)")
axs[1].semilogx(uno_menos_x, legendre_serie(2, xx), color=COLORES[0], label="l = 2")
axs[1].invert_xaxis()
axs[1].set(xlabel="1 − x (escala logarítmica)", title="si la serie no termina, diverge en x = 1"); axs[1].legend()
plt.show()

err = max(np.abs(legendre_serie(l, x) - eval_legendre(l, x)).max() for l in range(9))
verificar("recurrencia = P_l de scipy (l ≤ 8)", err, 0.0, tol=1e-12)
xg, wg = np.polynomial.legendre.leggauss(60)
M = np.array([[np.sum(wg * eval_legendre(l, xg) * eval_legendre(m, xg)) for m in range(8)] for l in range(8)])
verificar("ortogonalidad y norma: máx |∫P_l P_m − 2δ/(2l+1)|", np.abs(M - np.diag(2 / (2 * np.arange(8) + 1))).max(), 0.0, tol=1e-12)
t, xv = 0.5, 0.3
verificar("función generatriz Σ tˡ P_l(x) = (1 − 2xt + t²)^{−1/2}", np.sum(t**np.arange(80) * eval_legendre(np.arange(80), xv)),
          1 / np.sqrt(1 - 2 * xv * t + t**2), tol=1e-12)

# %% [markdown]
# ### ¿Qué pasó?
# La recurrencia da exactamente los $P_l$, que son ortogonales con norma $\frac{2}{2l+1}$ y salen de la función generatriz. Con $l=2.5$ la serie no termina y, cerca de $x=1$, crece en módulo como $\ln\frac{1}{1-x}$ (una recta en el gráfico logarítmico): un potencial así sería infinito sobre el eje. Por eso $l$ tiene que ser entero.
#
# ## Experimento 3 — La esfera en un campo uniforme
#
# Una esfera conductora de radio $R=1$ en un campo $E_0=1$ según $z$. Fuera de ella $\phi=-E_0\left(r-\frac{R^3}{r^2}\right)\cos\theta$. Para las líneas de campo usamos una función de flujo, como en la Clase 3: para el campo uniforme más el dipolo $p=R^3E_0$ es $\Psi\propto s^2\left(\frac12+\frac{R^3}{r^3}\right)$, con $s=r\sin\theta$. (La comprobamos numéricamente: $\mathbf E\cdot\nabla\Psi=0$.)
#
# ### Predecí
# ¿Dónde es más intenso el campo sobre la esfera? ¿Cuánto vale ahí?

# %%
def phi_esfera(x, z, E0=1.0, R=1.0):
    r = np.hypot(x, z)
    return np.where(r >= R, -E0 * (1 - R**3 / r**3) * z, 0.0)

def Psi(x, z, R=1.0):
    r = np.hypot(x, z)
    return x**2 * (0.5 + R**3 / r**3)

t = np.linspace(-2.5, 2.5, 401); X, Z = np.meshgrid(t, t, indexing="ij")
afuera = np.hypot(X, Z) > 1
fig, ax = plt.subplots(figsize=(6, 6))
ax.contour(X, Z, np.where(afuera, Psi(X, Z), np.nan), levels=np.linspace(0.05, 3, 16)**2 / 2, colors=[COLORES[0]], linewidths=1)
ax.contour(X, Z, np.where(afuera, phi_esfera(X, Z), np.nan), levels=np.linspace(-2.4, 2.4, 13), colors=[COLORES[1]], linewidths=0.8, linestyles="solid")
ax.add_patch(plt.Circle((0, 0), 1, color="0.8"))
ax.set(aspect="equal", xlabel="x", ylabel="z", title="líneas de campo (azul) y equipotenciales (naranja)"); ax.grid(False)
plt.show()

# comprobaciones: E·∇Ψ = 0, σ y la cáscara con σ₀ cos θ
h = 1e-5; xp, zp = 0.7, 1.3
Ex = -(phi_esfera(xp + h, zp) - phi_esfera(xp - h, zp)) / (2 * h); Ez = -(phi_esfera(xp, zp + h) - phi_esfera(xp, zp - h)) / (2 * h)
gx = (Psi(xp + h, zp) - Psi(xp - h, zp)) / (2 * h); gz = (Psi(xp, zp + h) - Psi(xp, zp - h)) / (2 * h)
verificar("E·∇Ψ = 0 (Ψ es constante sobre las líneas)", Ex * gx + Ez * gz, 0.0, tol=1e-8)
Er_polo = -(phi_esfera(0.0, 1 + 2 * h) - phi_esfera(0.0, 1 + h)) / h
verificar("campo en el polo = 3E₀", Er_polo, 3.0, tol=1e-4)

def direcciones(M=400):
    """M direcciones casi uniformes (espiral de Fibonacci)."""
    i = np.arange(M) + 0.5
    th = np.arccos(1 - 2 * i / M); ph = np.pi * (1 + 5**0.5) * i
    return np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], axis=1)

n = direcciones(200000); s0 = 3 / (4 * np.pi)                    # σ₀ = 3E₀/4π
def campo_cascara(p):
    d = p[None, :] - n
    return np.sum((s0 * n[:, 2] * 4 * np.pi / len(n))[:, None] * d / np.linalg.norm(d, axis=1)[:, None]**3, axis=0)
for p in ([0, 0, 0], [0.3, -0.2, 0.4]):
    verificar(f"cáscara σ₀cos θ: E_z adentro en {p} = −4πσ₀/3 = −E₀", campo_cascara(np.array(p, float))[2], -1.0, tol=1e-3)

# %% [markdown]
# ### ¿Qué pasó?
# Las líneas llegan perpendiculares a la esfera y se concentran en los polos, donde el campo vale $3E_0$. La cáscara con $\sigma=\frac{3E_0}{4\pi}\cos\theta$ produce adentro un campo uniforme $-E_0\hat{\mathbf z}$ en cualquier punto: sumado al campo aplicado, da cero. Eso es lo que hace la carga inducida en el conductor.
#
# ## Experimento 4 — Una carga dentro de una esfera a tierra
#
# Una carga $q=1$ en $z=d=0.5$ dentro de una esfera a tierra de radio $R=1$. Las notas encuentran con Legendre $A_l=-qd^l/R^{2l+1}$, y muestran que la suma es la imagen $-qR/d$ en $R^2/d$.
#
# ### Predecí
# ¿Cuántos términos hacen falta para que la serie reproduzca la imagen en el borde de la esfera? ¿Y cerca del centro?

# %%
d = 0.5
def homogenea_serie(r, theta, L):
    l = np.arange(L)
    return np.sum(-d**l / 1.0**(2 * l + 1) * np.asarray(r)[..., None]**l * eval_legendre(l, np.cos(np.asarray(theta))[..., None]), axis=-1)

def homogenea_imagen(r, theta):
    b = 1 / d
    return -(1 / d) / np.sqrt(r**2 + b**2 - 2 * r * b * np.cos(theta))

rr = np.array([0.2, 0.6, 0.95]); tt = np.array([0.3, 1.7, 2.9])
Ls = np.arange(1, 60)
fig, ax = plt.subplots(figsize=(6.6, 4.3))
for r_, t_, c in zip(rr, tt, RAMPA[1:]):
    ax.semilogy(Ls, [abs(homogenea_serie(r_, t_, L) - homogenea_imagen(r_, t_)) for L in Ls], color=c, label=f"r = {r_}R")
ax.set(xlabel="número de términos L", ylabel="|serie − imagen|", title="la serie de Legendre converge a la imagen", ylim=(1e-15, 1)); ax.legend()
plt.show()
verificar("serie de Legendre (60 términos) = imagen −qR/d en R²/d, en r = 0.95R",
          homogenea_serie(0.95, 2.9, 60), homogenea_imagen(0.95, 2.9), tol=1e-12)

# %% [markdown]
# ### ¿Qué pasó?
# La serie converge a la imagen en toda la esfera, con un error que baja como $(rd/R^2)^L$: rápido cerca del centro, más lento cerca del borde (en $r=R$, el cociente es $d/R=0.5$). Legendre encontró la imagen de la Clase 8 sin que tuviéramos que adivinarla.
#
# ## Explorá
# 1. **Guía 4, P9.** Para el anillo, desarrollá el potencial del eje para $r<C$ y para $r>C$, continuá, y compará con el potencial exacto del anillo, $\frac{2Q}{\pi}\frac{K(m)}{\sqrt{(C+s)^2+z^2}}$ con $m=\frac{4Cs}{(C+s)^2+z^2}$ (Clase 3). ¿Cómo converge cerca de $r=C$?
# 2. **Guía 4, P8.** Calculá numéricamente los $A_l$ de la cáscara con $V_0(1-\cos\theta)$ con cuadratura de Gauss (`np.polynomial.legendre.leggauss`) y compará con tu resultado.
# 3. **Guía 4, P10.** Sumá la serie de Legendre del cuadrupolo lineal dentro de la cáscara cargada, y compará con la solución por imágenes.
# 4. **Hemisferios a $\pm V_0$.** Calculá muchos coeficientes $A_l=(2l+1)V_0\int_0^1P_l\,dx$, sumá la serie sobre el eje y compará con la fórmula cerrada de la Clase 9. ¿Qué pasa con la serie justo sobre la esfera, en el ecuador?
