# %% [markdown]
# # Clase 8 — El método de imágenes
#
# **Objetivos**
# - Comprobar que la solución por imágenes es **la** solución: la relajación de la Clase 7, que no sabe nada de imágenes, llega al mismo potencial.
# - Ver las líneas de campo de una carga frente a una esfera conductora (a tierra, neutra, cargada), la densidad inducida y la fuerza, que puede cambiar de signo.
# - Entender de dónde sale el **factor ½** en la energía de una carga frente a un plano.
# - Sumar infinitas imágenes entre dos placas y comprobar el reparto lineal de la carga inducida.
#
# **Material relacionado:** notas de la Clase 8. Guía 4: problemas 2 a 4.
#
# **Unidades.** Gaussianas, adimensionales: $q=1$ y longitudes en unidades del radio del conductor (o de la separación entre placas).

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
from scipy.integrate import quad, dblquad, trapezoid
from scipy.optimize import brentq

# herramienta: relajacion v1 (NB07)
def energia_discreta(phi, h=1.0):
    """U_h = (1/8π) Σ (φ_a − φ_b)², sumando sobre las aristas de la grilla (en 2D, energía por unidad de longitud)."""
    return (np.sum(np.diff(phi, axis=0)**2) + np.sum(np.diff(phi, axis=1)**2)) / (8 * np.pi)

def relajar(phi, fijo, rho=None, h=1.0, omega=1.0, tol=1e-8, max_pasos=100000, cada=0):
    """Resuelve ∇²φ = −4πρ en una grilla 2D por Gauss–Seidel rojo-negro (sobrerrelajación si omega > 1).

    phi   : valores iniciales; en los nodos con fijo=True quedan como condición de Dirichlet.
    fijo  : arreglo booleano; los nodos del contorno de la grilla tienen que ser fijos.
    rho   : densidad de carga en los nodos (opcional).
    tol   : se detiene cuando ningún nodo cambia más que tol en un paso.
    cada  : si es > 0, guarda la energía discreta cada `cada` pasos.
    Devuelve (phi, pasos, historia), con historia = [(paso, U_h), ...].
    """
    phi = np.array(phi, dtype=float)
    fuente = (np.zeros_like(phi) if rho is None else np.pi * h**2 * rho)[1:-1, 1:-1]
    i, j = np.indices(phi.shape)
    colores = [(((i + j) % 2 == c) & ~fijo)[1:-1, 1:-1] for c in (0, 1)]
    interior = phi[1:-1, 1:-1]                       # vista: modificarla modifica phi
    historia = [(0, energia_discreta(phi, h))] if cada else []
    paso = 0
    for paso in range(1, max_pasos + 1):
        cambio = 0.0
        for m in colores:                            # primero los rojos, después los negros
            prom = 0.25 * (phi[:-2, 1:-1] + phi[2:, 1:-1] + phi[1:-1, :-2] + phi[1:-1, 2:]) + fuente
            delta = omega * (prom[m] - interior[m])
            interior[m] += delta
            cambio = max(cambio, np.abs(delta).max(initial=0.0))
        if cada and paso % cada == 0:
            historia.append((paso, energia_discreta(phi, h)))
        if cambio < tol:
            break
    return phi, paso, historia
# fin herramienta

# %% [markdown]
# ## Experimento 1 ★ — Imágenes frente a relajación
#
# Un hilo con carga $\lambda=1$ por unidad de longitud, a distancia $a=0.5$ del eje, dentro de un cilindro conductor a tierra de radio $R=1$. Las notas deducen la solución por imágenes: una imagen $-\lambda$ en $R^2/a=2$, afuera del cilindro, y
# $$\phi=2\lambda\ln\frac{a\,|\mathbf r-\mathbf r_0'|}{R\,|\mathbf r-\mathbf r_0|}.$$
# Por otro lado, resolvemos el mismo problema con la relajación de la Clase 7: el hilo es una carga $\lambda$ concentrada en un nodo (densidad $\lambda/h^2$), y los nodos con $r\ge R$ están fijos en $\phi=0$.
#
# ### Predecí
# ¿Van a coincidir? ¿Dónde esperás que difieran más?

# %%
R, a, lam = 1.0, 0.5, 1.0
h = 1 / 60
x = np.arange(-61, 62) * h
X, Y = np.meshgrid(x, x, indexing="ij"); s = np.hypot(X, Y)
fijo = s >= R
fijo[0, :] = fijo[-1, :] = fijo[:, 0] = fijo[:, -1] = True
rho = np.zeros_like(X)
i0, j0 = np.argmin(np.abs(x - a)), np.argmin(np.abs(x))
rho[i0, j0] = lam / h**2                              # el hilo, en un nodo
phi_rel, pasos, _ = relajar(np.zeros_like(X), fijo, rho=rho, h=h, omega=2 / (1 + np.sin(np.pi / 122)), tol=1e-9)

def phi_imagen(X, Y, a, R=1.0, lam=1.0):
    """Hilo λ en (a, 0) dentro de un cilindro a tierra de radio R: imagen −λ en (R²/a, 0)."""
    d1 = np.hypot(X - a, Y); d2 = np.hypot(X - R**2 / a, Y)
    with np.errstate(divide="ignore"):
        return 2 * lam * np.log(a * d2 / (R * d1))

phi_im = np.where(s < R, phi_imagen(X, Y, a, R, lam), 0.0)
lejos = (s < 0.95 * R) & (np.hypot(X - a, Y) > 0.1)

fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
niveles = np.linspace(0, 4, 21)
for ax, campo_, titulo in [(axs[0], phi_rel, f"relajación ({pasos} pasos)"), (axs[1], phi_im, "imágenes")]:
    im = ax.contourf(X, Y, np.clip(campo_, 0, 4), levels=niveles, cmap="viridis")
    ax.add_patch(plt.Circle((0, 0), R, fill=False, color="w", lw=1.5))
    ax.set(aspect="equal", title=titulo); ax.grid(False)
fig.colorbar(im, ax=axs[:2], shrink=0.85, label="φ")
err = np.where(lejos, np.abs(phi_rel - phi_im), np.nan)
im = axs[2].imshow(err.T, origin="lower", extent=[x[0], x[-1], x[0], x[-1]], cmap="magma")
axs[2].set(title="|diferencia| (lejos del hilo)"); axs[2].grid(False)
fig.colorbar(im, ax=axs[2], shrink=0.85)
guardar(fig, "nb08_imagen_relajacion"); plt.show()

verificar("máx |relajación − imágenes| / máx φ, lejos del hilo", np.nanmax(err) / phi_im[lejos].max(), 0.0, tol=1e-2)

# %% [markdown]
# ### ¿Qué pasó?
# La relajación no sabe nada de imágenes: solo promedia vecinos. Aun así encuentra el mismo potencial, porque el problema tiene **una sola** solución (Clase 7) y la imagen la había adivinado. La diferencia, de menos del 1 %, se concentra en el borde, donde la grilla aproxima el círculo con escalones.
#
# Mové el hilo: la imagen está en $R^2/a$. ¿Qué le pasa cuando el hilo se acerca al eje? ¿Y cuando se acerca a la pared?

# %%
def mover_hilo(a=0.5):
    t = np.linspace(-1.2, 3.2, 300); T, U = np.meshgrid(t, np.linspace(-1.6, 1.6, 150), indexing="ij")
    ph = np.where(np.hypot(T, U) < R, phi_imagen(T, U, a), np.nan)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.contourf(T, U, np.clip(ph, 0, 4), levels=niveles, cmap="viridis")
    ax.add_patch(plt.Circle((0, 0), R, fill=False, color="k", lw=1.5))
    ax.plot(a, 0, "o", color=COLORES[1], ms=8, label="hilo +λ")
    ax.plot(R**2 / a, 0, "o", mfc="none", mec=COLORES[0], mew=2, ms=8, label=f"imagen −λ en R²/a = {R**2 / a:.2f}")
    ax.set(aspect="equal", xlim=(-1.2, 3.2), ylim=(-1.6, 1.6)); ax.grid(False); ax.legend(loc="upper right")
    plt.show()

interactuar(mover_hilo, a=deslizador("a", 0.5, 0.35, 0.95, 0.05))

# %% [markdown]
# ## Experimento 2 — Una carga frente a una esfera
#
# Una carga $q=1$ en $z=a$ frente a una esfera conductora de radio $R=1$. Las imágenes son $q'=-qR/a$ en $b=R^2/a$ y, si la esfera tiene carga $Q$, una carga $Q-q'$ en el centro. Dibujamos las líneas de campo como curvas de nivel de $\Psi=\sum_iq_i\cos\theta_i$ (Clase 3), solo afuera de la esfera; en gris, cómo seguirían adentro si las imágenes fueran reales.
#
# ### Predecí
# 1. Con la esfera a tierra, ¿todas las líneas que salen de $q$ terminan en la esfera?
# 2. Con la esfera cargada con $Q>0$, ¿se atraen o se repelen?

# %%
def cargas_esfera(a, Q=None, q=1.0, R=1.0):
    """Cargas (valor, z) que reproducen el campo afuera. Q=None: esfera a tierra."""
    qp = -q * R / a
    lista = [(q, a), (qp, R**2 / a)]
    if Q is not None:
        lista.append((Q - qp, 0.0))
    return lista

def Psi(S, Z, lista):
    return sum(qi * (Z - zi) / np.hypot(S, Z - zi) for qi, zi in lista)

def lineas_esfera(a=2.0, Q=0.0, a_tierra=False):
    lista = cargas_esfera(a, None if a_tierra else Q)
    s_ = np.linspace(-3.5, 3.5, 350); z_ = np.linspace(-3, 4.5, 375)
    S, Z = np.meshgrid(s_, z_, indexing="ij")
    ps = Psi(np.abs(S) + 1e-9, Z, lista)
    afuera = np.hypot(S, Z) > 1
    total = sum(abs(qi) for qi, _ in lista)
    niveles_ = np.linspace(-total, total, 26)[1:-1]            # pasos iguales de Ψ = flujos iguales
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    ax = axs[0]
    ax.contour(S, Z, np.where(afuera, ps, np.nan), levels=niveles_, colors=[COLORES[0]], linewidths=1, linestyles="solid")
    ax.contour(S, Z, np.where(~afuera, ps, np.nan), levels=niveles_, colors="0.7", linewidths=0.7, linestyles="--")
    ax.add_patch(plt.Circle((0, 0), 1, color="0.85", zorder=0))
    for qi, zi in lista:
        ax.plot(0, zi, "o", ms=6, color=COLORES[1] if qi > 0 else COLORES[0], mfc="w" if zi < 1 else None)
    ax.set(aspect="equal", xlabel="s", ylabel="z", title="líneas de campo (curvas de nivel de Ψ)"); ax.grid(False)
    # fuerza sobre q en función de la distancia
    ax = axs[1]
    aa = np.linspace(1.02, 6, 300)
    for Qv, c in [(0.0, COLORES[0]), (0.5, COLORES[2]), (2.0, COLORES[3])]:
        ax.plot(aa, [fuerza(av, Qv) for av in aa], color=c, label=f"Q = {Qv}")
    ax.plot(aa, [fuerza(av, None) for av in aa], "k--", lw=1, label="a tierra")
    ax.axhline(0, color="gray", lw=0.8); ax.axvline(a, color="gray", lw=0.8, ls=":")
    ax.set(ylim=(-1, 0.6), xlabel="distancia a", ylabel="fuerza sobre q (> 0: repulsiva)", title="fuerza"); ax.legend()
    plt.show()

def fuerza(a, Q=0.0, q=1.0, R=1.0):
    """Fuerza radial sobre q (en z = a) debida a las imágenes."""
    return sum(q * qi / (a - zi)**2 for qi, zi in cargas_esfera(a, Q, q, R)[1:])

interactuar(lineas_esfera, a=deslizador("a", 2.0, 1.1, 4.0, 0.1), Q=deslizador("Q", 0.0, -2.0, 3.0, 0.25),
            a_tierra=False)

# %% [markdown]
# Ahora la densidad inducida sobre la esfera a tierra. La calculamos derivando numéricamente el potencial de las imágenes justo afuera, y la comparamos con la fórmula de las notas.

# %%
a2 = 2.0
gam = np.linspace(0, np.pi, 2001)
def phi_esfera(r, g, lista):
    return sum(qi / np.sqrt(r**2 + zi**2 - 2 * r * zi * np.cos(g)) for qi, zi in lista)
lista = cargas_esfera(a2)
eps = 1e-5
sigma_num = -(phi_esfera(1 + eps, gam, lista) - phi_esfera(1 - eps, gam, lista)) / (2 * eps) / (4 * np.pi)
sigma_formula = -(a2**2 - 1) / (4 * np.pi * (1 + a2**2 - 2 * a2 * np.cos(gam))**1.5)

fig, ax = plt.subplots(figsize=(6.8, 4.2))
ax.plot(np.degrees(gam), sigma_num, color=COLORES[0], lw=3, alpha=0.5, label="derivada numérica")
ax.plot(np.degrees(gam), sigma_formula, "k--", lw=1.2, label="fórmula de las notas")
ax.set(xlabel="ángulo γ desde la dirección de q (grados)", ylabel="σ", title=f"esfera a tierra, q en a = {a2}"); ax.legend()
plt.show()

Q_ind = trapezoid(sigma_num * 2 * np.pi * np.sin(gam), gam)
verificar("carga inducida = −qR/a", Q_ind, -1 / a2, tol=1e-5)
verificar("σ en γ = 0: −q(a+R)/[4πR(a−R)²]", sigma_num[0], -(a2 + 1) / (4 * np.pi * (a2 - 1)**2), tol=1e-5)
verificar("esfera neutra lejos (a = 30): F = −2q²R³/a⁵", fuerza(30.0, 0.0), -2 / 30**5, tol=1e-2)
verificar("esfera a tierra muy cerca (a = 1.001): F = −q²/4δ²", fuerza(1.001, None), -1 / (4 * 0.001**2), tol=2e-3)
a_cero = brentq(lambda av: fuerza(av, 2.0), 1.001, 10)
print(f"con Q = 2, la fuerza cambia de signo en a = {a_cero:.3f}")

# %% [markdown]
# ### ¿Qué pasó?
# Con la esfera a tierra, solo una fracción $R/a$ de las líneas de $q$ termina en la esfera; las demás se van al infinito. Las líneas grises muestran que, prolongadas adentro, convergen en la imagen; pero adentro el campo real es cero. La densidad inducida es máxima frente a la carga, y su integral es exactamente $q'$.
#
# La fuerza sobre $q$ con la esfera neutra es atractiva y decae como $1/a^5$. Con $Q>0$ es repulsiva de lejos y **atractiva de cerca**: la curva cruza el cero, cada vez más cerca de la esfera a medida que $Q$ crece.
#
# ## Experimento 3 — El factor ½
#
# Una carga $q=1$ frente a un plano a tierra. Calculamos la energía de tres maneras.
# 1. El trabajo para traerla desde el infinito hasta $d=1$, con la imagen **que se mueve** junto con la carga (el problema real).
# 2. El mismo trabajo, pero con una carga $-q$ **fija** en $z=-1$ (el problema ingenuo).
# 3. La energía de interacción del campo, $\frac{1}{4\pi}\int\mathbf E_q\cdot\mathbf E_{\rm imagen}\,d^3r$, integrada solo en $z>0$ (donde hay campo en el problema real) y en todo el espacio.
#
# ### Predecí
# ¿Cuáles de los cuatro números coinciden?

# %%
d = 1.0
W_real = quad(lambda z: 1 / (4 * z**2), np.inf, d)[0]            # agente externo: +q²/4z² hacia arriba, recorrido hacia abajo
W_fija = quad(lambda z: 1 / (z + d)**2, np.inf, d)[0]

def densidad_int(z, s_):
    """(1/4π) E_q · E_imagen, por el elemento 2πs ds dz."""
    Eq = np.array([s_, z - d]) / (s_**2 + (z - d)**2)**1.5
    Ei = -np.array([s_, z + d]) / (s_**2 + (z + d)**2)**1.5
    return (Eq @ Ei) / (4 * np.pi) * 2 * np.pi * s_

U_semi = dblquad(densidad_int, 0, np.inf, 0, np.inf, epsabs=1e-10)[0]
U_todo = U_semi + dblquad(densidad_int, 0, np.inf, -np.inf, 0, epsabs=1e-10)[0]
print(f"trabajo (imagen que se mueve):  {W_real:+.6f}")
print(f"trabajo (imagen fija):          {W_fija:+.6f}")
print(f"interacción en z > 0:           {U_semi:+.6f}")
print(f"interacción en todo el espacio: {U_todo:+.6f}")
verificar("trabajo real = −q²/4d", W_real, -0.25, tol=1e-8)
verificar("interacción del campo en z > 0 = −q²/4d", U_semi, -0.25, tol=1e-6)
verificar("interacción en todo el espacio = −q²/2d", U_todo, -0.5, tol=1e-6)

# %%
# dónde está la energía de interacción: mapa de E_q · E_imagen / 4π en el plano y = 0
t = np.linspace(-3, 3, 301); Xm, Zm = np.meshgrid(t, t, indexing="ij")
with np.errstate(divide="ignore", invalid="ignore"):
    Eq = np.stack([Xm, Zm - d]) / np.hypot(Xm, Zm - d)**3
    Ei = -np.stack([Xm, Zm + d]) / np.hypot(Xm, Zm + d)**3
    mapa = np.sum(Eq * Ei, axis=0) / (4 * np.pi)
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.contourf(Xm, Zm, mapa, levels=np.linspace(-0.05, 0.05, 21), cmap="RdBu_r", extend="both")
ax.axhline(0, color="k", lw=1.5); ax.plot(0, d, "ko"); ax.plot(0, -d, "o", mfc="w", mec="k")
ax.set(aspect="equal", xlabel="x", ylabel="z", title=r"$\mathbf{E}_q\cdot\mathbf{E}_{\rm imagen}/4\pi$ (rojo > 0, azul < 0)"); ax.grid(False)
fig.colorbar(im, ax=ax); plt.show()

# %% [markdown]
# ### ¿Qué pasó?
# El trabajo real, $-q^2/4d$, es la mitad del ingenuo, $-q^2/2d$. La energía de interacción del campo tiene la misma distribución arriba y abajo del plano (el mapa es simétrico): en todo el espacio da $-q^2/2d$, pero en el problema real solo existe la mitad de arriba. **Las imágenes reproducen el campo en $V$, no la energía del sistema ficticio.** La fuerza sí es la de la imagen, porque al mover la carga también se mueve la imagen.
#
# ## Experimento 4 — Infinitas imágenes entre dos placas
#
# Una carga $q=1$ en $x_0$ entre dos placas a tierra en $x=0$ y $x=L=1$. Las imágenes son $+q$ en $x_0+2nL$ y $-q$ en $-x_0+2nL$. Sumamos el campo de muchas imágenes para obtener la densidad inducida en cada placa, y la integramos sobre la placa.
#
# ### Predecí
# ¿Cómo depende de $x_0$ la carga inducida en cada placa? ¿Cuántas imágenes hacen falta para que el potencial se anule en las placas?

# %%
L = 1.0
def imagenes_placas(x0, K):
    n = np.arange(-K, K + 1)
    return np.concatenate([np.ones_like(n, float), -np.ones_like(n, float)]), np.concatenate([x0 + 2 * n * L, -x0 + 2 * n * L])

def potencial_placas(xx, x0, K, s_=0.3):
    qs, xs = imagenes_placas(x0, K)
    return np.sum(qs[None, :] / np.hypot(xx[:, None] - xs[None, :], s_), axis=1)

xx = np.linspace(-0.5, 1.5, 400)
fig, axs = plt.subplots(1, 2, figsize=(12, 4.3))
for K, c in zip([0, 1, 3, 30], RAMPA):
    axs[0].plot(xx, potencial_placas(xx, 0.3, K), color=c, label=f"imágenes con |n| ≤ {K}")
axs[0].axvspan(-0.5, 0, color="0.9"); axs[0].axvspan(L, 1.5, color="0.9")
axs[0].set(xlabel="x", ylabel="φ (a distancia 0.3 de la recta de las cargas)", title="sumas parciales (x₀ = 0.3)"); axs[0].legend()

s_grid = np.linspace(0, 8 * L, 2001)                          # σ decae como exp(−πs/L): en s = 8L ya es despreciable
def carga_placa(x0, placa, K=400):
    """Carga inducida en la placa (x = 0 o x = L), integrando σ = E_n/4π sobre el plano."""
    qs, xs = imagenes_placas(x0, K)
    xp, signo = (0.0, 1.0) if placa == 0 else (L, -1.0)          # E_n = E·(normal que sale de V hacia el metal) con signo
    dx = xp - xs
    Ex = np.sum(qs[None, :] * dx[None, :] / (s_grid[:, None]**2 + dx[None, :]**2)**1.5, axis=1)
    sigma = signo * Ex / (4 * np.pi)
    return trapezoid(sigma * 2 * np.pi * s_grid, s_grid)

x0s = np.linspace(0.1, 0.9, 9)
Q1 = np.array([carga_placa(x0, 0) for x0 in x0s]); Q2 = np.array([carga_placa(x0, 1) for x0 in x0s])
axs[1].plot(x0s, Q1, "o", color=COLORES[0], label="placa 1 (x = 0), imágenes")
axs[1].plot(x0s, Q2, "s", color=COLORES[1], label="placa 2 (x = L), imágenes")
axs[1].plot(x0s, -(1 - x0s / L), "k--", lw=1, label="reciprocidad"); axs[1].plot(x0s, -x0s / L, "k--", lw=1)
axs[1].set(xlabel="posición x₀ de la carga", ylabel="carga inducida", title="reparto lineal", ylim=(-1.2, 0))
axs[1].legend(loc="lower center", ncol=2, fontsize=8)
plt.show()

verificar("Q₁(x₀ = 0.3) = −q(1 − x₀/L)", carga_placa(0.3, 0), -0.7, tol=1e-4)
verificar("Q₂(x₀ = 0.3) = −q x₀/L", carga_placa(0.3, 1), -0.3, tol=1e-4)
verificar("φ en la placa x = 0 (30 pares de imágenes)", potencial_placas(np.array([0.0]), 0.3, 30)[0], 0.0, tol=1e-12)

# %% [markdown]
# ### ¿Qué pasó?
# Con una sola imagen el potencial se anula en una placa pero no en la otra; al agregar reflexiones, se anula en las dos. La suma converge porque cada par de imágenes es un dipolo lejano. La carga inducida es exactamente lineal en $x_0$, como predice la reciprocidad sin sumar ninguna imagen. Si la carga se mueve con velocidad $v$, la placa 2 recibe una corriente $qv/L$: así funcionan los detectores de partículas.
#
# **Cuidado con el orden de las sumas.** Cada imagen, sola, induce $\pm q/2$ en un plano infinito, sin importar a qué distancia esté. Si sumamos esas cargas imagen por imagen, la serie no converge. Hay que sumar primero el campo (que sí converge) y después integrar.
#
# ## Explorá
# 1. **Guía 4, P2.** Con `cargas_esfera` como modelo, poné una esfera neutra entre $+q$ en $z=-a$ y $-q$ en $z=a$ (cada una con su imagen, más las cargas en el centro que hagan falta). Aumentá $a$ con $E_0=2q/a^2$ fijo, dibujá las líneas con `Psi`, y compará la densidad inducida con tu resultado.
# 2. **Guía 4, P3.** Construí las imágenes para dos semiplanos a tierra que forman un ángulo recto, y verificá que el potencial se anula sobre los dos. Probá con $60^\circ$ y con $70^\circ$. ¿Qué falla en el segundo caso?
# 3. **Guía 4, P4.** Dividí el segmento cargado en muchas cargas puntuales, sumá sus cargas imagen $-qR/z$, y compará con tu fórmula de la carga inducida.
# 4. **El dipolo frente al plano.** Representá el dipolo como dos cargas $\pm q$ separadas $\delta\ll h$, calculá la mitad de su energía de interacción con las imágenes en función del ángulo, y compará con $-p^2(1+\cos^2\theta)/16h^3$. ¿Hacia dónde gira el dipolo?
