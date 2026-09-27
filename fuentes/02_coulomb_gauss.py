# %% [markdown]
# # Clase 2 — Coulomb, la delta de Dirac y la ley de Gauss
#
# **Objetivos**
# - Comprobar que el flujo de $\mathbf E$ a través de una superficie cerrada **no depende de su forma**: vale $4\pi q$ si la carga está adentro y $0$ si está afuera, aunque el campo sobre la superficie no sea nulo.
# - Ver cuándo una colección de cargas puntuales "parece" una distribución continua.
# - Dibujar líneas de campo en las que el **número de líneas es proporcional a la carga**.
# - Construir la delta de Dirac, y su derivada, como límite de distribuciones cada vez más angostas.
#
# **Material relacionado:** notas de la Clase 2. Guía 1: problemas 3 y 6. Guía 2: problemas 1, 2 y 4.

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

from scipy.optimize import brentq

# %% [markdown]
# ## Experimento 1 ★ — El flujo a través de una papa
#
# Tomamos una superficie cerrada deformada, una "papa", descripta en esféricas por
# $$ r(\theta,\varphi) = 1 + A\left[\,0.6\cos 3\theta + 0.4\,\sin^2\theta\cos 2\varphi\,\right], $$
# y una carga puntual $q=1$ que movemos a lo largo del eje $x$, desde el centro hasta afuera. Calculamos numéricamente el flujo $\Phi=\oint \mathbf E\cdot d\mathbf a$.
#
# ### Predecí
# 1. Dibujá $\Phi/(4\pi q)$ en función de la posición $x_0$ de la carga.
# 2. Cuando la carga está **afuera**, ¿el campo sobre la superficie es cero? ¿Y el flujo?
# 3. ¿Qué pasa si la carga está **exactamente sobre** la superficie?

# %%
def superficie(A, n=400):
    """Puntos X y normales N = ∂X/∂θ × ∂X/∂φ (hacia afuera) de la papa, en una grilla de punto medio."""
    th = (np.arange(n) + 0.5) * np.pi / n
    ph = (np.arange(2 * n) + 0.5) * np.pi / n
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    r = 1 + A * (0.6 * np.cos(3 * TH) + 0.4 * np.sin(TH)**2 * np.cos(2 * PH))
    r_th = A * (-1.8 * np.sin(3 * TH) + 0.8 * np.sin(TH) * np.cos(TH) * np.cos(2 * PH))
    r_ph = A * (-0.8 * np.sin(TH)**2 * np.sin(2 * PH))
    er = np.array([np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH), np.cos(TH)])
    eth = np.array([np.cos(TH) * np.cos(PH), np.cos(TH) * np.sin(PH), -np.sin(TH)])
    eph = np.array([-np.sin(PH), np.cos(PH), 0 * PH])
    X = r * er
    N = r**2 * np.sin(TH) * er - r * r_th * np.sin(TH) * eth - r * r_ph * eph
    dA = (np.pi / n) * (np.pi / n)          # dθ dφ
    return X, N, dA

def campo_carga(X, q=1.0, x0=(0.0, 0.0, 0.0)):
    """Campo de Coulomb (gaussiano) de una carga q en x0, evaluado en los puntos X (3 x ...)."""
    R = X - np.reshape(x0, (3,) + (1,) * (X.ndim - 1))
    return q * R / np.sum(R**2, axis=0)**1.5

def flujo(A, x0, n=400):
    X, N, dA = superficie(A, n)
    return np.sum(campo_carga(X, 1.0, x0) * N) * dA

A = 0.4
borde_x = 1 + A * (0.6 * np.cos(3 * np.pi / 2) + 0.4)      # donde la papa corta el eje +x
posiciones = np.linspace(0.0, 2.2, 45)
flujos = np.array([flujo(A, (x0, 0, 0), n=200) for x0 in posiciones])

fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
t = np.linspace(0, 2 * np.pi, 400)                           # corte y = 0 (φ = 0 y φ = π)
th = np.where(t <= np.pi, t, 2 * np.pi - t); ph = np.where(t <= np.pi, 0.0, np.pi)
rr = 1 + A * (0.6 * np.cos(3 * th) + 0.4 * np.sin(th)**2 * np.cos(2 * ph))
axs[0].plot(rr * np.sin(th) * np.cos(ph), rr * np.cos(th), color="k")
axs[0].plot(posiciones, 0 * posiciones, ".", color=COLORES[1], ms=4, label="posiciones de la carga")
axs[0].set(aspect="equal", xlabel="x", ylabel="z", title="corte y = 0 de la papa")
axs[0].legend(loc="upper right")
axs[1].plot(posiciones, flujos / (4 * np.pi), "o-", ms=4)
axs[1].axvline(borde_x, color="0.5", ls="--", lw=1)
axs[1].text(borde_x + 0.03, 0.55, "borde", color="0.4")
axs[1].set(xlabel="posición de la carga  $x_0$", ylabel=r"$\Phi / 4\pi q$", title="flujo a través de la papa")
plt.tight_layout(); guardar(fig, "nb02_flujo_papa"); plt.show()

verificar("flujo con la carga adentro (x0 = 0.3) = 4π", flujo(A, (0.3, 0, 0)), 4 * np.pi, tol=1e-4)
verificar("flujo con la carga afuera (x0 = 2.0) = 0", flujo(A, (2.0, 0, 0)), 0.0, tol=1e-4)

# %% [markdown]
# ### ¿Qué pasó?
# El flujo es un **escalón**: vale $4\pi q$ mientras la carga está adentro, sin importar dónde ni la forma de la papa, y cae a $0$ cuando sale. No hay nada "intermedio" salvo exactamente sobre el borde.
#
# **¿El campo es cero cuando la carga está afuera?** No. Miremos $\mathbf E\cdot\hat{\mathbf n}$ sobre la superficie: hay zonas donde el flujo entra (azul) y zonas donde sale (rojo), y se cancelan exactamente.

# %%
def mapa_flujo_local(x0=1.8, A=0.4):
    X, N, _ = superficie(A, n=60)
    En = np.sum(campo_carga(X, 1.0, (x0, 0, 0)) * N, axis=0) / np.linalg.norm(N, axis=0)
    escala = np.percentile(np.abs(En), 80)              # satura los extremos para ver ambos signos
    colores = plt.cm.RdBu_r(0.5 + 0.5 * np.clip(En / escala, -1, 1))
    fig = plt.figure(figsize=(6.5, 5.5))
    ax = fig.add_subplot(projection="3d")
    X, colores = np.concatenate([X, X[:, :, :1]], axis=2), np.concatenate([colores, colores[:, :1]], axis=1)  # cerrar la costura en φ
    ax.plot_surface(*X, facecolors=colores, rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)
    ax.scatter([x0], [0], [0], color=COLORES[7], s=60)
    ax.set(xlim=(-1.5, 2), ylim=(-1.5, 1.5), zlim=(-1.5, 1.5), box_aspect=(3.5, 3, 3))
    ax.set_title(f"E·n̂ sobre la papa (rojo: sale, azul: entra)\nflujo / 4πq = {flujo(A, (x0, 0, 0)) / (4 * np.pi):.4f}")
    ax.set_axis_off(); plt.show()

interactuar(mapa_flujo_local, x0=deslizador("x0", 1.8, 0.0, 2.5, 0.05), A=deslizador("A", 0.4, 0.0, 0.6, 0.05))

# %% [markdown]
# **La ley de Gauss no dice que el campo sea cero; dice que el flujo neto solo "ve" la carga encerrada.** Por eso, para *calcular* $\mathbf E$ con Gauss hace falta una simetría que fije su dirección y su módulo sobre la superficie.
#
# ## Experimento 2 — ¿Cuándo un anillo de cargas parece un anillo continuo?
#
# Repartimos una carga total $Q=1$ en $N$ cargas iguales sobre un anillo de radio $a=1$. Comparamos el campo con el del anillo continuo:
# - **sobre el eje**;
# - en el plano del anillo, a una distancia $\delta$ del alambre, hacia adentro.
#
# ### Predecí
# ¿Cuántas cargas hacen falta para que el campo sobre el eje coincida con el del anillo continuo? ¿Y a una distancia $\delta = 0.1$ del anillo?

# %%
def campo_anillo_discreto(puntos, N, a=1.0, Q=1.0):
    """Campo de N cargas Q/N equiespaciadas en un anillo de radio a (plano z = 0), en puntos (3 x P)."""
    ang = 2 * np.pi * np.arange(N) / N
    fuentes = np.array([a * np.cos(ang), a * np.sin(ang), 0 * ang])    # 3 x N
    R = puntos[:, :, None] - fuentes[:, None, :]                       # 3 x P x N
    return (Q / N) * np.sum(R / np.sum(R**2, axis=0)**1.5, axis=2)

z = 0.7
Ez_eje = campo_anillo_discreto(np.array([[0.0], [0.0], [z]]), 3)[2, 0]
verificar("E_z en el eje con N = 3 cargas = Qz/(z²+a²)^(3/2)", Ez_eje, z / (z**2 + 1)**1.5, tol=1e-12)

def error_en_el_plano(N, delta, n_ang=40):
    """Error relativo máximo, sobre un período angular, a distancia delta del anillo (hacia adentro)."""
    phi = np.linspace(0, 2 * np.pi / N, n_ang)
    p = np.array([(1 - delta) * np.cos(phi), (1 - delta) * np.sin(phi), 0 * phi])
    E_N = campo_anillo_discreto(p, N)
    E_ref = campo_anillo_discreto(p, 3000)            # prácticamente continuo
    return np.max(np.linalg.norm(E_N - E_ref, axis=0) / np.linalg.norm(E_ref, axis=0))

Ns = np.arange(4, 121, 4)
fig, ax = plt.subplots(figsize=(7, 4.6))
for delta, color in zip([0.05, 0.1, 0.2], COLORES):
    err = [error_en_el_plano(N, delta) for N in Ns]
    ax.semilogy(Ns, err, "o-", ms=3, color=color, label=f"δ = {delta}")
    ax.semilogy(Ns, err[3] * (1 - delta)**(Ns - Ns[3]), "--", lw=1, color=color)
ax.set(xlabel="número de cargas N", ylabel="error relativo de E", ylim=(1e-12, 10),
       title="Discreto frente a continuo, a distancia δ del anillo\n(guiones: pendiente $(1-\\delta)^N$)")
ax.legend(); plt.tight_layout(); guardar(fig, "nb02_discreto_continuo"); plt.show()

# %% [markdown]
# ### ¿Qué pasó?
# - **Sobre el eje** basta con $N = 2$ o $3$: por simetría, todas las cargas están a la misma distancia y las componentes transversales se cancelan.
# - **Cerca del anillo** el error decae **exponencialmente** con $N$, aproximadamente como $(1-\delta)^N \approx e^{-N\delta}$. La "granularidad" de las cargas se nota solo a distancias comparables con la separación entre ellas, $2\pi a/N$; más lejos, la descripción continua es excelente.
#
# Esto justifica usar densidades $\rho$, $\sigma$ y $\lambda$ para materia hecha de átomos. Volveremos a ver el mismo decaimiento exponencial en la Clase 10: los detalles finos de un borde se "curan" a una distancia del orden de su escala.
#
# ## Experimento 3 — Líneas de campo que respetan la carga
#
# Para cargas sobre el eje $z$, las líneas de campo son curvas de nivel de la **función de flujo**
# $$\Psi(\rho,z) = \sum_i q_i \cos\theta_i, \qquad \cos\theta_i = \frac{z - z_i}{|\mathbf r - \mathbf r_i|}.$$
# Con niveles equiespaciados de $\Psi$, entre dos líneas vecinas pasa siempre el mismo flujo. Por eso **el número de líneas que nacen de cada carga es proporcional a $q$**, cosa que `streamplot` no garantiza.
#
# ### Predecí
# Con $q_1 = +2$ abajo y $q_2 = -1$ arriba: ¿todas las líneas que salen de $q_1$ terminan en $q_2$? ¿Hay un punto donde $\mathbf E = 0$? ¿Dónde?

# %%
def lineas(q2=-1.0, d=1.0, q1=2.0, n_lineas=24):
    zs, qs = np.array([-d / 2, d / 2]), np.array([q1, q2])
    x = np.linspace(-3, 3, 500); zz = np.linspace(-3, 4, 600)
    X, Z = np.meshgrid(x, zz)
    Psi = sum(q * (Z - zi) / np.sqrt(X**2 + (Z - zi)**2 + 1e-12) for q, zi in zip(qs, zs))
    niveles = np.linspace(-np.abs(qs).sum(), np.abs(qs).sum(), n_lineas * 2 + 1)
    fig, ax = plt.subplots(figsize=(5.5, 6))
    ax.contour(X, Z, Psi, levels=niveles, colors="0.25", linewidths=0.8, linestyles="solid")
    for q, zi in zip(qs, zs):
        ax.plot(0, zi, "o", ms=10 + 3 * abs(q), color=COLORES[7] if q > 0 else COLORES[0])
        ax.text(0.3, zi, f"{q:+.1f}", va="center", ha="left", backgroundcolor="white")
    ax.set(aspect="equal", xlabel="x", ylabel="z", title=f"q1 = {q1:+.1f},  q2 = {q2:+.1f}")
    ax.grid(False); plt.show()

interactuar(lineas, q2=deslizador("q2", -1.0, -3, 3, 0.25), d=deslizador("d", 1.0, 0.5, 2, 0.1))

# punto de campo nulo sobre el eje para q1 = +2 en z = -1/2, q2 = -1 en z = +1/2
Ez = lambda z: 2 / (z + 0.5)**2 * np.sign(z + 0.5) - 1 / (z - 0.5)**2 * np.sign(z - 0.5)
verificar("punto de campo nulo: z = (3 + 2√2)/2", brentq(Ez, 0.6, 10), (3 + 2 * np.sqrt(2)) / 2, tol=1e-8)

# %% [markdown]
# ### ¿Qué pasó?
# De $q_1=+2$ salen el doble de líneas de las que llegan a $q_2=-1$. La mitad de las líneas de $q_1$ termina en $q_2$ y la otra mitad escapa al infinito: lejos, el sistema se ve como una carga $+1$. En el eje, por encima de $q_2$, hay un **punto de campo nulo** (en $z = (3+2\sqrt2)/2 \approx 2.91$) donde las líneas se cruzan en ángulo.
#
# ## Experimento 4 — La delta de Dirac como límite (Guía 2, P4)
#
# **Capa simple:** una lámina de espesor $w$ con densidad $\rho_0 = \sigma/w$. Al achicar $w$ manteniendo $\sigma$, $\rho \to \sigma\,\delta(z)$.
#
# **Capa doble:** dos láminas contiguas con $\mp\rho_0$ (negativa abajo, positiva arriba). Para que quede algo en el límite, $\rho_0 = 4D/w^2$, donde $D$ es el momento dipolar por unidad de área. Así $\rho \to -D\,\delta'(z)$.
#
# ### Predecí
# En cada caso, al hacer $w \to 0$: ¿qué función salta, $E_z$ o $\phi$? ¿En cuánto?

# %%
def perfiles(w=0.5, sigma=1.0, D=1.0):
    dz = 1e-3; z = -2 + (np.arange(4000) + 0.5) * dz     # grilla de punto medio: los bordes caen entre puntos
    simple = np.where(np.abs(z) < w / 2, sigma / w, 0.0)
    doble = np.where((z > -w / 2) & (z < 0), -4 * D / w**2, 0.0) + np.where((z >= 0) & (z < w / 2), 4 * D / w**2, 0.0)
    fig, axs = plt.subplots(3, 2, figsize=(10, 7), sharex=True)
    for col, (rho, nombre) in enumerate([(simple, "capa simple"), (doble, "capa doble")]):
        Q_abajo = np.cumsum(rho) * dz                      # carga por unidad de área debajo de z
        Ez = 2 * np.pi * (Q_abajo - (Q_abajo[-1] - Q_abajo))   # 2π(abajo − arriba)
        phi = -np.cumsum(Ez) * dz; phi -= phi[0]
        for ax, y, et in zip(axs[:, col], (rho, Ez, phi), ("ρ", "$E_z$", "φ")):
            ax.plot(z, y, color=COLORES[col]); ax.set_ylabel(et)
        axs[0, col].set_title(nombre)
    axs[2, 0].set_xlabel("z"); axs[2, 1].set_xlabel("z")
    plt.tight_layout(); plt.show()

interactuar(perfiles, w=deslizador("ancho w", 0.5, 0.02, 1.0, 0.02))

# saltos en el límite de ancho chico
dz = 1e-4; z = -2 + (np.arange(40000) + 0.5) * dz; w = 0.01
simple = np.where(np.abs(z) < w / 2, 1.0 / w, 0.0)
Qs = np.cumsum(simple) * dz; Ez_s = 2 * np.pi * (2 * Qs - Qs[-1])
verificar("salto de E_z en la capa simple = 4πσ", Ez_s[-1] - Ez_s[0], 4 * np.pi, tol=1e-3)
doble = np.where((z > -w / 2) & (z < 0), -4 / w**2, 0.0) + np.where((z >= 0) & (z < w / 2), 4 / w**2, 0.0)
Qd = np.cumsum(doble) * dz; Ez_d = 2 * np.pi * (2 * Qd - Qd[-1]); phi_d = -np.cumsum(Ez_d) * dz
verificar("salto de φ en la capa doble = 4πD", phi_d[-1] - phi_d[0], 4 * np.pi, tol=2e-3)

# %% [markdown]
# ### ¿Qué pasó?
# - **Capa simple** ($\rho\to\sigma\delta$): $E_z$ **salta** en $4\pi\sigma$ y $\phi$ es continuo, aunque tiene un quiebre.
# - **Capa doble** ($\rho\to -D\delta'$): afuera $E_z = 0$, adentro $E_z$ se vuelve un pico cada vez más alto y angosto (una delta), y $\phi$ **salta** en $4\pi D$.
#
# Cada derivada de delta en la fuente "sube" un nivel la singularidad: una delta en $\rho$ produce un salto en $E$; una $\delta'$ produce un salto en $\phi$.
#
# ## Explorá
# 1. **La carga sobre la superficie.** En el Experimento 1 poné la carga justo en el borde ($x_0 = 1.16$ para $A = 0.4$). ¿Cuánto vale el flujo? Explicalo con el ángulo sólido que "ve" una carga sobre una superficie suave.
# 2. **La carga en un vértice.** Una carga $q$ está en un vértice de un cubo. Calculá numéricamente el flujo a través de las tres caras que no tocan la carga, y compará con $4\pi q/8$. ¿Por qué $1/8$?
# 3. **Guía 2, P2.** Una esfera uniformemente cargada tiene un hueco esférico descentrado. Calculá el campo dentro del hueco por superposición (esfera llena + esfera de carga opuesta) y graficalo con `quiver`: ¿es uniforme?
# 4. **Guía 1, P3.** Calculá numéricamente el flujo de $\nabla(1/r) = -\hat{\mathbf r}/r^2$ a través de la papa. ¿Qué relación tiene con $\nabla^2(1/r) = -4\pi\delta^3(\mathbf r)$?
# 5. **Del discreto al continuo con $\sigma$.** Repetí el Experimento 2 con un **disco** formado por anillos concéntricos. ¿A qué distancia de la superficie deja de notarse la granularidad?
