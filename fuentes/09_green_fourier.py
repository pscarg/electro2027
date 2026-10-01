# %% [markdown]
# # Clase 9 — Funciones de Green y funciones ortogonales
#
# **Objetivos**
# - Ver el **fenómeno de Gibbs**: el sobrepico de las sumas parciales de Fourier junto a un salto no baja al agregar términos.
# - Comprobar que, en un plano con franjas a $\pm V_0$, la **serie de Fourier** y la **integral de Green** dan el mismo potencial, y que los detalles se borran como $e^{-kz}$.
# - Verificar la **reciprocidad** $G_D(\mathbf r_1,\mathbf r_2)=G_D(\mathbf r_2,\mathbf r_1)$ en una geometría sin ninguna simetría, y que el peso de cada punto del borde es la carga inducida.
# - Usar la **fórmula de Poisson de la esfera** para los hemisferios a $\pm V_0$.
#
# **Material relacionado:** notas de la Clase 9. Guía 4: problema 5.
#
# **Unidades.** Gaussianas, adimensionales: potenciales en unidades de $V_0$, longitudes en unidades del semiperíodo $a$ o del radio $R$.

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
from scipy.special import sici
from scipy.integrate import quad, trapezoid

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
# ## Experimento 1 ★ — El fenómeno de Gibbs
#
# La onda cuadrada ($+1$ en $(0,\pi)$, $-1$ en $(-\pi,0)$) tiene la serie
# $$f(x)=\frac4\pi\left(\sin x+\frac{\sin3x}{3}+\frac{\sin5x}{5}+\cdots\right).$$
# Llamamos $S_K$ a la suma de los primeros $K$ términos.
#
# ### Predecí
# 1. Junto al salto, ¿la suma parcial se pasa de 1? ¿Ese exceso baja al aumentar $K$?
# 2. ¿Qué vale la suma parcial justo en el salto, $x=0$?

# %%
def suma_parcial(x, K):
    n = 2 * np.arange(K) + 1
    return 4 / np.pi * np.sum(np.sin(np.outer(x, n)) / n, axis=1)

def gibbs(K=10):
    x = np.linspace(-np.pi, np.pi, 4001)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    axs[0].plot(x, np.sign(x), color="0.6", lw=1)
    axs[0].plot(x, suma_parcial(x, int(K)), color=COLORES[0])
    axs[0].set(xlabel="x", title=f"suma parcial con K = {int(K)} términos")
    xz = np.linspace(1e-4, 6 * np.pi / (2 * K), 800)
    axs[1].plot(xz, suma_parcial(xz, int(K)), color=COLORES[0])
    axs[1].axhline(1, color="0.6", lw=1); axs[1].axhline(2 / np.pi * sici(np.pi)[0], color=COLORES[1], ls="--", lw=1, label="(2/π) Si(π) = 1.179")
    axs[1].set(xlabel="x", title="cerca del salto", ylim=(0, 1.3)); axs[1].legend()
    fig.tight_layout(); plt.show()

interactuar(gibbs, K=IntSlider(value=10, min=1, max=200, step=1, description="K", continuous_update=False))

maximo_teorico = 2 / np.pi * sici(np.pi)[0]
for K in (10, 100, 1000):
    xs = np.linspace(1e-6, 3 * np.pi / (2 * K), 3001)
    S = suma_parcial(xs, K)
    print(f"K = {K:5d}: máximo = {S.max():.5f} en x = {xs[S.argmax()]:.5f}   (π/2K = {np.pi / (2 * K):.5f})")
verificar("máximo de S_K para K = 1000 = (2/π) Si(π)", S.max(), maximo_teorico, tol=1e-4)
verificar("posición del máximo para K = 1000 = π/2K", xs[S.argmax()], np.pi / 2000, tol=1e-2)
verificar("S_K en el salto (x = 0)", suma_parcial(np.array([0.0]), 1000)[0], 0.0, tol=1e-12)

# %% [markdown]
# **Cuán rápido decaen los coeficientes.** Calculamos numéricamente $b_n=\frac1\pi\int_{-\pi}^{\pi}f(x)\sin nx\,dx$ para la onda cuadrada (con salto) y para una onda triangular (continua, con quiebres).

# %%
x = np.linspace(-np.pi, np.pi, 200001)
cuadrada = np.sign(x)
triangular = np.where(np.abs(x) <= np.pi / 2, x, np.sign(x) * np.pi - x)
ns = np.arange(1, 200, 2)
b_cuad = np.array([trapezoid(cuadrada * np.sin(n * x), x) / np.pi for n in ns])
b_tri = np.array([trapezoid(triangular * np.sin(n * x), x) / np.pi for n in ns])

fig, ax = plt.subplots(figsize=(6.6, 4.4))
ax.loglog(ns, np.abs(b_cuad), "o", ms=3, color=COLORES[0], label="cuadrada (salto)")
ax.loglog(ns, np.abs(b_tri), "s", ms=3, color=COLORES[1], label="triangular (quiebre)")
ax.loglog(ns, 4 / (np.pi * ns), "k--", lw=1, label="4/(nπ)")
ax.set(xlabel="n", ylabel="|bₙ|", title="decaimiento de los coeficientes"); ax.legend()
plt.show()
verificar("pendiente de |bₙ| con salto = −1", np.polyfit(np.log(ns), np.log(np.abs(b_cuad)), 1)[0], -1.0, tol=1e-3)
verificar("pendiente de |bₙ| con quiebre = −2", np.polyfit(np.log(ns), np.log(np.abs(b_tri)), 1)[0], -2.0, tol=1e-3)

# %% [markdown]
# ### ¿Qué pasó?
# El máximo junto al salto vale $1.179$ para $K=10$, $100$ y $1000$: solo se corre hacia el salto, como $\pi/2K$. En el salto, la suma da el promedio, $0$. La serie converge en cada punto fijo, pero no de manera uniforme. Los coeficientes de la onda cuadrada decaen como $1/n$, y los de la triangular como $1/n^2$: el salto es lo que hace lenta la serie y produce el sobrepico.
#
# ## Experimento 2 — Franjas en un plano: Fourier y Green
#
# El plano $z=0$ tiene franjas de ancho $a$ alternadas a $+V_0$ y $-V_0$ (una onda cuadrada de período $2a$ en $x$). Las notas dan dos maneras de obtener el potencial arriba.
# - **Fourier:** cada modo $\sin(n\pi x/a)$ decae como $e^{-n\pi z/a}$, así que
# $$\phi=\sum_{n\ \rm impar}\frac{4V_0}{n\pi}\sin\frac{n\pi x}{a}\,e^{-n\pi z/a}.$$
# - **Green:** la integral $\phi=\int f(x')\,\frac{z\,dx'}{\pi[(x-x')^2+z^2]}$. Cada franja da exactamente $\pm\frac{V_0}{\pi}\left[\arctan\frac{x'-x}{z}\right]$ entre sus bordes.
#
# ### Predecí
# ¿Qué le pasa al sobrepico de Gibbs a una altura $z=0.05a$? ¿Cuánto vale la amplitud a $z=a$?

# %%
def phi_fourier(x, z, nmax=401):
    n = np.arange(1, nmax + 1, 2)
    return np.sum(4 / (n * np.pi) * np.sin(np.outer(x, n) * np.pi) * np.exp(-n * np.pi * z), axis=1)

def phi_green(x, z, M=4000):
    """Suma exacta franja por franja (a = 1): franja j entre j y j+1 con signo (−1)^j."""
    j = np.arange(-M, M)
    signo = np.where(j % 2 == 0, 1.0, -1.0)
    x = np.atleast_1d(x)[:, None]
    return np.sum(signo / np.pi * (np.arctan((j + 1 - x) / z) - np.arctan((j - x) / z)), axis=1)

xs = np.linspace(-1, 1, 801)
fig, ax = plt.subplots(figsize=(7.5, 4.4))
ax.plot(xs, np.sign(np.sin(np.pi * xs)), color="0.6", lw=1, label="z = 0 (dato)")
ax.plot(xs, phi_fourier(xs, 0.0, nmax=61), color=RAMPA[0], label="z = 0, 30 términos")
for z, c in zip([0.05, 0.2, 0.5], RAMPA[1:]):
    ax.plot(xs, phi_fourier(xs, z), color=c, label=f"z = {z}a")
ax.set(xlabel="x / a", ylabel="φ / V₀", title="los detalles se borran con la altura"); ax.legend(fontsize=9, loc="lower left")
plt.show()

verificar("Fourier = Green en (x, z) = (0.3a, 0.2a)", phi_fourier(np.array([0.3]), 0.2)[0], phi_green(0.3, 0.2)[0], tol=1e-6)
verificar("Fourier = Green en (x, z) = (0.9a, 0.05a)", phi_fourier(np.array([0.9]), 0.05, nmax=2001)[0], phi_green(0.9, 0.05)[0], tol=1e-6)
verificar("amplitud a z = a: (4/π) e^{−π}", phi_fourier(np.array([0.5]), 1.0)[0], 4 / np.pi * np.exp(-np.pi), tol=1e-3)

# %% [markdown]
# ### ¿Qué pasó?
# Las dos maneras dan lo mismo: sumar cargas con $G$ o sumar modos. A $z=0.05a$ el sobrepico ya no está, porque los modos altos, que son los que lo forman, decaen primero. A $z=a$ solo sobrevive el modo fundamental, con amplitud $\frac{4}{\pi}e^{-\pi}\simeq0.055$: desde una altura igual al ancho de las franjas, el patrón es casi una sinusoide chiquita.
#
# ## Experimento 3 — Reciprocidad en una caja sin simetrías
#
# Una caja a tierra con dos obstáculos metálicos a tierra, de formas arbitrarias. Calculamos con la relajación de la Clase 7 la función de Green $G_D(\mathbf r,\mathbf r_A)$ (carga unitaria por unidad de longitud en $A$) y $G_D(\mathbf r,\mathbf r_B)$.
#
# ### Predecí
# ¿Es igual el potencial en $B$ con la carga en $A$ que el potencial en $A$ con la carga en $B$?

# %%
N = 101; h = 1 / (N - 1)
x = np.linspace(0, 1, N); X, Y = np.meshgrid(x, x, indexing="ij")
caja = np.zeros((N, N), dtype=bool); caja[0, :] = caja[-1, :] = caja[:, 0] = caja[:, -1] = True
obst = ((X - 0.62)**2 / 0.02 + (Y - 0.35)**2 / 0.006 < 1) | ((np.abs(X - 0.3) < 0.04) & (Y > 0.45) & (Y < 0.85))
fijo = caja | obst
A, B = (20, 30), (80, 75)
omega = 2 / (1 + np.sin(np.pi / N))

def green(fuente):
    rho = np.zeros((N, N)); rho[fuente] = 1 / h**2
    return relajar(np.zeros((N, N)), fijo, rho=rho, h=h, omega=omega, tol=1e-12)[0]

GA, GB = green(A), green(B)
fig, axs = plt.subplots(1, 2, figsize=(11, 4.8))
for ax, G, fuente, otro, nombre in [(axs[0], GA, A, B, "A"), (axs[1], GB, B, A, "B")]:
    ax.contourf(X, Y, np.where(obst, np.nan, np.log10(np.maximum(G, 1e-6))), levels=30, cmap="viridis")
    ax.contourf(X, Y, obst.astype(float), levels=[0.5, 1.5], colors=["0.5"])
    ax.plot(x[fuente[0]], x[fuente[1]], "w*", ms=14); ax.plot(x[otro[0]], x[otro[1]], "wo", ms=8)
    ax.set(aspect="equal", title=f"carga en {nombre}: log₁₀ G"); ax.grid(False)
fig.tight_layout(); plt.show()
print(f"G(B, A) = {GA[B]:.10f}    G(A, B) = {GB[A]:.10f}")
verificar("reciprocidad G(B, A) = G(A, B)", GA[B], GB[A], tol=1e-7)

# %% [markdown]
# **El peso de cada punto del borde.** Ahora resolvemos Laplace sin cargas, con la pared de arriba a potencial $1$ y todo lo demás a tierra. Según las notas, $\phi(A)$ es la carga que induciría en esa pared una carga unitaria en $A$ (cambiada de signo). En la grilla, la carga inducida en un nodo de borde es $G_D/4\pi$ evaluada en su vecino interior.

# %%
datos = np.zeros((N, N)); datos[:, -1] = 1.0
u = relajar(datos, fijo, h=h, omega=omega, tol=1e-12)[0]
inducida_arriba = np.sum(GA[1:-1, -2]) / (4 * np.pi)
# carga inducida total: todos los pares (nodo libre, vecino fijo)
libre = ~fijo
total = 0.0
for desplazamiento in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
    vecino_fijo = np.roll(fijo, desplazamiento, axis=(0, 1))
    total += np.sum(GA[libre & vecino_fijo])
total /= 4 * np.pi
print(f"φ(A) = {u[A]:.8f}   carga inducida en la pared de arriba (cambiada de signo) = {inducida_arriba:.8f}")
verificar("φ(A) = −(carga inducida en la pared por una carga unitaria en A)", u[A], inducida_arriba, tol=1e-6)
verificar("carga inducida total = −1", total, 1.0, tol=1e-6)

# %% [markdown]
# ### ¿Qué pasó?
# Los dos valores coinciden con la precisión del método, aunque la caja no tiene ninguna simetría que los relacione. Y el potencial en $A$ debido a la pared de arriba es exactamente la fracción de la carga inducida por una carga unitaria en $A$ que cae en esa pared: el peso de cada punto del borde en la fórmula de Green. Como toda la carga inducida suma $-1$, los pesos suman 1: el potencial es un promedio de los valores de borde.
#
# ## Experimento 4 — Hemisferios a $\pm V_0$
#
# Una esfera de radio $R=1$ con el hemisferio de arriba a $+V_0$ y el de abajo a $-V_0$. Adentro, la fórmula de Poisson de las notas,
# $$\phi(\mathbf r)=\frac{R(R^2-r^2)}{4\pi}\oint\frac{f(\Omega')\,d\Omega'}{(r^2+R^2-2rR\cos\gamma)^{3/2}},$$
# la calculamos con muchas direcciones casi uniformes (la espiral de Fibonacci de la Clase 6).
#
# ### Predecí
# ¿Cuánto vale $\phi$ en el centro? ¿Cómo es el campo cerca del centro?

# %%
def direcciones(M=400):
    """M direcciones casi uniformes (espiral de Fibonacci)."""
    i = np.arange(M) + 0.5
    th = np.arccos(1 - 2 * i / M); ph = np.pi * (1 + 5**0.5) * i
    return np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], axis=1)

esfera = direcciones(40000)
f_borde = np.sign(esfera[:, 2])

def poisson_esfera(puntos, f=f_borde, R=1.0):
    """φ adentro de la esfera (puntos: M × 3) con valores de borde f en las direcciones `esfera`."""
    salida = np.empty(len(puntos))
    for i in range(0, len(puntos), 100):                              # de a 100 puntos, para no llenar la memoria
        p = puntos[i:i + 100]
        D = np.linalg.norm(p[:, None, :] - R * esfera[None, :, :], axis=2)
        salida[i:i + 100] = R * (R**2 - np.sum(p**2, axis=1)) * np.mean(f[None, :] / D**3, axis=1)   # ∮ dΩ/4π = promedio
    return salida

zs = np.linspace(-0.85, 0.85, 35)
eje = np.stack([np.zeros_like(zs), np.zeros_like(zs), zs], axis=1)
phi_eje = poisson_esfera(eje)
zz = np.linspace(1e-3, 0.95, 300)
formula = (1 / zz) * (1 - (1 - zz**2) / np.sqrt(1 + zz**2))

rr, tt = np.meshgrid(np.linspace(0, 0.97, 28), np.linspace(0, 2 * np.pi, 73), indexing="ij")   # grilla polar en el corte
Xs, Zs = rr * np.sin(tt), rr * np.cos(tt)
mapa = poisson_esfera(np.stack([Xs.ravel(), 0 * Xs.ravel(), Zs.ravel()], axis=1)).reshape(Xs.shape)

fig, axs = plt.subplots(1, 2, figsize=(12, 4.8))
im = axs[0].contourf(Xs, Zs, mapa, levels=np.linspace(-1, 1, 21), cmap="RdBu_r")
axs[0].add_patch(plt.Circle((0, 0), 1, fill=False, color="k", lw=1.5))
axs[0].set(aspect="equal", xlabel="x", ylabel="z", title="φ en un corte meridional"); axs[0].grid(False)
fig.colorbar(im, ax=axs[0])
axs[1].plot(zs, phi_eje, "o", ms=4, color=COLORES[0], label="Poisson numérico")
axs[1].plot(zz, formula, "k", lw=1.2, label="fórmula de las notas"); axs[1].plot(-zz, -formula, "k", lw=1.2)
axs[1].plot(zz, 1.5 * zz, ":", color=COLORES[1], label="3z/2R")
axs[1].set(xlabel="z", ylabel="φ / V₀", title="sobre el eje", ylim=(-1.1, 1.1)); axs[1].legend()
plt.show()

z0 = 0.5
verificar("φ en el eje (z = 0.5R) = fórmula", poisson_esfera(np.array([[0, 0, z0]]))[0],
          (1 / z0) * (1 - (1 - z0**2) / np.sqrt(1 + z0**2)), tol=1e-4)
verificar("φ en el centro = promedio de los valores de borde", poisson_esfera(np.zeros((1, 3)))[0], np.mean(f_borde), tol=1e-12)
dz = 1e-3
Ez = -(poisson_esfera(np.array([[0, 0, dz]]))[0] - poisson_esfera(np.array([[0, 0, -dz]]))[0]) / (2 * dz)
verificar("campo en el centro E_z = −3V₀/2R", Ez, -1.5, tol=1e-3)

# %% [markdown]
# ### ¿Qué pasó?
# En el centro, $\phi=0$: el promedio de $+V_0$ y $-V_0$, como dice el teorema del valor medio. Cerca del centro el campo es casi uniforme, $E_z=-3V_0/2R$, más intenso que el de dos placas a $\pm V_0$ separadas $2R$ ($V_0/R$). Sobre el eje, la integral numérica sigue la fórmula de las notas, que llega a $\pm V_0$ en los polos.
#
# ## Explorá
# 1. **Guía 4, P5.** Escribí la integral de Green del semiespacio para un disco de radio $a$ a potencial $V$ (en coordenadas polares centradas en el disco), calculala numéricamente sobre el eje y fuera de él, y comparala con tus resultados de los ítems c) y d).
# 2. **Curar el sobrepico.** Multiplicá cada coeficiente de la onda cuadrada por el factor de Lanczos $\sin(\pi n/N)/(\pi n/N)$ y mirá qué le pasa al sobrepico. ¿Qué se pierde a cambio?
# 3. **Una sola franja.** El potencial de una franja de ancho $w$ a $V_0$ (el resto del plano a tierra) es la diferencia de dos medios planos. Escribilo con la fórmula del medio plano de las notas y verificá con la integral de Green numérica. ¿A qué altura deja de "verse" el ancho de la franja?
# 4. **Afuera de la esfera.** Usá la fórmula de Poisson exterior de las notas para los hemisferios a $\pm V_0$ y calculá el potencial sobre el eje para $r\gg R$. ¿Qué multipolo domina? Estimá su momento.
