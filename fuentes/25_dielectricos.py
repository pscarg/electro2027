# %% [markdown]
# # Clase 25 — Dieléctricos: polarización, D y condiciones de borde
#
# **Objetivos**
# - Ver cómo se refractan las líneas de campo en la interfaz entre dos dieléctricos, con un resolvedor numérico y con imágenes parciales.
# - Comprobar con una red de moléculas polarizables el campo local de Lorentz y la relación de Clausius–Mossotti.
# - Dibujar las líneas de **E** y de **D** de una esfera dieléctrica en un campo uniforme, y ver dónde empiezan y terminan.
#
# **Material relacionado:** notas de la Clase 25. Guía 10: problemas 1 a 4.
#
# **Unidades.** Gaussianas, adimensionales (cargas y longitudes de orden 1).

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

# %% [markdown]
# ## La herramienta
#
# `resolver_conduccion` (del notebook de la Clase 15) resuelve $\nabla\cdot(\kappa\nabla\phi)=-f$ en una grilla 2D, con $\kappa$ en las caras entre nodos igual a la media armónica. Para un dieléctrico sin cargas libres en la interfaz, $\nabla\cdot\mathbf D=4\pi\rho_{\text{libre}}$ con $\mathbf D=-\varepsilon\nabla\phi$ es exactamente esa ecuación, con $\kappa=\varepsilon$ y $f=4\pi\rho_{\text{libre}}$. La media armónica es la combinación correcta cuando la interfaz pasa por la mitad entre dos nodos: dos capas en serie (sección 8 de las notas).

# %%
# herramienta: conduccion v1 (NB15)
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

def _caras(kappa):
    """κ en las caras entre nodos vecinos (media armónica): (κx entre (i,j) e (i+1,j), κy entre (i,j) e (i,j+1))."""
    def armonica(a, b):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(a + b > 0, 2 * a * b / (a + b), 0.0)
    return armonica(kappa[:-1, :], kappa[1:, :]), armonica(kappa[:, :-1], kappa[:, 1:])

def divergencia_flujo(kappa, phi, h=1.0):
    """∇·(κ∇φ) en cada nodo, con flujo nulo a través de los bordes de la grilla."""
    kx, ky = _caras(kappa)
    fx = kx * (phi[1:, :] - phi[:-1, :]); fy = ky * (phi[:, 1:] - phi[:, :-1])
    d = np.zeros_like(phi)
    d[:-1, :] += fx; d[1:, :] -= fx; d[:, :-1] += fy; d[:, 1:] -= fy
    return d / h**2

def resolver_conduccion(kappa, fijo, valores, fuente=None, h=1.0):
    """Resuelve ∇·(κ∇φ) = −fuente en una grilla 2D (diferencias finitas, κ en las caras = media armónica).

    fijo: nodos con φ dado (Dirichlet), con los valores de `valores`.
    Los bordes de la grilla que no son fijos tienen flujo nulo (un aislante).
    """
    Nx, Ny = kappa.shape
    libres = ~fijo; n = int(libres.sum())
    idx = -np.ones((Nx, Ny), dtype=int); idx[libres] = np.arange(n)
    kx, ky = _caras(kappa)
    b = np.zeros(n) if fuente is None else -fuente[libres] * h**2
    filas, cols, vals = [], [], []
    diag = np.zeros(n)
    for (k, a_sl, b_sl) in [(kx, (slice(None, -1), slice(None)), (slice(1, None), slice(None))),
                            (ky, (slice(None), slice(None, -1)), (slice(None), slice(1, None)))]:
        for (p_sl, q_sl) in [(a_sl, b_sl), (b_sl, a_sl)]:
            ip, iq = idx[p_sl], idx[q_sl]
            lp = libres[p_sl]
            kk = k[lp]; ipp = ip[lp]; iqq = iq[lp]
            np.add.at(diag, ipp, -kk)
            vecino_libre = iqq >= 0
            filas.append(ipp[vecino_libre]); cols.append(iqq[vecino_libre]); vals.append(kk[vecino_libre])
            np.add.at(b, ipp[~vecino_libre], -kk[~vecino_libre] * valores[q_sl][lp][~vecino_libre])
    filas.append(np.arange(n)); cols.append(np.arange(n)); vals.append(diag)
    A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(filas), np.concatenate(cols))), shape=(n, n))
    phi = np.array(valores, dtype=float)
    phi[libres] = spsolve(A, b)
    return phi
# fin herramienta

# %% [markdown]
# ## Experimento 1 ★ — Una carga lineal frente a un dieléctrico
#
# Una carga lineal $\lambda=1$ (perpendicular al plano del dibujo) está en $(0,d)$, con $d=1$, en el medio $\varepsilon_1=1$ ($y>0$); abajo ($y<0$) hay un dieléctrico $\varepsilon_2$. Las notas (sección 6) dan la solución con imágenes parciales:
# $$\phi_1=-\frac{2}{\varepsilon_1}\left(\lambda\ln s_1+\lambda'\ln s_2\right),\qquad\phi_2=-\frac{2\lambda''}{\varepsilon_2}\ln s_1,\qquad\lambda'=\frac{\varepsilon_1-\varepsilon_2}{\varepsilon_1+\varepsilon_2}\lambda,\quad\lambda''=\frac{2\varepsilon_2}{\varepsilon_1+\varepsilon_2}\lambda,$$
# con $s_1$ y $s_2$ las distancias a $(0,d)$ y a $(0,-d)$. Las líneas de $\mathbf D$ son curvas de nivel de la función de flujo $\psi$, con $\mathbf D=\hat{\mathbf z}\times\nabla\psi$ (la versión plana de la de la Clase 3), que es continua en la interfaz. Resolvemos también el problema numéricamente, con $\varepsilon(y)$ en la grilla y los valores exactos en el borde, y dibujamos las equipotenciales numéricas.
#
# ### Predecí
# Si $\varepsilon_2=4$, las líneas que cruzan hacia abajo, ¿se acercan a la normal o se alejan? ¿Y el campo $\mathbf D$ arriba, entre la carga y la interfaz: es el mismo que sin dieléctrico?

# %%
d = 1.0
h = d / 19.5                                   # la interfaz y = 0 queda a mitad de camino entre dos filas de nodos
Ny = 2 * int(round(5 / h)); Nx = Ny + 1
xg = (np.arange(Nx) - (Nx - 1) / 2) * h; yg = (np.arange(Ny) - Ny / 2 + 0.5) * h
Xg, Yg = np.meshgrid(xg, yg, indexing="ij")
i0, j0 = (Nx - 1) // 2, int(np.argmin(abs(yg - d)))   # la carga, en un nodo

def coeficientes(e1, e2, lam=1.0):
    return lam * (e1 - e2) / (e1 + e2), 2 * e2 * lam / (e1 + e2)

def imagenes_lineal(X, Y, e1, e2, lam=1.0):
    """φ de una carga lineal λ en (0, d) en el medio ε1 (y > 0), con ε2 en y < 0 (imágenes parciales)."""
    lp, lpp = coeficientes(e1, e2, lam)
    s1 = np.hypot(X, Y - d); s2 = np.hypot(X, Y + d)
    with np.errstate(divide="ignore"):
        return np.where(Y > 0, -2 * (lam * np.log(s1) + lp * np.log(s2)) / e1, -2 * lpp * np.log(s1) / e2)

def campo_lineal(X, Y, e1, e2, lam=1.0):
    """E de la solución con imágenes: E = (2/ε) Σ λ_k (r − r_k)/s_k²."""
    lp, lpp = coeficientes(e1, e2, lam)
    s1c = X**2 + (Y - d)**2; s2c = X**2 + (Y + d)**2
    arriba = (2 / e1) * np.array([lam * X / s1c + lp * X / s2c, lam * (Y - d) / s1c + lp * (Y + d) / s2c])
    abajo = (2 / e2) * lpp * np.array([X / s1c, (Y - d) / s1c])
    return np.where(Y > 0, arriba, abajo)

def flujo_D_lineal(X, Y, e1, e2, lam=1.0):
    """Función de flujo de D: las líneas de D son sus curvas de nivel. Los ángulos se miden con el corte
    hacia arriba desde la carga y hacia abajo desde la imagen, fuera de la región donde se usa cada uno."""
    lp, lpp = coeficientes(e1, e2, lam)
    th1 = np.arctan2(X, -(Y - d)); th2 = np.arctan2(-X, Y + d)
    return np.where(Y > 0, -2 * (lam * th1 + lp * th2), -2 * lpp * th1)

def resolver_interfaz(e1, e2, lam=1.0):
    eps = np.where(Yg > 0, e1, e2)
    fijo = np.zeros((Nx, Ny), bool); fijo[[0, -1], :] = True; fijo[:, [0, -1]] = True
    val = np.where(fijo, imagenes_lineal(Xg, Yg, e1, e2, lam), 0.0)
    f = np.zeros((Nx, Ny)); f[i0, j0] = 4 * np.pi * lam / h**2
    return resolver_conduccion(eps, fijo, val, fuente=f, h=h)

def dibujar_interfaz(eps2=4.0, exportar=False):
    e1 = 1.0
    phi = resolver_interfaz(e1, eps2)
    sel = (abs(Xg) < 3.2) & (abs(Yg) < 3.2)
    xs, ys = xg[abs(xg) < 3.2], yg[abs(yg) < 3.2]
    P = phi[sel].reshape(len(xs), len(ys))
    X, Y = np.meshgrid(np.linspace(-3, 3, 601), np.linspace(-3, 3, 600), indexing="ij")
    psi = np.ma.array(flujo_D_lineal(X, Y, e1, eps2), mask=(abs(X) < 0.02) & (Y > d))   # sin el corte de θ₁
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    ax.axhspan(-3, 0, color=COLORES[2], alpha=0.12)
    ax.contour(X, Y, psi, levels=np.linspace(-4 * np.pi, 4 * np.pi, 49), colors=[COLORES[0]], linewidths=0.9, linestyles="solid")
    ax.contour(xs, ys, P.T, levels=np.linspace(np.percentile(P, 3), np.percentile(P, 97), 14), colors=[COLORES[1]], linewidths=0.8, linestyles="dashed")
    ax.plot([0], [d], "o", color=COLORES[1])
    ax.set(aspect="equal", xlim=(-3, 3), ylim=(-3, 3), xlabel="x / d", ylabel="y / d",
           title=f"líneas de D (azul) y equipotenciales numéricas (naranja), ε₂/ε₁ = {eps2:g}")
    ax.text(2.0, -2.7, "ε₂", fontsize=14); ax.text(2.0, 2.5, "ε₁", fontsize=14)
    ax.grid(False)
    if exportar:
        guardar(fig, "nb25_interfaz")
    plt.show()

interactuar(dibujar_interfaz, eps2=deslizador("ε₂/ε₁", 4.0, 0.1, 10.0, 0.1))
if CARPETA_FIGURAS:
    dibujar_interfaz(4.0, exportar=True)

# Comprobaciones
for e2 in (4.0, 0.25):
    phi = resolver_interfaz(1.0, e2); exacto = imagenes_lineal(Xg, Yg, 1.0, e2)
    lejos = np.hypot(Xg, Yg - d) > 0.5
    verificar(f"ε₂ = {e2}: φ numérico = imágenes parciales (error / rango de φ)", np.max(np.abs(phi - exacto)[lejos]) / np.ptp(exacto[lejos]), 0.0, tol=2e-3)
xi = np.array([0.4, 1.3, 2.5]); e2 = 4.0
E1 = campo_lineal(xi, np.full(3, 1e-9), 1.0, e2); E2 = campo_lineal(xi, np.full(3, -1e-9), 1.0, e2)
verificar("refracción: tan θ₁ / tan θ₂ = ε₁/ε₂ en la interfaz (θ medido desde la normal)", np.max(np.abs((E1[0] / E1[1]) / (E2[0] / E2[1]) - 1 / e2)), 0.0, tol=1e-6)

# %% [markdown]
# ### ¿Qué pasó?
# Las equipotenciales numéricas, calculadas sin saber nada de imágenes, coinciden con la solución de las notas. Al cruzar hacia un medio de mayor $\varepsilon$, las líneas se **alejan** de la normal: $\tan\theta_2=\frac{\varepsilon_2}{\varepsilon_1}\tan\theta_1$, porque $E_t$ se conserva y $E_n$ se divide por $\varepsilon_2/\varepsilon_1$. Las líneas de $\mathbf D$ se juntan en el dieléctrico, que las ``atrae''; con $\varepsilon_2/\varepsilon_1<1$ pasa lo contrario. Y arriba, entre la carga y la interfaz, $\mathbf D$ **cambia** con $\varepsilon_2$, aunque la carga libre sea la misma: $\nabla\cdot\mathbf D=4\pi\rho_{\text{libre}}$ no alcanza para fijar $\mathbf D$, porque $\nabla\times\mathbf D\neq0$ en la interfaz. Para $\varepsilon_2\to\infty$ la imagen es $-\lambda$: el plano conductor de la Clase 8.
#
# ## Experimento 2 — ¿Qué campo siente una molécula?
#
# Una esfera de radio $R=6$ hecha de moléculas en los nodos de una red cúbica de paso 1 (una por celda, $n=1$), cada una con polarizabilidad $\alpha$, en un campo uniforme $E_0\hat{\mathbf z}$. Cada molécula se polariza con el campo de las demás: $\mathbf p_i=\alpha\left(\mathbf E_0+\sum_{j\neq i}\frac{3(\mathbf p_j\cdot\hat{\mathbf u}_{ij})\hat{\mathbf u}_{ij}-\mathbf p_j}{u_{ij}^3}\right)$, con $\mathbf u_{ij}=\mathbf r_i-\mathbf r_j$. Es un sistema lineal de $3N$ ecuaciones, que resolvemos exactamente. Las notas predicen (secciones 4 y 7) que el campo local es $\mathbf E+\frac{4\pi}{3}\mathbf P$, y que dentro de una esfera eso da exactamente $\mathbf E_0$: $\mathbf p=\alpha\mathbf E_0$. Si la molécula sintiera solo el campo macroscópico, sería $\mathbf p=\alpha\mathbf E_0/(1+\frac{4\pi n\alpha}{3})$.
#
# ### Predecí
# Con $\frac{4\pi n\alpha}{3}=0.25$, ¿las moléculas del interior tienen $p=\alpha E_0$ o $p=0.8\,\alpha E_0$?

# %%
def red_esferica(R):
    """Sitios de una red cúbica simple de paso 1 dentro de una esfera de radio R centrada en un sitio."""
    m = int(np.ceil(R)); g = np.arange(-m, m + 1)
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    r = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1).astype(float)
    return r[np.sum(r**2, axis=1) <= R**2]

def tensor_dipolar(r):
    """T[i, j] (matriz 3×3): el campo en r_i del dipolo p_j es T[i, j] @ p_j; T[i, i] = 0."""
    u = r[:, None, :] - r[None, :, :]
    dist = np.linalg.norm(u, axis=-1); np.fill_diagonal(dist, np.inf)
    uh = u / dist[..., None]
    return (3 * uh[..., :, None] * uh[..., None, :] - np.eye(3)) / dist[..., None, None]**3

def dipolos_acoplados(r, alpha, E0):
    """Resuelve p_i = α (E0 + Σ_j T_ij p_j) para todos los sitios a la vez."""
    N = len(r)
    A = np.eye(3 * N) - alpha * tensor_dipolar(r).transpose(0, 2, 1, 3).reshape(3 * N, 3 * N)
    return np.linalg.solve(A, alpha * np.tile(E0, N)).reshape(N, 3)

r_red = red_esferica(6.0)
E0 = np.array([0.0, 0.0, 1.0])

def mostrar_red(x=0.25):
    alpha = 3 * x / (4 * np.pi)
    p = dipolos_acoplados(r_red, alpha, E0)
    rr = np.linalg.norm(r_red, axis=1)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(rr, p[:, 2] / alpha, ".", color=COLORES[0], ms=4, label="cada molécula")
    ax.axhline(1, color=COLORES[2], label="campo local de Lorentz: p = αE₀")
    ax.axhline(1 / (1 + x), color=COLORES[1], ls="--", label="solo el campo macroscópico")
    ax.set(xlabel="distancia al centro (pasos de red)", ylabel="p_z / αE₀", title=f"{len(r_red)} moléculas, 4πnα/3 = {x:g}")
    ax.legend(); plt.show()

interactuar(mostrar_red, x=deslizador("4πnα/3", 0.25, 0.05, 0.6, 0.05))

# Comprobaciones
centro = np.argmin(np.linalg.norm(r_red, axis=1))
Tc = tensor_dipolar(r_red)[centro]
verificar("suma de Lorentz: el campo de una red cúbica de dipolos iguales en el centro es cero", np.linalg.norm(np.einsum("jab,b->a", Tc, E0)), 0.0, tol=1e-12)
x = 0.25; alpha = 3 * x / (4 * np.pi)
p = dipolos_acoplados(r_red, alpha, E0)
interior = np.linalg.norm(r_red, axis=1) < 3.0
pz = p[interior, 2].mean()
verificar("en una esfera, cada molécula del interior siente E₀: p = αE₀", pz / alpha, 1.0, tol=2e-2)
P = pz; E_in = 1 - 4 * np.pi * P / 3                    # n = 1; campo macroscópico adentro: E₀ − 4πP/3
verificar("ε = 1 + 4πP/E de la red = Clausius–Mossotti (1 + 2x)/(1 − x)", 1 + 4 * np.pi * P / E_in, (1 + 2 * x) / (1 - x), tol=2e-2)

# %% [markdown]
# ### ¿Qué pasó?
# Las moléculas del interior tienen $p\simeq\alpha E_0$ (la diferencia de menos del 1% viene de que una esfera hecha de cubitos no es una esfera), lejos de $0.8\,\alpha E_0$: la molécula no siente el campo macroscópico $\mathbf E$ sino el campo local $\mathbf E+\frac{4\pi}{3}\mathbf P$. Con $\mathbf P$ y el campo macroscópico de adentro se obtiene $\varepsilon\simeq2.0$, el valor de Clausius–Mossotti, y no el $1+4\pi n\alpha=1.75$ ingenuo. La suma de Lorentz es cero por la simetría cúbica: los vecinos cercanos no aportan nada en el centro. Cerca de la superficie, la esfera de cubitos tiene esquinas, y allí las moléculas se apartan del valor ideal.
#
# ## Experimento 3 — Las líneas de E y de D de una esfera dieléctrica
#
# Una esfera de radio $a=1$ y permitividad $\varepsilon$ en un campo $E_0\hat{\mathbf z}$ (sección 7 de las notas): adentro, $\phi=-\frac{3E_0}{\varepsilon+2}r\cos\theta$; afuera, $\phi=-E_0r\cos\theta+\frac{\varepsilon-1}{\varepsilon+2}\frac{a^3E_0\cos\theta}{r^2}$. Con la función de flujo de la Clase 3, $\partial_\theta\Psi=-E_rr^2\sin\theta$: un campo uniforme da $\Psi=-\frac12E_0r^2\sin^2\theta$ y un dipolo, $\Psi=-\frac{p}{r}\sin^2\theta$. Para $\mathbf D$, adentro se multiplica por $\varepsilon$.
#
# ### Predecí
# Con $\varepsilon=4$, ¿entran a la esfera más líneas de $\mathbf E$ que las que le llegarían sin ella, o menos? ¿Y de $\mathbf D$?

# %%
def esfera_dielectrica(r, th, eps, E0=1.0, a=1.0):
    """φ y las funciones de flujo de E y de D de la esfera dieléctrica en un campo uniforme."""
    Ein = 3 * E0 / (eps + 2); p = (eps - 1) / (eps + 2) * a**3 * E0
    s2 = np.sin(th)**2
    phi = np.where(r < a, -Ein * r * np.cos(th), -E0 * r * np.cos(th) + p * np.cos(th) / r**2)
    PsiE = np.where(r < a, -0.5 * Ein * r**2 * s2, -0.5 * E0 * r**2 * s2 - p * s2 / r)
    PsiD = np.where(r < a, eps * PsiE, PsiE)
    return phi, PsiE, PsiD

def mostrar_esfera(eps=4.0, exportar=False):
    X, Z = np.meshgrid(np.linspace(-2, 2, 501), np.linspace(-2, 2, 501), indexing="ij")
    r = np.hypot(X, Z); th = np.arccos(np.clip(Z / np.maximum(r, 1e-12), -1, 1))
    _, PsiE, PsiD = esfera_dielectrica(r, th, eps)
    niveles = np.sort(-(np.arange(0, 2.2, 0.125) + 0.0625))   # igual flujo entre líneas vecinas
    fig, axs = plt.subplots(1, 2, figsize=(11, 5.4))
    for ax, Psi, nombre in zip(axs, (PsiE, PsiD), ("E", "D")):
        ax.add_patch(plt.Circle((0, 0), 1, color=COLORES[2], alpha=0.15))
        for region in (r < 1, r >= 1):                    # por separado: E salta en la superficie
            ax.contour(X, Z, np.ma.array(Psi, mask=~region), levels=niveles, colors=[COLORES[0]], linewidths=0.9, linestyles="solid")
        ax.set(aspect="equal", xlabel="x / a", ylabel="z / a", title=f"líneas de {nombre}, ε = {eps:g}")
        ax.grid(False)
    for z, signo in ((1.08, "+"), (-1.16, "−")):
        axs[0].text(0, z, signo, ha="center", fontsize=14, color=COLORES[1])
    plt.tight_layout()
    if exportar:
        guardar(fig, "nb25_esfera")
    plt.show()

interactuar(mostrar_esfera, eps=deslizador("ε", 4.0, 1.0, 20.0, 0.5))
if CARPETA_FIGURAS:
    mostrar_esfera(4.0, exportar=True)

# Comprobaciones: las condiciones de borde en r = a, y el dipolo de las cargas de polarización
eps = 4.0; th = np.linspace(0, np.pi, 2001); dr = 1e-6
Er_in = -(esfera_dielectrica(1 - dr / 2, th, eps)[0] - esfera_dielectrica(1 - 3 * dr / 2, th, eps)[0]) / dr
Er_out = -(esfera_dielectrica(1 + 3 * dr / 2, th, eps)[0] - esfera_dielectrica(1 + dr / 2, th, eps)[0]) / dr
verificar("D_r continuo en r = a: ε E_r(adentro) = E_r(afuera)", np.max(np.abs(eps * Er_in - Er_out)), 0.0, tol=1e-4)
sigma = (Er_out - Er_in) / (4 * np.pi)                   # σ_pol del salto de E_r (no hay carga libre)
p_sigma = trapezoid(sigma * np.cos(th) * 2 * np.pi * np.sin(th), th)
verificar("el dipolo de σ_pol = (ε − 1)/(ε + 2) a³E₀, el coeficiente del potencial de afuera", p_sigma, (eps - 1) / (eps + 2), tol=1e-4)

# %% [markdown]
# ### ¿Qué pasó?
# Las líneas de $\mathbf D$ son continuas: no hay carga libre, y se juntan dentro de la esfera, con $D_{\text{adentro}}=\frac{3\varepsilon}{\varepsilon+2}E_0>E_0$. Las de $\mathbf E$, en cambio, son menos adentro ($E_{\text{adentro}}=\frac{3}{\varepsilon+2}E_0<E_0$): varias terminan en las cargas de polarización $\sigma_{\text{pol}}=P\cos\theta$, positivas arriba y negativas abajo, y salen de ellas del otro lado. Esas cargas producen adentro un campo uniforme $-\frac{4\pi}{3}\mathbf P$ que se opone a $\mathbf E_0$, y afuera el campo de un dipolo. Para $\varepsilon\to\infty$, $\mathbf E$ adentro se anula y afuera queda la esfera conductora de la Clase 11.
#
# ## Explorá
#
# 1. **Guía 10, P1.** Escribí las imágenes de la carga puntual (en 3D) y dibujá las líneas de $\mathbf D$ con la función de flujo axial: para una carga $q$ sobre el eje, $\Psi=q\cos\vartheta$, con $\vartheta$ el ángulo medido desde esa carga (Clase 3). Comprobá numéricamente, como en el Experimento 1, que $\phi$ y $D_n$ son continuos en $z=0$.
# 2. **Guía 10, P2.** Con `resolver_conduccion`, armá el capacitor con $\varepsilon(x)=1+Ax/a$: placas fijas a $\phi=0$ y $\phi=V$, y los otros dos bordes aislantes (un capacitor sin efectos de borde). Calculá la carga libre de una placa con $D_n$, la capacidad para varios $A$, y la densidad de carga de polarización $-\nabla\cdot\mathbf P$ con `divergencia_flujo`. Comparalas con tus resultados.
# 3. **Guía 10, P3, en 2D.** Con `resolver_conduccion`, un cilindro dieléctrico hueco (radios $r_1<r_2$) en un campo uniforme ($\phi=-E_0x$ en los bordes de la grilla). Medí el campo en el hueco: ¿es uniforme? ¿Cómo depende de $r_1/r_2$ y de $\varepsilon$? El cilindro no es la esfera, pero el comportamiento en los límites ($r_1\to0$, $r_1\to r_2$, $\varepsilon\to\infty$) te sirve para controlar tu resultado.
# 4. **Una red no cúbica.** Con `red_esferica` como modelo, armá una red estirada en $z$ (paso $1.3$ en $z$ y $1$ en $x$ e $y$) y calculá la suma de Lorentz en el centro para dipolos según $\hat{\mathbf z}$ y según $\hat{\mathbf x}$. ¿Sigue siendo cero? ¿Qué implica para $\varepsilon$ de un material con moléculas esféricas en esa red?
