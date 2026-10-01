# %% [markdown]
# # Clase 10 — Separación de variables en cartesianas y cilíndricas
#
# **Objetivos**
# - Comparar la serie de Fourier de una caja con una tapa a $V_0$ con la relajación de la Clase 7, y ver cómo los **detalles del borde se borran exponencialmente**.
# - Comprobar con la serie doble el resultado sin cuentas de las notas: el centro de un cubo con una cara a $V_0$ está a $V_0/6$.
# - Medir cómo se comporta el campo cerca de un **rincón** y de una **arista**: $|\mathbf E|\propto\rho^{\pi/\beta-1}$.
# - Conocer la función de Bessel $J_0$ y resolver un **cilindro con tapa**.
#
# **Material relacionado:** notas de la Clase 10. Guía 4: problemas 6 y 7.
#
# **Unidades.** Gaussianas, adimensionales: potenciales en unidades de $V_0$, longitudes en unidades del ancho de la caja o del radio del cilindro.

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
from scipy.special import j0, j1, jn_zeros, factorial
from scipy.integrate import trapezoid

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
# ## Experimento 1 ★ — Una caja con una tapa a potencial
#
# Una caja bidimensional de ancho $a=1$ y altura $b$, con las paredes y el fondo a tierra y la tapa $y=b$ a potencial $V(x)$. La separación de variables da
# $$\phi=\sum_n b_n\sin\frac{n\pi x}{a}\,\frac{\sinh(n\pi y/a)}{\sinh(n\pi b/a)},\qquad b_n=\frac2a\int_0^aV(x)\sin\frac{n\pi x}{a}\,dx .$$
# Para una tapa uniforme, $b_n=4V_0/n\pi$ con $n$ impar. Comparamos con la relajación, que no sabe nada de senos.
#
# ### Predecí
# 1. En una caja cuadrada, ¿cuánto vale el potencial en el centro?
# 2. En una caja alta ($b=3a$), ¿cómo decae el potencial hacia el fondo? ¿Y si la tapa tiene un patrón fino, $V_0\sin(5\pi x/a)$?

# %%
def caja_serie(X, Y, b, coef, nmax=199):
    """φ en la caja [0,1]×[0,b] con la tapa en y = b; coef(n) da bₙ."""
    phi = np.zeros_like(X, dtype=float)
    for n in range(1, nmax + 1):
        c = coef(n)
        if c != 0:
            phi += c * np.sin(n * np.pi * X) * np.exp(n * np.pi * (Y - b)) * (1 - np.exp(-2 * n * np.pi * Y)) / (1 - np.exp(-2 * n * np.pi * b))
    return phi                                                 # sinh(nπy)/sinh(nπb) escrito sin desbordes

uniforme = lambda n: 4 / (n * np.pi) if n % 2 else 0.0

# caja cuadrada: serie y relajación
N = 101; x = np.linspace(0, 1, N); h = x[1] - x[0]
X, Y = np.meshgrid(x, x, indexing="ij")
fijo = np.zeros((N, N), dtype=bool); fijo[0, :] = fijo[-1, :] = fijo[:, 0] = fijo[:, -1] = True
borde = np.zeros((N, N)); borde[1:-1, -1] = 1.0
phi_rel = relajar(borde, fijo, h=h, omega=2 / (1 + np.sin(np.pi / N)), tol=1e-11)[0]
phi_ser = caja_serie(X, Y, 1.0, uniforme)

fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
for ax, campo_, titulo in [(axs[0], phi_ser, "serie de Fourier"), (axs[1], phi_rel, "relajación")]:
    im = ax.contourf(X, Y, np.clip(campo_, 0, 1), levels=np.linspace(0, 1, 21), cmap="viridis")
    ax.contour(X, Y, campo_, levels=[0.25], colors="w", linewidths=1.5)
    ax.set(aspect="equal", title=titulo + " (blanco: φ = V₀/4)"); ax.grid(False)
fig.colorbar(im, ax=axs[:2], shrink=0.85)

# caja alta: decaimiento hacia el fondo, tapa uniforme y tapa con patrón fino
b = 3.0
yy = np.linspace(0.01, b, 300)
fino = lambda n: 1.0 if n == 5 else 0.0
for coef, c, nombre, xo in [(uniforme, COLORES[0], "tapa uniforme", 0.5), (fino, COLORES[1], "tapa V₀ sin(5πx)", 0.1)]:
    perfil = np.abs(caja_serie(np.full_like(yy, xo), yy, b, coef))
    axs[2].semilogy(b - yy, perfil, color=c, label=f"{nombre} (x = {xo})")
axs[2].semilogy(b - yy, 4 / np.pi * np.exp(-np.pi * (b - yy)), "k--", lw=1, label="(4/π) e^{−π(b−y)}")
axs[2].set(xlabel="distancia a la tapa, b − y", ylabel="|φ|", ylim=(1e-10, 2), title="caja alta (b = 3)"); axs[2].legend(fontsize=8)
guardar(fig, "nb10_caja"); plt.show()

verificar("centro de la caja cuadrada, serie = V₀/4", caja_serie(np.array(0.5), np.array(0.5), 1.0, uniforme), 0.25, tol=1e-6)
verificar("centro de la caja cuadrada, relajación = V₀/4", phi_rel[N // 2, N // 2], 0.25, tol=2e-3)
lejos = (b - yy > 1.0) & (b - yy < 2.0)               # lejos de la tapa (modos altos) y del fondo (el sinh)
pend = np.polyfit((b - yy)[lejos], np.log(np.abs(caja_serie(np.full_like(yy, 0.5), yy, b, uniforme)))[lejos], 1)[0]
verificar("decaimiento hacia el fondo: −π/a", pend, -np.pi, tol=3e-3)

# %% [markdown]
# ### ¿Qué pasó?
# La serie y la relajación dan el mismo potencial, y en el centro de la caja cuadrada valen $V_0/4$ (cuatro lados, misma contribución por simetría). En la caja alta, a una distancia de la tapa mayor que $\sim a/2$, el potencial cae como $e^{-\pi(b-y)/a}$: solo queda el modo más suave. El patrón fino, $\sin(5\pi x/a)$, cae cinco veces más rápido. **Los detalles del borde se curan exponencialmente**, con una longitud igual a su tamaño dividido por $\pi$.
#
# Mové el deslizador para cambiar la altura de la caja y mirar cuánto llega del potencial de la tapa al fondo.

# %%
def caja_alta(b=2.0):
    xs = np.linspace(0, 1, 120); ys = np.linspace(0, b, int(120 * b))
    Xs, Ys = np.meshgrid(xs, ys, indexing="ij")
    fig, ax = plt.subplots(figsize=(3.5, 1.8 * b + 1))
    ax.contourf(Xs, Ys, np.log10(np.clip(caja_serie(Xs, Ys, b, uniforme), 1e-8, 1)), levels=np.linspace(-8, 0, 17), cmap="viridis")
    ax.set(aspect="equal", title="log₁₀ φ"); ax.grid(False); plt.show()

interactuar(caja_alta, b=deslizador("altura b", 2.0, 0.5, 4.0, 0.5))

# %% [markdown]
# ## Experimento 2 — El centro de un cubo
#
# Un cubo de lado 1 con la tapa a $V_0$ y las demás caras a tierra. La serie doble de las notas, con $C_{nm}=16V_0/\pi^2nm$ ($n,m$ impares), da el potencial en el centro.
#
# ### Predecí
# ¿Cuánto vale? (Las notas lo deducen sin sumar ninguna serie.)

# %%
def centro_cubo(nmax):
    total = 0.0
    for n in range(1, nmax + 1, 2):
        for m in range(1, nmax + 1, 2):
            g = np.pi * np.hypot(n, m)
            total += 16 / (np.pi**2 * n * m) * np.sin(n * np.pi / 2) * np.sin(m * np.pi / 2) * np.sinh(g / 2) / np.sinh(g)
    return total

for nmax in (1, 3, 9, 41):
    print(f"términos hasta n, m = {nmax:3d}: φ(centro) = {centro_cubo(nmax):.8f}")
verificar("centro del cubo = V₀/6", centro_cubo(41), 1 / 6, tol=1e-6)

# %% [markdown]
# ### ¿Qué pasó?
# Con un solo término ($n=m=1$) ya se obtiene $0.1676$, a menos del 1 % de $1/6$: en el centro, los modos altos están exponencialmente suprimidos. Con unos pocos términos más la serie da $V_0/6$, el resultado que sale de la superposición de las seis caras.
#
# ## Experimento 3 — Rincones, aristas y un cilindro
#
# Un recinto cuadrado a potencial $V_0$ contiene un conductor a tierra que ocupa el cuadrante $x<0$, $y<0$. La región con campo tiene una **arista saliente** del conductor en el origen ($\beta=3\pi/2$) y **rincones entrantes** en las esquinas del recinto ($\beta=\pi/2$). Según las notas, cerca de un vértice $\phi-\phi_{\rm vértice}\propto\rho^{\pi/\beta}$ y $|\mathbf E|\propto\rho^{\pi/\beta-1}$.
#
# ### Predecí
# ¿Dónde es más intenso el campo: en la arista del conductor o en los rincones del recinto?

# %%
N = 241; x = np.linspace(-1, 1, N); h = x[1] - x[0]
X, Y = np.meshgrid(x, x, indexing="ij")
conductor = (X <= 1e-12) & (Y <= 1e-12)
fijo = conductor.copy(); fijo[0, :] = fijo[-1, :] = fijo[:, 0] = fijo[:, -1] = True
phi = relajar(np.where(conductor, 0.0, 1.0), fijo, h=h, omega=2 / (1 + np.sin(np.pi / N)), tol=1e-10)[0]
Ex, Ey = np.gradient(-phi, h, h)
E = np.hypot(Ex, Ey)

fig, axs = plt.subplots(1, 2, figsize=(12, 5))
ax = axs[0]
im = ax.contourf(X, Y, np.log10(np.where(conductor, np.nan, E) + 1e-6), levels=30, cmap="magma")
ax.contour(X, Y, np.where(conductor, np.nan, phi), levels=np.linspace(0.05, 0.95, 10), colors="w", linewidths=0.7)
ax.add_patch(plt.Rectangle((-1, -1), 1, 1, color="0.6"))
ax.set(aspect="equal", title="log₁₀ |E| y equipotenciales"); ax.grid(False); fig.colorbar(im, ax=ax)

i0 = N // 2; k = np.arange(1, 60); rho = np.sqrt(2) * k * h
cerca = (rho > 4 * h) & (rho < 0.1)
arista = phi[i0 + k, i0 + k]                       # sobre la bisectriz de la región con campo
rincon = 1 - phi[N - 1 - k, N - 1 - k]             # desde la esquina (1, 1) del recinto
axs[1].loglog(rho, arista, "o", ms=3, color=COLORES[1], label="arista (β = 3π/2): φ")
axs[1].loglog(rho, rincon, "s", ms=3, color=COLORES[0], label="rincón (β = π/2): V₀ − φ")
axs[1].loglog(rho, 0.9 * rho**(2 / 3), "k--", lw=1, label="ρ^{2/3}"); axs[1].loglog(rho, 2 * rho**2, "k:", lw=1, label="ρ²")
axs[1].set(xlabel="distancia ρ al vértice", title="potencial cerca de los vértices"); axs[1].legend(fontsize=9)
plt.show()

verificar("exponente cerca de la arista: π/β = 2/3", np.polyfit(np.log(rho[cerca]), np.log(arista[cerca]), 1)[0], 2 / 3, tol=2e-2)
verificar("exponente cerca del rincón: π/β = 2", np.polyfit(np.log(rho[cerca]), np.log(rincon[cerca]), 1)[0], 2.0, tol=2e-2)

# %% [markdown]
# Ahora un cilindro conductor a tierra de radio $R=0.5$ en un campo uniforme $E_0=1$. Imponemos en el borde de una caja la solución de las notas, $\phi=-E_0(\rho-R^2/\rho)\cos\varphi$, y dejamos que la relajación encuentre el resto.

# %%
N = 201; x = np.linspace(-2, 2, N); h = x[1] - x[0]
X, Y = np.meshgrid(x, x, indexing="ij"); s = np.hypot(X, Y); R = 0.5
with np.errstate(divide="ignore", invalid="ignore"):
    exacta = np.where(s > R, -(s - R**2 / s) * X / s, 0.0)
fijo = s <= R; fijo[0, :] = fijo[-1, :] = fijo[:, 0] = fijo[:, -1] = True
phi_c = relajar(np.where(fijo, exacta, 0.0), fijo, h=h, omega=2 / (1 + np.sin(np.pi / N)), tol=1e-10)[0]
fig, ax = plt.subplots(figsize=(5.5, 5))
ax.contour(X, Y, phi_c, levels=np.linspace(-2, 2, 21), colors=[COLORES[0]], linewidths=1, linestyles="solid")
ax.add_patch(plt.Circle((0, 0), R, color="0.6"))
ax.set(aspect="equal", title="cilindro en un campo uniforme: equipotenciales"); ax.grid(False); plt.show()
verificar("máx |relajación − exacta| (fuera del cilindro)", np.abs(phi_c - exacta)[s > R].max(), 0.0, tol=3e-2)
i0 = N // 2; j = np.argmax(x > R)                  # primer nodo afuera, sobre el eje x
E_sup = -(phi_c[j + 1, i0] - phi_c[j, i0]) / h
print(f"campo cerca de la superficie en φ = 0 (relajación): {E_sup:.3f}   exacto en ρ = R: 2E₀")

# %% [markdown]
# ### ¿Qué pasó?
# Cerca de la arista saliente $\phi\propto\rho^{2/3}$, así que $|\mathbf E|\propto\rho^{-1/3}$ **diverge**: es la zona brillante del mapa. En los rincones del recinto $V_0-\phi\propto\rho^2$ y $|\mathbf E|\propto\rho$ **se anula**. El cilindro en un campo uniforme concentra el campo en los puntos que miran en la dirección de $\mathbf E_0$, donde llega a $2E_0$ (la grilla escalonada lo subestima un poco).
#
# ## Experimento 4 — La función de Bessel y el cilindro con tapa
#
# Primero comprobamos la serie de $J_0$ de las notas y sus ceros. Después resolvemos un cilindro de radio $R=1$ y altura $L$, con la pared y el fondo a tierra y la tapa a $V_0$:
# $$\phi=\sum_n\frac{2V_0}{x_nJ_1(x_n)}\,J_0\!\left(\frac{x_n\rho}{R}\right)\frac{\sinh(x_nz/R)}{\sinh(x_nL/R)} .$$
#
# ### Predecí
# ¿Cómo decae el potencial a lo largo del eje, lejos de la tapa? ¿Más rápido o más lento que en la caja cuadrada de lado $2R$?

# %%
def J0_serie(x, M=40):
    m = np.arange(M)
    return np.sum((-1.0)**m[None, :] / factorial(m)[None, :]**2 * (np.atleast_1d(x)[:, None] / 2)**(2 * m[None, :]), axis=1)

xs = np.linspace(0, 20, 400)
ceros = jn_zeros(0, 6)
fig, axs = plt.subplots(1, 2, figsize=(12, 4.3))
axs[0].plot(xs, j0(xs), color=COLORES[0], label="J₀ (scipy)")
axs[0].plot(xs[::8], J0_serie(xs[::8]), "o", ms=3, color=COLORES[1], label="serie de las notas")
axs[0].plot(ceros, 0 * ceros, "kx"); axs[0].axhline(0, color="0.6", lw=0.8)
axs[0].set(xlabel="x", title="J₀ y sus ceros"); axs[0].legend()
print("ceros de J₀:", np.round(ceros, 4), "   diferencias:", np.round(np.diff(ceros), 4))
verificar("serie de J₀ en x = 7.3", J0_serie(7.3)[0], j0(7.3), tol=1e-10)
rr = np.linspace(0, 1, 20001)
verificar("ortogonalidad ∫ρ J₀(x₁ρ) J₀(x₂ρ) dρ", trapezoid(rr * j0(ceros[0] * rr) * j0(ceros[1] * rr), rr), 0.0, tol=1e-8)

# cilindro con tapa
L = 2.0; xn = jn_zeros(0, 200); An = 2 / (xn * j1(xn))
def cilindro(rho, z, L=L):
    rho, z = np.broadcast_arrays(rho, z)
    out = np.zeros(rho.shape)
    for x_, A in zip(xn, An):
        out += A * j0(x_ * rho) * np.exp(x_ * (z - L)) * (1 - np.exp(-2 * x_ * z)) / (1 - np.exp(-2 * x_ * L))
    return out

Rg, Zg = np.meshgrid(np.linspace(0, 1, 80), np.linspace(0, L, 160), indexing="ij")
ph = cilindro(Rg, Zg)
im = axs[1].contourf(np.concatenate([-Rg[::-1], Rg]), np.concatenate([Zg[::-1], Zg]), np.concatenate([ph[::-1], ph]),
                     levels=np.linspace(0, 1, 21), cmap="viridis")
axs[1].set(aspect="equal", xlabel="ρ", ylabel="z", title="cilindro con tapa a V₀ (corte)"); axs[1].grid(False)
fig.colorbar(im, ax=axs[1]); plt.show()

zz = np.linspace(1.5, 3.0, 50)                              # cilindro alto (L = 5), lejos de la tapa y del fondo
pend = np.polyfit(zz, np.log(cilindro(0 * zz, zz, L=5.0)), 1)[0]
verificar("decaimiento sobre el eje: x₁/R = 2.405", pend, ceros[0], tol=5e-3)
verificar("en la tapa, a media distancia del eje: φ = V₀", cilindro(np.array(0.5), np.array(L)), 1.0, tol=1e-2)

# %% [markdown]
# ### ¿Qué pasó?
# La serie de las notas reproduce $J_0$, sus ceros se espacian cada vez más cerca de $\pi$, y las $J_0(x_n\rho)$ son ortogonales con peso $\rho$. En el cilindro, lejos de la tapa, el potencial decae como $e^{-2.405\,(L-z)/R}$. Una caja cuadrada de lado $2R$ daría $\gamma_{11}=\pi\sqrt2/2R=2.22/R$: el cilindro, con la misma "anchura", apantalla un poco más.
#
# ## Explorá
# 1. **Guía 4, P6.** Escribí la densidad de un peine de hilos como una serie de Fourier en $x$, usá el resultado de la lámina sinusoidal de las notas para cada modo, y compará la suma con la suma directa de los potenciales logarítmicos de muchos hilos.
# 2. **Guía 4, P7.** Calculá numéricamente los coeficientes $A_m$, $B_m$ del cilindro partido con las fórmulas de las notas, sumá la serie, y compará con una relajación en un disco (los nodos del borde con $V_1$ o $V_2$ según el lado).
# 3. **Otras cuñas.** Cambiá el conductor del Experimento 3 por cuñas de $60^\circ$, $90^\circ$ y una lámina delgada ($\beta=2\pi$), y medí el exponente. ¿Coincide con $\pi/\beta$?
# 4. **Un agujero en la jaula.** En el cilindro con tapa, poné la tapa a $V_0$ solo en un disco central de radio $R/3$. ¿Cambia la longitud de penetración lejos de la tapa? ¿Por qué?
