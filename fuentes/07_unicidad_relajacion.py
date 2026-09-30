# %% [markdown]
# # Clase 7 — Unicidad, principio variacional y funciones armónicas
#
# **Objetivos**
# - Ver que el **método de relajación** (cada punto pasa a ser el promedio de sus vecinos) baja la energía en cada paso y llega **a la misma solución** desde cualquier punto de partida: unicidad y principio de Dirichlet en acción.
# - Ver la **ensilladura de Earnshaw**: una carga entre cuatro cargas fijas parece atrapada en el plano, pero escapa por arriba.
# - Comprobar el **teorema del valor medio**, con cargas afuera y adentro de la esfera.
# - Medir cuántos pasos necesita la relajación, y por qué crece como $N^2$.
#
# **Material relacionado:** notas de la Clase 7. Guía 4: problema 1.
#
# **Unidades.** Gaussianas, adimensionales: cargas en unidades de $q$, longitudes en unidades del radio exterior o del lado del cuadrado. En dos dimensiones, las energías son por unidad de longitud en $z$.

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
from scipy.integrate import solve_ivp

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
# ## Experimento 1 ★ — Relajar desde tres puntos de partida
#
# Un capacitor cilíndrico: el cilindro interior (radio $a=0.3$) está a potencial $V_0=1$ y el exterior (radio $b=1$) a tierra. Visto en un corte, es un anillo. Fijamos los nodos de la grilla que caen dentro del cilindro interior ($\phi=1$) y fuera del exterior ($\phi=0$), y relajamos el resto partiendo de tres potenciales iniciales muy distintos:
# - **cero** en todos los nodos libres;
# - **ruido**: números al azar entre $-5$ y $5$;
# - **ondas**: $3\sin(10x)\cos(7y)+2$.
#
# Usamos Gauss–Seidel rojo-negro con sobrerrelajación, $\omega=2/(1+\sin(\pi/N))$, que según las notas también baja la energía en cada paso (porque $0<\omega<2$).
#
# ### Predecí
# 1. ¿Cómo evoluciona la energía $U_h$ en cada corrida? ¿Puede subir en algún paso?
# 2. ¿Llegan las tres corridas al mismo potencial? ¿A la misma energía? ¿Cuál es esa energía según las notas?

# %%
def anillo(N, a=0.3, b=1.0, V0=1.0):
    """Grilla N×N que cubre el capacitor cilíndrico; devuelve coordenadas, paso, radio, nodos fijos y valores de borde."""
    x = np.linspace(-1.02 * b, 1.02 * b, N); h = x[1] - x[0]
    X, Y = np.meshgrid(x, x, indexing="ij"); s = np.hypot(X, Y)
    fijo = (s <= a) | (s >= b)
    fijo[0, :] = fijo[-1, :] = fijo[:, 0] = fijo[:, -1] = True
    borde = np.where(s <= a, V0, 0.0)
    return x, h, X, Y, s, fijo, borde

a, b, V0 = 0.3, 1.0, 1.0
N = 121
x, h, X, Y, s, fijo, borde = anillo(N, a, b, V0)
omega = 2 / (1 + np.sin(np.pi / N))
rng = np.random.default_rng(7)
iniciales = {
    "cero":  borde.copy(),
    "ruido": np.where(fijo, borde, rng.uniform(-5, 5, borde.shape)),
    "ondas": np.where(fijo, borde, 3 * np.sin(10 * X) * np.cos(7 * Y) + 2),
}
resultados = {}
for nombre, phi0 in iniciales.items():
    resultados[nombre] = relajar(phi0, fijo, h=h, omega=omega, tol=1e-9, cada=1)

U_exacta = V0**2 / (4 * np.log(b / a))
fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
for (nombre, (phi, n, hist)), c in zip(resultados.items(), COLORES):
    pasos, U = np.array(hist).T
    axs[0].semilogy(pasos, U, color=c, label=f"{nombre} ({n} pasos)")
axs[0].axhline(U_exacta, color="k", ls="--", lw=1, label=r"$V_0^2/4\ln(b/a)$")
axs[0].set(xlabel="paso", ylabel="energía $U_h$", title="la energía baja en cada paso"); axs[0].legend()
phi = resultados["cero"][0]
im = axs[1].contourf(X, Y, phi, levels=20, cmap="viridis")
axs[1].contour(X, Y, phi, levels=10, colors="w", linewidths=0.6)
axs[1].set(aspect="equal", title="potencial final", xlabel="x", ylabel="y"); axs[1].grid(False)
fig.colorbar(im, ax=axs[1]); guardar(fig, "nb07_relajacion"); plt.show()

for nombre, (phi, n, hist) in resultados.items():
    U = np.array(hist)[:, 1]
    verificar(f"mayor aumento de U_h entre pasos ({nombre})", max(np.diff(U).max(), 0.0), 0.0, tol=1e-12)
dif = max(np.abs(resultados["cero"][0] - resultados[k][0]).max() for k in ("ruido", "ondas"))
verificar("máxima diferencia entre las tres soluciones finales", dif, 0.0, tol=1e-6)

# %% [markdown]
# ### ¿Qué pasó?
# La energía baja en **cada** paso en las tres corridas, aunque la del ruido empieza unas $10^5$ veces más arriba. Las tres llegan al mismo potencial (a menos del criterio de corte): el punto de partida solo cambia cuántos pasos hacen falta. Es la versión discreta de dos teoremas de la clase: la unicidad (hay un solo potencial con esos valores de borde) y el principio de Dirichlet (ese potencial es el de mínima energía).
#
# Mové el deslizador para ver cómo se va borrando el ruido. Lo primero que desaparece son los detalles finos; lo último, las variaciones a la escala del anillo.

# %%
def instantanea(pasos=5):
    phi = relajar(iniciales["ruido"], fijo, h=h, omega=1.0, max_pasos=int(pasos), tol=0)[0] if pasos > 0 else iniciales["ruido"]
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    im = ax.contourf(X, Y, phi, levels=np.linspace(-1, 2, 31), cmap="viridis", extend="both")
    ax.set(aspect="equal", title=f"ruido después de {int(pasos)} pasos de Gauss–Seidel"); ax.grid(False)
    fig.colorbar(im, ax=ax); plt.show()

interactuar(instantanea, pasos=IntSlider(value=5, min=0, max=400, step=5, description="pasos", continuous_update=False))

# %% [markdown]
# **Comparación con la solución exacta.** Las notas deducen $\phi(s)=V_0\ln(b/s)/\ln(b/a)$ y $U=V_0^2/4\ln(b/a)$. La grilla aproxima los círculos con escalones de tamaño $h$, así que esperamos un error proporcional a $h$. Lo medimos con tres resoluciones y extrapolamos a $h\to0$ suponiendo $U_h=U+c\,h$.

# %%
Ns = [61, 121, 241]
Uh, hs = [], []
for n_ in Ns:
    x_, h_, X_, Y_, s_, fijo_, borde_ = anillo(n_, a, b, V0)
    phi_, _, _ = relajar(borde_, fijo_, h=h_, omega=2 / (1 + np.sin(np.pi / n_)), tol=1e-9)
    Uh.append(energia_discreta(phi_, h_)); hs.append(h_)
Uh, hs = np.array(Uh), np.array(hs)

fig, axs = plt.subplots(1, 2, figsize=(12, 4.4))
m = (s > a) & (s < b)
axs[0].plot(s[m], phi[m], ".", ms=1.5, color=COLORES[0], label="relajación (N = 121)")
ss = np.linspace(a, b, 200)
axs[0].plot(ss, V0 * np.log(b / ss) / np.log(b / a), "k", lw=1.5, label=r"$V_0\ln(b/s)/\ln(b/a)$")
axs[0].set(xlabel="s", ylabel="φ", title="potencial en función del radio"); axs[0].legend()
axs[1].loglog(hs, np.abs(Uh - U_exacta) / U_exacta, "o-", color=COLORES[1])
axs[1].set(xlabel="paso de la grilla h", ylabel="error relativo de $U_h$", title="error de discretización ∝ h")
axs[1].set_xticks(hs, labels=[f"{v:.3f}" for v in hs]); axs[1].set_xticks([], minor=True)
axs[1].set_yticks(np.abs(Uh - U_exacta) / U_exacta, labels=[f"{v:.1%}" for v in np.abs(Uh - U_exacta) / U_exacta]); axs[1].set_yticks([], minor=True)
fig.tight_layout(); plt.show()

pendiente = np.polyfit(np.log(hs), np.log(np.abs(Uh - U_exacta)), 1)[0]
verificar("pendiente del error de la energía (esperada 1)", pendiente, 1.0, tol=0.1)
U_extrap = (hs[1] * Uh[2] - hs[2] * Uh[1]) / (hs[1] - hs[2])
verificar("energía extrapolada a h → 0 = V0²/4 ln(b/a)", U_extrap, U_exacta, tol=5e-3)

# %% [markdown]
# ## Experimento 2 — La ensilladura de Earnshaw
#
# Cuatro cargas $Q=1$ fijas en $(\pm1,0,0)$ y $(0,\pm1,0)$. En el origen el campo es cero. Ponemos ahí una carga de prueba positiva.
#
# ### Predecí
# ¿El equilibrio es estable? ¿En qué direcciones? Mirá primero el potencial en el plano del cuadrado, y después en un corte vertical.

# %%
cargas = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]], dtype=float)

def potencial(p):
    """φ de las cuatro cargas en puntos p (..., 3)."""
    return sum(1 / np.linalg.norm(p - c, axis=-1) for c in cargas)

t = np.linspace(-0.6, 0.6, 121)
A, B = np.meshgrid(t, t, indexing="ij")
cero = np.zeros_like(A)
fig, axs = plt.subplots(1, 2, figsize=(12, 4.8))
for ax, P, titulo, etiqueta in [(axs[0], np.stack([A, B, cero], -1), "plano del cuadrado (z = 0)", "y"),
                                (axs[1], np.stack([A, cero, B], -1), "corte vertical (y = 0)", "z")]:
    im = ax.contourf(A, B, potencial(P), levels=30, cmap="viridis")
    ax.plot(0, 0, "w+", ms=12, mew=2)
    ax.set(aspect="equal", xlabel="x", ylabel=etiqueta, title=titulo); ax.grid(False)
    fig.colorbar(im, ax=ax, label="φ")
fig.tight_layout(); plt.show()

# matriz de derivadas segundas en el origen, por diferencias finitas
d = 1e-4
H = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        ei, ej = np.eye(3)[i] * d, np.eye(3)[j] * d
        H[i, j] = (potencial(ei + ej) - potencial(ei - ej) - potencial(-ei + ej) + potencial(-ei - ej)) / (4 * d**2)
autovalores = np.sort(np.linalg.eigvalsh(H))
print("derivadas segundas de φ en el origen:\n", np.round(H, 5))
verificar("traza = ∇²φ", np.trace(H), 0.0, tol=1e-5)
verificar("∂²φ/∂z² = −4Q/d³", autovalores[0], -4.0, tol=1e-5)
verificar("∂²φ/∂x² = ∂²φ/∂y² = 2Q/d³", autovalores[2], 2.0, tol=1e-5)

# %% [markdown]
# Ahora la soltamos (masa 1, carga 1) cerca del centro: una vez apartada solo en $x$, y otra vez con además un desplazamiento chiquito en $z$.

# %%
def campo(p):
    return sum((p - c) / np.linalg.norm(p - c)**3 for c in cargas)

def movimiento(t, u):
    return np.concatenate([u[3:], campo(u[:3])])

fig, ax = plt.subplots(figsize=(7, 4.2))
for dz, c in [(0.0, COLORES[0]), (1e-3, COLORES[1])]:
    sol = solve_ivp(movimiento, [0, 12], [0.05, 0, dz, 0, 0, 0], rtol=1e-10, atol=1e-12, max_step=0.02)
    ax.plot(sol.t, sol.y[0], color=c, label=f"x(t), z₀ = {dz:g}")
    ax.plot(sol.t, sol.y[2], "--", color=c, label=f"z(t), z₀ = {dz:g}")
ax.set(xlabel="t", ylabel="posición", ylim=(-0.3, 0.3), title="¿atrapada?"); ax.legend(ncol=2, fontsize=9)
plt.show()

# %% [markdown]
# ### ¿Qué pasó?
# En el plano del cuadrado el origen es un **mínimo** de $\phi$: apartada en $x$, la carga oscila. En el corte vertical es un **máximo**: el menor desplazamiento en $z$ crece exponencialmente y la carga escapa. Las derivadas segundas son $2$, $2$ y $-4$ (en unidades de $Q/d^3$), como en las notas, y suman cero porque $\nabla^2\phi=0$. Es una ensilladura, y el teorema de Earnshaw dice que siempre lo es.

# %% [markdown]
# ## Experimento 3 — El teorema del valor medio
#
# Promediamos el potencial de una carga sobre una esfera de radio $R=1$ centrada en el origen, usando muchas direcciones casi uniformes (la espiral de Fibonacci de la Clase 6). La carga está a distancia $d$ del centro, afuera ($d>R$) o adentro ($d<R$).
#
# ### Predecí
# Dibujá el promedio en función de $d$, de $0$ a $3$. ¿Qué pasa cuando la carga está adentro?

# %%
def direcciones(M=400):
    """M direcciones casi uniformes (espiral de Fibonacci)."""
    i = np.arange(M) + 0.5
    th = np.arccos(1 - 2 * i / M); ph = np.pi * (1 + 5**0.5) * i
    return np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], axis=1)

esfera = direcciones(40000)
eje = np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8])   # una dirección cualquiera

def promedio(d):
    return np.mean(1 / np.linalg.norm(esfera - d * eje, axis=1))

ds = np.concatenate([np.linspace(0, 0.9, 10), np.linspace(1.1, 3, 12)])
fig, ax = plt.subplots(figsize=(6.8, 4.4))
ax.plot(ds, [promedio(d) for d in ds], "o", color=COLORES[0], label="promedio numérico")
dd = np.linspace(0, 3, 300)
ax.plot(dd, 1 / np.maximum(dd, 1), "k", lw=1.5, label="1/máx(d, R)")
ax.axvline(1, color="gray", lw=0.8, ls=":")
ax.set(xlabel="distancia d de la carga al centro", ylabel="promedio de φ sobre la esfera"); ax.legend()
plt.show()

verificar("carga afuera (d = 2): promedio = φ(centro) = q/d", promedio(2.0), 0.5, tol=1e-4)
verificar("carga adentro (d = 0.5): promedio = q/R", promedio(0.5), 1.0, tol=1e-4)

# varias cargas afuera, al azar: el promedio es el potencial en el centro
pos = rng.normal(size=(8, 3)); pos *= (rng.uniform(1.5, 4, 8) / np.linalg.norm(pos, axis=1))[:, None]
qs = rng.uniform(-1, 1, 8)
prom = np.mean(sum(q / np.linalg.norm(esfera - p, axis=1) for q, p in zip(qs, pos)))
verificar("8 cargas afuera: promedio sobre la esfera = φ(centro)", prom, np.sum(qs / np.linalg.norm(pos, axis=1)), tol=1e-4)

# %% [markdown]
# ### ¿Qué pasó?
# Con la carga afuera, el promedio es $q/d$: el potencial en el centro, como dice el teorema del valor medio. Con la carga adentro, el promedio es $q/R$ **sin importar dónde esté**: la parte cercana de la esfera aporta más, pero ocupa menos ángulo sólido visto desde la carga. Es el potencial de una cáscara uniforme evaluado en la carga (Clase 3).

# %% [markdown]
# ## Experimento 4 — ¿Cuántos pasos hacen falta?
#
# Una caja cuadrada de $N\times N$ celdas con borde a tierra: la solución exacta es $\phi=0$, así que $\phi$ **es** el error. Partimos de un error uniforme, $\phi=1$ en el interior, y comparamos Jacobi, Gauss–Seidel y sobrerrelajación.
#
# ### Predecí
# Las notas predicen que en Jacobi el error se multiplica, a la larga, por $\cos(\pi/N)$ en cada paso. ¿Cómo crece el número de pasos con $N$ en cada método?

# %%
def paso_jacobi(phi, fijo=None):
    """Un paso de Jacobi: todos los nodos interiores pasan a ser el promedio de sus vecinos del paso anterior."""
    nuevo = phi.copy()
    nuevo[1:-1, 1:-1] = 0.25 * (phi[:-2, 1:-1] + phi[2:, 1:-1] + phi[1:-1, :-2] + phi[1:-1, 2:])
    return nuevo

def paso_gs(omega):
    """Un paso de Gauss–Seidel rojo-negro (sobrerrelajación si omega > 1)."""
    return lambda phi, fijo: relajar(phi, fijo, omega=omega, max_pasos=1, tol=0)[0]

def caja(N):
    """Caja de N×N celdas con borde a tierra; error inicial uniforme e = 1 en el interior."""
    phi = np.zeros((N + 1, N + 1)); phi[1:-1, 1:-1] = 1.0
    fijo = np.zeros_like(phi, dtype=bool); fijo[0, :] = fijo[-1, :] = fijo[:, 0] = fijo[:, -1] = True
    return phi, fijo

def pasos_hasta(paso, N, eps):
    """Pasos necesarios para que el error máximo baje en un factor eps (la solución exacta es φ = 0)."""
    phi, fijo = caja(N); n = 0
    while np.abs(phi).max() > eps:
        phi = paso(phi, fijo); n += 1
    return n

# el factor de Jacobi por paso, después de que se apagan los modos rápidos
for N in (20, 40):
    phi, fijo = caja(N)
    for _ in range(3000):
        anterior, phi = phi, paso_jacobi(phi)
    verificar(f"factor de Jacobi por paso, N = {N}: cos(π/N)",
              np.linalg.norm(phi) / np.linalg.norm(anterior), np.cos(np.pi / N), tol=1e-5)

# pasos necesarios en función de N
Ns4 = np.array([10, 20, 40, 80])
eps = 1e-6
pasos = {"Jacobi": [pasos_hasta(paso_jacobi, N, eps) for N in Ns4],
         "Gauss–Seidel": [pasos_hasta(paso_gs(1.0), N, eps) for N in Ns4],
         "sobrerrelajación": [pasos_hasta(paso_gs(2 / (1 + np.sin(np.pi / N))), N, eps) for N in Ns4]}

fig, ax = plt.subplots(figsize=(6.8, 4.6))
for (nombre, p), c in zip(pasos.items(), COLORES):
    ax.loglog(Ns4, p, "o-", color=c, label=nombre)
ax.loglog(Ns4, 2 * Ns4**2 / np.pi**2 * np.log(1 / eps), "k--", lw=1, label=r"$(2N^2/\pi^2)\ln(1/\varepsilon)$")
ax.set(xlabel="N", ylabel="pasos para reducir el error en $10^{-6}$", title="costo de la relajación"); ax.legend()
ax.set_xticks(Ns4, labels=[str(n) for n in Ns4]); ax.set_xticks([], minor=True)
plt.show()
for nombre, p in pasos.items():
    print(f"{nombre:18s}", p)

pend = {k: np.polyfit(np.log(Ns4[1:]), np.log(np.array(v)[1:]), 1)[0] for k, v in pasos.items()}
verificar("Jacobi con N = 80: (2N²/π²) ln(1/ε)", pasos["Jacobi"][-1], 2 * 80**2 / np.pi**2 * np.log(1 / eps), tol=0.05)
verificar("Jacobi: pasos ∝ N² (pendiente 2)", pend["Jacobi"], 2.0, tol=0.02)
verificar("Jacobi / Gauss–Seidel = 2 (N = 80)", pasos["Jacobi"][-1] / pasos["Gauss–Seidel"][-1], 2.0, tol=0.01)
print(f"pendiente de la sobrerrelajación: {pend['sobrerrelajación']:.2f}")

# %% [markdown]
# ### ¿Qué pasó?
# El factor medido coincide con $\cos(\pi/N)$: el modo más lento es el más suave, $\sin(\pi x/L)\sin(\pi y/L)$. Por eso Jacobi necesita del orden de $N^2$ pasos: duplicar la resolución cuadruplica el trabajo, como el tiempo de difusión $L^2/D$. Gauss–Seidel, que usa los valores nuevos enseguida, tarda la mitad, y sigue siendo $\propto N^2$. La sobrerrelajación con el $\omega$ óptimo baja el costo a $\propto N$: con $N=80$ necesita unas 40 veces menos pasos que Gauss–Seidel.
#
# ## Explorá
# 1. **Guía 4, P1.** Agregá a `energia_discreta` el término del funcional que encontraste en el problema, y verificá que `relajar` con una densidad `rho` (por ejemplo, una carga concentrada en un nodo dentro de una caja a tierra) lo hace bajar en cada paso.
# 2. **Neumann en todo el borde.** Cambiá la relajación para que en los bordes de la caja valga $\partial\phi/\partial n=0$ (en cada paso, copiá en cada nodo del borde el valor de su vecino interior). Poné una carga $+1$ y una $-1$ adentro. ¿Llegan dos puntos de partida distintos a la misma solución, o difieren en una constante? ¿Qué pasa si las dos cargas son $+1$? (Pista: la condición de compatibilidad de las notas.)
# 3. **Plano de simetría.** Dos cargas iguales en una caja a tierra, simétricas respecto de la recta $y=0$. Resolvé la caja completa, y después solo la mitad $y>0$ con $\partial\phi/\partial n=0$ sobre $y=0$. Compará.
# 4. **Trampa de Paul.** Cerca del centro del Experimento 2, $\phi\simeq\phi_0+k\,(x^2+y^2-2z^2)$. Si el voltaje oscila, $k(t)=k_0\cos\Omega t$, integrá el movimiento con `solve_ivp` y buscá para qué $\Omega$ la carga queda atrapada en las tres direcciones. ¿Contradice esto el teorema de Earnshaw?
