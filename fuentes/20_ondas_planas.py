# %% [markdown]
# # Clase 20 — Ondas planas: polarización, energía, momento y presión de radiación
#
# **Objetivos**
# - Dibujar la elipse de polarización a partir del vector complejo $\mathbf E_0$, con sus semiejes, su sentido de giro y su descomposición en ondas circulares.
# - Medir la presión de radiación sobre un espejo en una simulación de las ecuaciones de Maxwell.
# - Superponer ondas planas y separar la velocidad de fase de la velocidad de la energía.
#
# **Material relacionado:** notas de la Clase 20. Guía 8: problemas 1 a 5.
#
# **Unidades.** Gaussianas, adimensionales, con $c=1$ y $k=\omega=1$ salvo que se diga otra cosa.

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
# herramienta: fdtd1d v1 (NB18)
def yee1d(E, B, pasos, S=1.0, dx=1.0, J=None, cada=1, absorbente=True):
    """Ecuaciones de Maxwell en 1D con el esquema de Yee, en unidades con c = 1.

    E: E_y en los nodos x_i = i dx, en t = 0 (N valores).
    B: B_z en los puntos medios x_{i+1/2}, en t = −dt/2 (N − 1 valores).
    dt = S dx, con S ≤ 1 (número de Courant). J(n): J_y en los nodos en t = (n + 1/2) dt (opcional).
    Actualiza ∂B_z/∂t = −∂E_y/∂x y ∂E_y/∂t = −∂B_z/∂x − 4π J_y, con bordes absorbentes de Mur (exactos si S = 1).
    Devuelve las historias (Es, Bs, ts), guardadas cada `cada` pasos (t es el tiempo de E; B va medio paso atrás).
    """
    E = np.array(E, float); B = np.array(B, float); dt = S * dx
    k = (S - 1) / (S + 1)
    Es, Bs, ts = [E.copy()], [B.copy()], [0.0]
    for n in range(pasos):
        B -= S * (E[1:] - E[:-1])
        E0, E1, Ef, Ef1 = E[0], E[1], E[-1], E[-2]
        E[1:-1] -= S * (B[1:] - B[:-1])
        if J is not None:
            E -= 4 * np.pi * dt * J(n)
        if absorbente:
            E[0] = E1 + k * (E[1] - E0)
            E[-1] = Ef1 + k * (E[-2] - Ef)
        if (n + 1) % cada == 0:
            Es.append(E.copy()); Bs.append(B.copy()); ts.append((n + 1) * dt)
    return np.array(Es), np.array(Bs), np.array(ts)
# fin herramienta

# %%
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (proyección 3D)
import sympy as sp

# %% [markdown]
# ## Experimento 1 ★ — La elipse de polarización
#
# Una onda según $\hat{\mathbf z}$ con $\mathbf E_0=A\,\hat{\mathbf x}+B\,e^{i\delta}\,\hat{\mathbf y}$ ($A$, $B$ reales). Las notas dan los semiejes,
# $$a^2,\,b^2=\tfrac12\left(|\mathbf E_0|^2\pm|\mathbf E_0\cdot\mathbf E_0|\right),$$
# el sentido de giro, con el signo de $i(\mathbf E_0\times\mathbf E_0^*)\cdot\hat{\mathbf z}$, y la descomposición $\mathbf E_0=E_+\hat{\mathbf e}_++E_-\hat{\mathbf e}_-$ con $\hat{\mathbf e}_\pm=(\hat{\mathbf x}\pm i\hat{\mathbf y})/\sqrt2$.
#
# ### Predecí
# Con $A=1$, $B=0.5$ y $\delta=90^\circ$, ¿qué curva describe $\mathbf E$ en un punto fijo, y hacia dónde gira? ¿Y con $\delta=45^\circ$?

# %%
e_mas = np.array([1, 1j, 0]) / np.sqrt(2); e_menos = np.array([1, -1j, 0]) / np.sqrt(2)

def elipse(E0):
    """Semiejes (a, b), ejes reales (E_r, E_i) con E_r ⟂ E_i, y helicidad i(E0 × E0*)·ẑ."""
    E0 = np.asarray(E0, complex)
    EE = E0 @ E0; norma2 = np.real(np.vdot(E0, E0))
    a = np.sqrt(0.5 * (norma2 + abs(EE))); b = np.sqrt(max(0.5 * (norma2 - abs(EE)), 0.0))
    rot = np.exp(-0.5j * np.angle(EE)) * E0                   # fase que hace E_r ⟂ E_i
    hel = np.real(1j * np.cross(E0, E0.conj())[2])
    return a, b, rot.real, rot.imag, hel

def mostrar(A=1.0, B=0.5, delta=90.0):
    E0 = np.array([A, B * np.exp(1j * np.radians(delta)), 0])
    a, b, Er, Ei, hel = elipse(E0)
    wt = np.linspace(0, 2 * np.pi, 400)
    Et = np.real(np.outer(np.exp(-1j * wt), E0))
    fig = plt.figure(figsize=(12, 4.8))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(Et[:, 0], Et[:, 1], color=COLORES[0])
    ax.annotate("", xy=Et[25, :2], xytext=Et[0, :2], arrowprops=dict(arrowstyle="->", color=COLORES[1], lw=2))
    for v, col in ((Er, COLORES[2]), (Ei, COLORES[3])):
        ax.plot([-v[0], v[0]], [-v[1], v[1]], "--", color=col, lw=1)
    giro = "positiva (antihoraria)" if hel > 1e-12 else ("negativa (horaria)" if hel < -1e-12 else "lineal")
    ax.set(aspect="equal", xlim=(-1.6, 1.6), ylim=(-1.6, 1.6), xlabel="E_x", ylabel="E_y",
           title=f"a = {a:.3f}, b = {b:.3f}, helicidad {giro}")
    Ep, Em = np.vdot(e_mas, E0), np.vdot(e_menos, E0)
    ax.text(-1.5, -1.5, f"|E₊| = {abs(Ep):.3f},  |E₋| = {abs(Em):.3f}", fontsize=9)
    ax3 = fig.add_subplot(1, 2, 2, projection="3d")
    z = np.linspace(0, 4 * np.pi, 300)
    Ez = np.real(np.outer(np.exp(1j * z), E0))                 # t = 0: E(z) = Re[E0 e^{ikz}]
    ax3.plot(z, Ez[:, 0], Ez[:, 1], color=COLORES[0])
    for i in range(0, 300, 12):
        ax3.plot([z[i], z[i]], [0, Ez[i, 0]], [0, Ez[i, 1]], color="0.6", lw=0.8)
    ax3.plot(z, 0 * z, 0 * z, "k", lw=0.8)
    ax3.set(xlabel="z", ylabel="E_x", zlabel="E_y", title="E a lo largo de z en t = 0")
    plt.show()

interactuar(mostrar, A=deslizador("A", 1.0, 0.0, 1.5, 0.05), B=deslizador("B", 0.5, 0.0, 1.5, 0.05),
            delta=deslizador("δ (grados)", 90.0, -180.0, 180.0, 5.0))

E0 = np.array([1.0, 0.7 * np.exp(1j * np.radians(60)), 0])
a, b, Er, Ei, hel = elipse(E0)
wt = np.linspace(0, 2 * np.pi, 200001)
Et = np.real(np.outer(np.exp(-1j * wt), E0)); modulo = np.linalg.norm(Et, axis=1)
verificar("semieje mayor = √[(|E₀|² + |E₀·E₀|)/2]", modulo.max(), a, tol=1e-8)
verificar("semieje menor = √[(|E₀|² − |E₀·E₀|)/2]", modulo.min(), b, tol=1e-8)
verificar("E_r ⟂ E_i con la fase elegida", Er @ Ei, 0.0, tol=1e-12)
dEt = np.gradient(Et, wt, axis=0)
verificar("(E × dE/dωt)_z = ½ i(E₀ × E₀*)_z", np.mean(np.cross(Et, dEt)[:, 2]), 0.5 * hel, tol=1e-6)
Ep, Em = np.vdot(e_mas, E0), np.vdot(e_menos, E0)
verificar("E₊ ê₊ + E₋ ê₋ = E₀", np.max(np.abs(Ep * e_mas + Em * e_menos - E0)), 0.0, tol=1e-14)
verificar("a = (|E₊| + |E₋|)/√2", (abs(Ep) + abs(Em)) / np.sqrt(2), a, tol=1e-12)
verificar("helicidad = |E₊|² − |E₋|²", abs(Ep)**2 - abs(Em)**2, hel, tol=1e-12)

# %% [markdown]
# ### ¿Qué pasó?
# Con $\delta=90^\circ$ la elipse tiene los ejes según $x$ e $y$, y gira con helicidad positiva; con otro $\delta$ los ejes se inclinan, pero los semiejes siguen dados por $|\mathbf E_0|^2$ y $|\mathbf E_0\cdot\mathbf E_0|$. El sentido de giro es el signo de $|E_+|^2-|E_-|^2$: gana la componente circular más grande. A la derecha, el campo a lo largo de $z$ en un instante: una hélice (aplastada) que avanza con la onda.
#
# ## Experimento 2 — La presión de radiación sobre un espejo
#
# Con el esquema de Yee de la Clase 18 (`yee1d` con `absorbente=False`, que deja $E_y=0$ en los extremos: dos espejos perfectos), mandamos un pulso $E_y=B_z=e^{-(x+8)^2}$ hacia el espejo de la derecha. La fuerza por unidad de área sobre el espejo es $-T_{xx}=\frac{E^2+B^2}{8\pi}$ en su superficie, y el impulso es su integral en el tiempo. Las notas predicen el doble del momento del pulso, $2U/c$.
#
# ### Predecí
# ¿Qué signo tienen $E_y$ y $B_z$ en el pulso reflejado?

# %%
dx = 0.02
x = np.arange(-20, 20 + dx / 2, dx); xm = 0.5 * (x[1:] + x[:-1])
pulso = lambda u: np.exp(-(u + 8.0)**2)
E_ini = pulso(x); E_ini[[0, -1]] = 0.0
B_ini = pulso(xm + dx / 2)                                 # B en t = −dt/2: pulso que va hacia +x
pasos = int(round(40 / dx))
Es, Bs, ts = yee1d(E_ini, B_ini, pasos, S=1.0, dx=dx, cada=1, absorbente=False)

U = (np.sum(E_ini**2) + np.sum(B_ini**2)) * dx / (8 * np.pi)
F = (Bs[:, -1]**2 + (0.5 * Es[:, -2])**2) / (8 * np.pi)       # en x_{N−1/2}: B, y E promedio de los dos nodos (E_N = 0)
impulso = np.sum(F[1:]) * dx

fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
axs[0].imshow(Es[::10], extent=(x[0], x[-1], ts[-1], 0), aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
axs[0].set(xlabel="x", ylabel="t", title="E_y: el pulso se refleja y cambia de signo"); axs[0].grid(False)
axs[1].plot(ts, F, color=COLORES[1]); axs[1].set(xlabel="t", ylabel="fuerza por unidad de área", title=f"impulso = {impulso:.5f},  2U/c = {2 * U:.5f}")
plt.show()

k = np.argmin(np.abs(ts - 36.0))
verificar("impulso sobre el espejo = 2U/c", impulso, 2 * U, tol=1e-3)
iref = np.argmin(np.abs(x - (2 * 20 - (-8 + ts[k]))))      # reflejado en el espejo x = 20: está en 2·20 − (−8 + t)
verificar("pulso reflejado: E_y = −(pulso incidente)", Es[k][iref], -1.0, tol=1e-3)
verificar("energía conservada después de reflejarse", (np.sum(Es[k]**2) + np.sum(Bs[k]**2)) * dx / (8 * np.pi), U, tol=1e-3)

# %% [markdown]
# ### ¿Qué pasó?
# Mientras el pulso está sobre el espejo, la fuerza es positiva: empuja. El impulso total es el doble del momento que traía el pulso, porque el pulso se va con el momento opuesto. El reflejado tiene $E_y$ invertido ($E_t=0$ en el conductor) y $B_z$ del mismo signo: con $\hat{\mathbf k}$ invertido, sigue cumpliendo $\mathbf B=\hat{\mathbf k}\times\mathbf E$. En la superficie del espejo $E=0$ y $B$ se duplica, y la fuerza es $B^2/8\pi$.
#
# ## Experimento 3 — Superposición de ondas planas
#
# Las funciones `campos` y `promedios` suman ondas planas de la misma frecuencia, dadas por $\mathbf k$ y $\mathbf E_0$, y calculan $\langle u\rangle=\frac{|\mathbf E|^2+|\mathbf B|^2}{16\pi}$ y $\langle\mathbf S\rangle=\frac{c}{8\pi}\operatorname{Re}(\mathbf E\times\mathbf B^*)$ con las amplitudes complejas. Primero, dos ondas polarizadas según $\hat{\mathbf z}$ con $\mathbf k_\pm=k(\cos\alpha,\pm\sin\alpha,0)$.
#
# ### Predecí
# ¿Con qué velocidad avanzan las crestas según $x$? ¿Y la energía?

# %%
def campos(ondas, X, Y, Z=0.0):
    """Amplitudes complejas (sin e^{−iωt}) de E y B para una suma de ondas planas [(k, E0), ...] de igual |k|."""
    E = np.zeros(np.shape(X) + (3,), complex); B = np.zeros_like(E)
    for kv, E0 in ondas:
        kv = np.asarray(kv, float); E0 = np.asarray(E0, complex)
        assert abs(kv @ E0) < 1e-12, "la onda tiene que ser transversal"
        fase = np.exp(1j * (kv[0] * X + kv[1] * Y + kv[2] * Z))[..., None]
        E = E + E0 * fase; B = B + np.cross(kv / np.linalg.norm(kv), E0) * fase
    return E, B

def promedios(E, B):
    """⟨u⟩ y ⟨S⟩ (c = 1) a partir de las amplitudes complejas."""
    u = (np.sum(np.abs(E)**2, axis=-1) + np.sum(np.abs(B)**2, axis=-1)) / (16 * np.pi)
    S = np.real(np.cross(E, B.conj())) / (8 * np.pi)
    return u, S

def cruzadas(alfa):
    return [((np.cos(alfa), np.sin(alfa), 0), (0, 0, 1)), ((np.cos(alfa), -np.sin(alfa), 0), (0, 0, 1))]

def mostrar_cruce(alfa_grados=30.0, t=0.0):
    al = np.radians(alfa_grados)
    X, Y = np.meshgrid(np.linspace(0, 30, 300), np.linspace(-15, 15, 300))
    E, B = campos(cruzadas(al), X, Y)
    u, S = promedios(E, B)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.4, 1]})
    axs[0].imshow(np.real(E[..., 2] * np.exp(-1j * t)), extent=(0, 30, -15, 15), origin="lower", cmap="RdBu_r", vmin=-2, vmax=2)
    axs[0].set(xlabel="x", ylabel="y", title=f"E_z en t = {t:g}"); axs[0].grid(False)
    y = Y[:, 0]
    axs[1].plot(u[:, 0], y, color=COLORES[0], label="⟨u⟩")
    axs[1].plot(S[:, 0, 0], y, color=COLORES[1], label="⟨S_x⟩")
    axs[1].set(xlabel="valor", ylabel="y", title="promedios temporales"); axs[1].legend(fontsize=9)
    plt.show()

interactuar(mostrar_cruce, alfa_grados=deslizador("α (grados)", 30.0, 0.0, 80.0, 5.0), t=deslizador("t", 0.0, 0.0, 6.28, 0.25))

al = np.radians(30)
ky = np.sin(al)
y = np.linspace(0, 2 * np.pi / ky, 4000, endpoint=False)     # un período del patrón en y (cos² tiene período π/κ)
E, B = campos(cruzadas(al), 0 * y, y)
u, S = promedios(E, B)
verificar("velocidad de la energía = c cos α", np.mean(S[:, 0]) / np.mean(u), np.cos(al), tol=1e-10)
verificar("⟨S_y⟩ = 0", np.max(np.abs(S[:, 1])), 0.0, tol=1e-14)
C2 = np.cos(ky * y)**2
verificar("⟨u⟩ de las notas", np.max(np.abs(u - ((1 + np.cos(al)**2) * C2 + np.sin(al)**2 * (1 - C2)) / (4 * np.pi))), 0.0, tol=1e-12)
xs = np.linspace(0, 10, 1001)
Ex, _ = campos(cruzadas(al), xs, 0 * xs)
beta = np.polyfit(xs, np.unwrap(np.angle(Ex[:, 2])), 1)[0]
verificar("velocidad de fase = c / cos α", 1.0 / beta, 1 / np.cos(al), tol=1e-10)

# %% [markdown]
# **La presión oblicua, con el tensor.** Un espejo en $x>0$; la onda incidente $\mathbf k=(\cos\theta,\sin\theta,0)$, $\mathbf E_0=\hat{\mathbf z}$, y la reflejada $\mathbf k'=(-\cos\theta,\sin\theta,0)$, $\mathbf E_0'=-\hat{\mathbf z}$ (para que $E_z=0$ en $x=0$). En la superficie, $P=-\langle T_{xx}\rangle$ con $\langle E_iE_j\rangle=\frac12\operatorname{Re}(E_iE_j^*)$. Las notas dan $P=\frac{2I}{c}\cos^2\theta$, con $I=\frac{c}{8\pi}$.

# %%
th = np.radians(60)
ondas = [((np.cos(th), np.sin(th), 0), (0, 0, 1)), ((-np.cos(th), np.sin(th), 0), (0, 0, -1))]
ys = np.linspace(0, 10, 7)
E, B = campos(ondas, 0 * ys, ys)
Txx = (0.5 * np.real(E[:, 0] * E[:, 0].conj() + B[:, 0] * B[:, 0].conj())
       - 0.25 * (np.sum(np.abs(E)**2, axis=1) + np.sum(np.abs(B)**2, axis=1))) / (4 * np.pi)
print("E_z en la superficie:", np.max(np.abs(E[:, 2])))
verificar("presión oblicua: −⟨T_xx⟩ = (2I/c) cos²θ", np.mean(-Txx), 2 / (8 * np.pi) * np.cos(th)**2, tol=1e-12)

# %% [markdown]
# ### ¿Qué pasó?
# El patrón de las dos ondas avanza según $x$ con crestas que van a $c/\cos\alpha$, más rápido que la luz, mientras que la energía, promediada en $y$, avanza a $c\cos\alpha$. Hay franjas: $\langle u\rangle$ y $\langle S_x\rangle$ oscilan en $y$, pero $\langle S_y\rangle=0$. Donde $\cos(\kappa y)=0$ se podría poner un conductor sin cambiar nada: es una guía de ondas. Con el espejo inclinado $60^\circ$, el tensor da $\frac{2I}{c}\cos^2\theta$, un cuarto de la presión normal.
#
# ## Explorá
#
# 1. **Guía 8, P1.** La celda de abajo define `maxwell_vacio`, que con `sympy` devuelve lo que sobra en cada una de las cuatro ecuaciones de Maxwell sin fuentes, comprobada con una onda monocromática. Usala con funciones arbitrarias $f_1(x-ct)$, $f_2(x-ct)$ (`sp.Function`) para el problema 1.
# 2. **Guía 8, P2 y P5.** Con `campos` y `promedios`, armá los campos de los problemas 2 y 5 como superposición de ondas planas, y mirá $\langle u\rangle$ y $\langle\mathbf S\rangle$ en función de la posición (y, con un factor $e^{-i\omega t}$, los campos en distintos instantes).
# 3. **Guía 8, P3.** En el Experimento 2 el espejo devuelve el pulso. ¿Qué impulso esperás si la superficie absorbiera todo? Pensalo con el flujo de momento $T_{xx}=-u$ de las notas.
# 4. **Guía 8, P4.** En el Experimento 1, buscá los valores de $A$, $B$ y $\delta$ que dan polarización lineal y circular, y explicalos con $|\mathbf E_0\cdot\mathbf E_0|$.

# %%
xs_, ys_, zs_, ts_ = sp.symbols("x y z t", real=True)
c_ = sp.Symbol("c", positive=True)

def maxwell_vacio(E, B):
    """Restos de las ecuaciones de Maxwell sin fuentes: (∇·E, ∇·B, ∇×E + ∂ₜB/c, ∇×B − ∂ₜE/c)."""
    X = (xs_, ys_, zs_)
    div = lambda F: sp.simplify(sum(sp.diff(F[i], X[i]) for i in range(3)))
    rot = lambda F: [sp.diff(F[2], ys_) - sp.diff(F[1], zs_), sp.diff(F[0], zs_) - sp.diff(F[2], xs_), sp.diff(F[1], xs_) - sp.diff(F[0], ys_)]
    rE, rB = rot(E), rot(B)
    return (div(E), div(B),
            [sp.simplify(rE[i] + sp.diff(B[i], ts_) / c_) for i in range(3)],
            [sp.simplify(rB[i] - sp.diff(E[i], ts_) / c_) for i in range(3)])

k_ = sp.Symbol("k", positive=True)
onda = sp.cos(k_ * (xs_ - c_ * ts_))
print(maxwell_vacio([0, onda, 0], [0, 0, onda]))
verificar("maxwell_vacio: una onda monocromática no deja restos", float(sum(abs(sp.N(r)) for grupo in maxwell_vacio([0, onda, 0], [0, 0, onda])[2:] for r in grupo)), 0.0, tol=1e-12)
