# %% [markdown]
# # Clase 16 — La ley de Faraday
#
# **Objetivos**
# - Simular un **imán que cae a través de una espira** y de un tubo: el frenado de Lenz, la fem en el tiempo, el balance de energía y la velocidad terminal.
# - Calcular la fem del **disco de Faraday** por caminos distintos dentro del disco.
# - Comprobar que la fem de una espira que se mueve frente a un imán es la misma calculada de tres maneras: fuerza de Lorentz, regla del flujo, y campo eléctrico inducido en el sistema de la espira.
# - Acelerar un electrón en un **betatrón** sin que cambie de órbita.
#
# **Material relacionado:** notas de la Clase 16. Guía 6: problemas 1 a 4.
#
# **Unidades.** Gaussianas, adimensionales: $c=1$, $M=g=1$, longitudes en unidades del radio de la espira.

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
from scipy.integrate import solve_ivp, quad

# %% [markdown]
# ## Experimento 1 ★ — Un imán que cae a través de una espira
#
# Un dipolo $m\hat{\mathbf z}$ de masa $M$ cae desde $z_0=5$ a lo largo del eje de una espira de radio $a=1$ y resistencia $R$, en $z=0$. Las notas dan el flujo $\Phi(z)=\frac{2\pi ma^2}{(a^2+z^2)^{3/2}}$ y la ecuación de movimiento
# $$M\ddot z=-Mg-k(z)\dot z,\qquad k(z)=\frac{\Phi'(z)^2}{c^2R}.$$
#
# ### Predecí
# 1. ¿En qué momento frena más el imán: al acercarse, al pasar por el centro de la espira, o al alejarse?
# 2. ¿Cómo es la fem en función del tiempo?

# %%
a, M, g, m = 1.0, 1.0, 1.0, 1.0
Phi = lambda z: 2 * np.pi * m * a**2 / (a**2 + z**2)**1.5
dPhi = lambda z: -6 * np.pi * m * a**2 * z / (a**2 + z**2)**2.5

def caer(R, z0=5.0, t_fin=8.0, anillos=None):
    """Integra la caída. anillos = posiciones de las espiras (por defecto, una en z = 0). Devuelve la solución con
    (z, v, energía disipada)."""
    zs = np.array([0.0]) if anillos is None else np.asarray(anillos)
    def k(z):
        return np.sum(dPhi(z - zs)**2) / R
    def f(t, y):
        z, v, Q = y
        return [v, -g - k(z) * v / M, k(z) * v**2]
    return solve_ivp(f, [0, t_fin], [z0, 0.0, 0.0], rtol=1e-10, atol=1e-12, max_step=0.01, dense_output=True)

def mostrar(R=10.0):
    sol = caer(R); t = np.linspace(0, 8, 800); z, v, Q = sol.sol(t)
    emf = -dPhi(z) * v
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    axs[0].plot(t, v, color=COLORES[0], label=f"con la espira (R = {R:g})")
    axs[0].plot(t, -g * t, "--", color="0.6", label="caída libre")
    axs[0].axvline(t[np.argmin(np.abs(z))], color="0.7", ls=":", lw=0.8)
    axs[0].set(xlabel="t", ylabel="velocidad", title="el imán frena dos veces", ylim=(-8, 0.5)); axs[0].legend()
    axs[1].plot(t, emf, color=COLORES[1]); axs[1].axhline(0, color="0.6", lw=0.8)
    axs[1].set(xlabel="t", ylabel="fem", title="la fem cambia de signo al pasar")
    plt.show()

interactuar(mostrar, R=deslizador("R", 10.0, 1.0, 100.0, 1.0))

# comprobaciones
sol = caer(10.0, t_fin=6.0); z, v, Q = sol.y[:, -1]
verificar("balance: Mg(z₀ − z) − ½Mv² = energía disipada", M * g * (5.0 - z) - 0.5 * M * v**2, Q, tol=1e-6)
zz = 1.7
Bz_eje = lambda s: m * (3 * zz**2 / (s**2 + zz**2) - 1) / (s**2 + zz**2)**1.5    # B_z del dipolo en el plano de la espira
verificar("flujo del dipolo por la espira = 2πma²/(a²+z²)^{3/2}", quad(lambda s: Bz_eje(s) * 2 * np.pi * s, 0, a)[0], Phi(zz), tol=1e-8)

# %% [markdown]
# **Un tubo.** Ahora el imán cae por dentro de una pila de espiras (una cada $0.1$, de $z=-60$ a $z=10$), cada una con resistencia $R=200$. Las notas predicen una velocidad terminal $v_t=Mg/K$ con $K=\frac{45\pi^3nm^2}{32c^2Ra^3}$.

# %%
anillos = np.arange(-60, 10.001, 0.1)
R_tubo = 200.0
sol = caer(R_tubo, z0=5.0, t_fin=60.0, anillos=anillos)
K = 45 * np.pi**3 * 10 * m**2 / (32 * R_tubo * a**3)
t = np.linspace(0, 60, 1200); z, v, Q = sol.sol(t)
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(t, v, color=COLORES[0], label="dentro del tubo")
ax.axhline(-M * g / K, color="k", ls="--", lw=1, label="−Mg/K")
ax.set(xlabel="t", ylabel="velocidad", title="velocidad terminal en un tubo"); ax.legend()
plt.show()
lejos = (z < -10) & (z > -50)
verificar("velocidad terminal = Mg/K", -v[lejos].mean(), M * g / K, tol=1e-2)

# %% [markdown]
# ### ¿Qué pasó?
# El imán frena **antes y después** de pasar por la espira, y no en el centro: ahí el flujo es máximo y no cambia ($\Phi'=0$). La fem tiene dos lóbulos de signo opuesto: al acercarse, la corriente inducida repele al imán; al alejarse, circula al revés y lo atrae. En los dos casos la fuerza se opone a la velocidad (Lenz), y la energía mecánica que falta es exactamente la disipada en la espira. En un tubo, la fuerza es proporcional a la velocidad y el imán cae a velocidad constante.
#
# ## Experimento 2 — El disco de Faraday
#
# Un disco de radio $a=1$ gira con $\omega=1$ en un campo $B=1$. Calculamos $\mathcal E=\int\frac{\mathbf v\times\mathbf B}{c}\cdot d\mathbf l$ desde el eje hasta el borde por un radio y por una espiral.
#
# ### Predecí
# ¿Dan lo mismo los dos caminos? ¿Cuánto vale la fem?

# %%
w, B = 1.0, 1.0
def fem_camino(s, phi):
    x, y = s * np.cos(phi), s * np.sin(phi)
    v = np.stack([-w * y, w * x, 0 * x], axis=1)
    f = np.cross(v, np.array([0, 0, B]))
    dl = np.diff(np.stack([x, y, 0 * x], axis=1), axis=0)
    fm = 0.5 * (f[1:] + f[:-1])
    return np.sum(fm * dl)
u = np.linspace(0, 1, 20001)
radio = fem_camino(u, 0 * u)
espiral = fem_camino(u, 3 * np.pi * u)
fig, ax = plt.subplots(figsize=(4.4, 4.4))
ax.add_patch(plt.Circle((0, 0), 1, color="0.85"))
ax.plot(u, 0 * u, color=COLORES[0], label=f"radio: {radio:.6f}")
ax.plot(u * np.cos(3 * np.pi * u), u * np.sin(3 * np.pi * u), color=COLORES[1], label=f"espiral: {espiral:.6f}")
ax.set(aspect="equal", title="caminos dentro del disco"); ax.legend(fontsize=8, loc="lower left"); ax.grid(False)
plt.show()
verificar("fem por un radio = ωBa²/2c", radio, w * B / 2, tol=1e-8)
verificar("fem por una espiral = la misma", espiral, radio, tol=1e-8)

# %% [markdown]
# ### ¿Qué pasó?
# Los dos caminos dan $\omega Ba^2/2c$: la fuerza por unidad de carga, $\frac{\omega sB}{c}\hat{\mathbf s}$, es radial y solo depende de $s$, así que su integral no depende del camino. El flujo a través de "el circuito" no está bien definido (el disco siempre ocupa la misma región), pero la fuerza de Lorentz da la fem sin ambigüedad.
#
# ## Experimento 3 — Tres cálculos de la misma fem
#
# Un dipolo $m\hat{\mathbf z}$ en el origen y una espira cuadrada de lado 1, horizontal, a altura $0.6$, que se mueve con velocidad $v\hat{\mathbf x}$. Calculamos la fem de tres maneras:
# 1. **Laboratorio, fuerza de Lorentz:** $\oint(\mathbf v\times\mathbf B)\cdot d\mathbf l/c$.
# 2. **Regla del flujo:** $-\frac1c\,d\Phi/dt$, con $\Phi=\oint\mathbf A\cdot d\mathbf l$ y una derivada numérica.
# 3. **Sistema de la espira:** ahí el imán se mueve con $-\mathbf v$, $\mathbf A'(\mathbf r',t)=\mathbf A(\mathbf r'+\mathbf vt)$, y el campo inducido es $\mathbf E'=-\frac1c\partial_t\mathbf A'=-\frac1c(\mathbf v\cdot\nabla)\mathbf A$ (más un gradiente, que no circula).
#
# ### Predecí
# ¿Coinciden los tres? ¿Qué signo tiene la fem cuando la espira se aleja del imán?

# %%
mv = np.array([0, 0, 1.0]); vel = np.array([0.3, 0, 0])
def A_dip(P):
    r = np.linalg.norm(P, axis=1)[:, None]
    return np.cross(mv, P) / r**3
def B_dip(P):
    r = np.linalg.norm(P, axis=1)[:, None]; n = P / r
    return (3 * (n @ mv)[:, None] * n - mv) / r**3
def cuadrado(xc, n=4000):
    """Puntos y elementos dl del cuadrado de lado 1 centrado en (xc, 0, 0.6), recorrido antihorario visto desde +z."""
    esq = np.array([[0.5, -0.5], [0.5, 0.5], [-0.5, 0.5], [-0.5, -0.5], [0.5, -0.5]])
    P, dl = [], []
    for p, q in zip(esq[:-1], esq[1:]):
        u = (np.arange(n) + 0.5) / n
        pts = p + np.outer(u, q - p)
        P.append(np.column_stack([pts[:, 0] + xc, pts[:, 1], np.full(n, 0.6)])); dl.append(np.tile(np.append((q - p) / n, 0), (n, 1)))
    return np.vstack(P), np.vstack(dl)

xc = 0.8
P, dl = cuadrado(xc)
fem_lorentz = np.sum(np.cross(vel, B_dip(P)) * dl)
flujo = lambda x: np.sum(A_dip(cuadrado(x)[0]) * cuadrado(x)[1])
hh = 1e-4
fem_flujo = -(flujo(xc + hh) - flujo(xc - hh)) / (2 * hh) * vel[0]
dA = sum(vel[k] * (A_dip(P + hh * e) - A_dip(P - hh * e)) / (2 * hh) for k, e in enumerate(np.eye(3)))
fem_inducido = np.sum(-dA * dl)
print(f"Lorentz: {fem_lorentz:.8f}   flujo: {fem_flujo:.8f}   E inducido: {fem_inducido:.8f}")
verificar("regla del flujo = fuerza de Lorentz", fem_flujo, fem_lorentz, tol=1e-6)
verificar("campo inducido en el sistema de la espira = fuerza de Lorentz", fem_inducido, fem_lorentz, tol=1e-6)

# %% [markdown]
# ### ¿Qué pasó?
# Los tres cálculos coinciden. En el laboratorio, las cargas de la espira se mueven y la fuerza es magnética; en el sistema de la espira, están quietas y la fuerza es la de un campo eléctrico inducido por el imán que se mueve. La regla del flujo da el mismo número sin preguntar cuál de las dos cosas está pasando. (La identidad detrás: $\mathbf v\times(\nabla\times\mathbf A)=\nabla(\mathbf v\cdot\mathbf A)-(\mathbf v\cdot\nabla)\mathbf A$, y el gradiente no circula.)
#
# ## Experimento 4 — El betatrón
#
# Un campo axial $B_z=B_0(t)R_0/s$, con $B_0(t)=1+0.05t$. El flujo dentro de un radio $s$ es $\Phi=2\pi B_0R_0s$, así que por Faraday $E_\varphi=-\frac{1}{2\pi sc}\frac{d\Phi}{dt}=-R_0\dot B_0/c$. En $s=R_0$ el campo es la mitad del campo medio encerrado: cumple la condición del betatrón. Integramos una carga con $q/M=1$ con el método de Boris (Clase 14), empezando en la órbita $R_0=1$ con $p=qB_0R_0/c$.
#
# ### Predecí
# ¿Qué le pasa al radio de la órbita mientras la carga se acelera? ¿Y si el campo fuera uniforme?

# %%
# herramienta: boris v1 (NB14)
def boris(x0, v0, campos, q_M=1.0, c=1.0, dt=1e-2, pasos=1000):
    """Integra M dv/dt = q (E + v×B/c) con el método de Boris (no relativista).

    campos(x, t) devuelve (E, B) en la posición x (arreglos de 3 componentes).
    Las posiciones quedan en los tiempos n·dt y las velocidades en (n + ½)·dt.
    Con E = 0, el paso de Boris es una rotación exacta de v: conserva |v| a precisión de máquina.
    """
    x = np.array(x0, float); v = np.array(v0, float)
    X = np.empty((pasos + 1, 3)); V = np.empty((pasos, 3)); X[0] = x
    for n in range(pasos):
        E, B = campos(x, n * dt)
        v_menos = v + 0.5 * dt * q_M * np.asarray(E)
        t = 0.5 * dt * q_M * np.asarray(B) / c
        s = 2 * t / (1 + t @ t)
        v_prima = v_menos + np.cross(v_menos, t)
        v = v_menos + np.cross(v_prima, s) + 0.5 * dt * q_M * np.asarray(E)
        x = x + dt * v
        X[n + 1] = x; V[n] = v
    return X, V
# fin herramienta

# %%
R0, beta = 1.0, 0.05
def campos_betatron(x, t, uniforme=False):
    s = np.hypot(x[0], x[1]); B0 = 1 + beta * t
    phi_hat = np.array([-x[1] / s, x[0] / s, 0])
    if uniforme:                       # B uniforme: Φ = πs²B₀, E_φ = −(s/2) dB₀/dt
        return -(s / 2) * beta * phi_hat, np.array([0, 0, B0])
    return -R0 * beta * phi_hat, np.array([0, 0, B0 * R0 / s])

dt = 0.002; pasos = 10000
X, V = boris([R0, 0, 0], [0, -1.0, 0], campos_betatron, dt=dt, pasos=pasos)
Xu, Vu = boris([R0, 0, 0], [0, -1.0, 0], lambda x, t: campos_betatron(x, t, uniforme=True), dt=dt, pasos=pasos)
t = np.arange(pasos) * dt
fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
axs[0].plot(t, np.hypot(X[1:, 0], X[1:, 1]), color=COLORES[0], label="B ∝ 1/s (condición del betatrón)")
axs[0].plot(t, np.hypot(Xu[1:, 0], Xu[1:, 1]), color=COLORES[1], label="B uniforme")
axs[0].set(xlabel="t", ylabel="radio de la órbita", title="radio"); axs[0].legend()
axs[1].plot(t, np.linalg.norm(V, axis=1), color=COLORES[0], label="rapidez")
axs[1].plot(t, (1 + beta * (t + dt / 2)) * R0, "k--", lw=1, label="qB₀(t)R₀/M")
axs[1].set(xlabel="t", ylabel="rapidez", title="la carga se acelera"); axs[1].legend()
plt.show()
r = np.hypot(X[:, 0], X[:, 1])
verificar("radio de la órbita constante (máx |r − R₀|)", np.abs(r - R0).max(), 0.0, tol=1e-2)
verificar("rapidez final = qB₀R₀/M", np.linalg.norm(V[-1]), (1 + beta * (t[-1] + dt / 2)) * R0, tol=1e-3)

# %% [markdown]
# ### ¿Qué pasó?
# Con el campo en la órbita igual a la mitad del campo medio encerrado, la carga se acelera (su rapidez sigue a $qB_0(t)R_0/Mc$) **sin cambiar de radio**: el campo eléctrico inducido la empuja a lo largo de la órbita, y el campo magnético crece justo lo necesario para mantenerla en el mismo círculo. Con un campo uniforme la condición no se cumple y la órbita se achica mientras el campo crece.
#
# ## Explorá
# 1. **Guía 6, P1.** Calculá numéricamente el flujo a través de una espira cuadrada que gira en un campo uniforme, derivalo y compará la fem con tu resultado.
# 2. **Guía 6, P2.** Calculá la fem de una espira que se aleja de un hilo largo con $v=v_0$ y con $v=v_0\sin\omega t$. Para el segundo caso, mirá el espectro de la fem con `np.fft.rfft`: ¿qué frecuencias aparecen, y por qué?
# 3. **Guía 6, P4.** Resolvé con las leyes de Kirchhoff el circuito de la barra con dos resistencias, y simulá la barra con masa empujada por una fuerza constante: ¿llega a una velocidad terminal?
# 4. **El tubo de cobre.** Agregá a cada espira del tubo su autoinductancia ($\mathcal E=IR+\frac{L}{c^2}\dot I$) y mirá cuándo importa. Con la fórmula de las notas, estimá cuánto tarda un imán de neodimio en atravesar un tubo de aluminio en lugar de cobre.
