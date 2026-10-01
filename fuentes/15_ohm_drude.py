# %% [markdown]
# # Clase 15 — Corriente en los materiales: Ohm, Drude, Joule y la fuerza electromotriz
#
# **Objetivos**
# - Simular el **modelo de Drude**: electrones que se aceleran en un campo y chocan al azar. Ver aparecer la velocidad de deriva $q\tau E/M$ y la ley de Ohm.
# - Resolver $\nabla\cdot(\sigma\nabla\phi)=0$ en conductores no uniformes, y ver las **cargas superficiales** que guían la corriente.
# - Calcular una resistencia y compararla con la fórmula de las notas.
# - Ver cómo **desaparece la carga** del interior de un conductor: en el lugar, con el tiempo $1/4\pi\sigma$.
#
# **Material relacionado:** notas de la Clase 15. Esta clase no tiene problemas de guía asignados.
#
# **Unidades.** Gaussianas, adimensionales: $q=M=1$, tiempos en unidades de $\tau$ (Drude) o de $1/4\pi\sigma$ (relajación), longitudes en unidades del tamaño del conductor.

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
# ## Experimento 1 ★ — Una simulación de Drude
#
# $N=20000$ electrones ($q=M=1$) con velocidades térmicas al azar (dispersión $v_{th}=0.2$ en cada componente). En cada paso de tiempo $dt$, el campo $E\hat{\mathbf x}$ los acelera, y cada uno choca con probabilidad $dt/\tau$; al chocar, su velocidad se reemplaza por una térmica nueva al azar. Las notas predicen que la velocidad media sigue $\frac{q\tau E}{M}\left(1-e^{-t/\tau}\right)$.
#
# ### Predecí
# 1. ¿Cómo se ve la trayectoria de un electrón? ¿Y el promedio?
# 2. Si duplicás $E$, ¿qué pasa con la velocidad de deriva?

# %%
rng = np.random.default_rng(5)
tau, dt, vth = 1.0, 0.01, 0.2
def drude(E, N=20000, pasos=2000, seguir=5):
    """Devuelve ⟨v_x⟩(t), las posiciones x de algunos electrones y el número total de choques."""
    v = rng.normal(0, vth, (N, 3)); x = np.zeros((N, 3))
    vmedia, tray, choques = [], [], 0
    for _ in range(pasos):
        v[:, 0] += E * dt
        x += v * dt
        choca = rng.random(N) < dt / tau
        choques += choca.sum()
        v[choca] = rng.normal(0, vth, (choca.sum(), 3))
        vmedia.append(v[:, 0].mean()); tray.append(x[:seguir, 0].copy())
    return np.array(vmedia), np.array(tray), choques

E0 = 0.05
vm, tray, choques = drude(E0)
t = np.arange(1, len(vm) + 1) * dt
fig, axs = plt.subplots(1, 2, figsize=(12, 4.3))
for k in range(tray.shape[1]):
    axs[0].plot(t, tray[:, k], lw=0.8, color=COLORES[k % len(COLORES)])
axs[0].plot(t, E0 * tau * (t - tau * (1 - np.exp(-t / tau))), "k--", lw=1.5, label="posición media")
axs[0].set(xlabel="t / τ", ylabel="x", title="cinco electrones (y la deriva media)"); axs[0].legend()
axs[1].plot(t, vm, color=COLORES[0], lw=1, label="⟨v_x⟩ simulada")
axs[1].plot(t, E0 * tau * (1 - np.exp(-t / tau)), "k--", label="(qτE/M)(1 − e^{−t/τ})")
axs[1].axhline(vth, color="0.7", ls=":", label="velocidad térmica típica")
axs[1].set(xlabel="t / τ", ylabel="velocidad", title="la deriva es mucho menor que la velocidad térmica"); axs[1].legend(fontsize=9)
guardar(fig, "nb15_drude"); plt.show()

estacionario = t > 5 * tau
verificar("velocidad de deriva = qτE/M", vm[estacionario].mean(), E0 * tau, tol=3e-2)
verificar("tasa de choques por electrón = 1/τ", choques / (20000 * len(vm) * dt), 1 / tau, tol=2e-2)
Es = np.array([0.05, 0.1, 0.2, 0.4])
vds = np.array([drude(Ev, N=10000, pasos=1000)[0][500:].mean() for Ev in Es])
pend = np.polyfit(Es, vds, 1)[0]
verificar("ley de Ohm: v_d ∝ E, con pendiente qτ/M (σ = nq²τ/M)", pend, tau, tol=3e-2)

# %% [markdown]
# ### ¿Qué pasó?
# Cada electrón zigzaguea con su velocidad térmica, mucho mayor que la deriva; pero en promedio todos avanzan lentamente en la dirección de la fuerza. La velocidad media llega a $q\tau E/M$ en unos pocos $\tau$, y es proporcional a $E$: la ley de Ohm, con $\sigma=nq^2\tau/M$. La tasa de choques es $1/\tau$, como pusimos en la simulación. (Con un paso $dt$ finito la deriva sale un poco menor, en una fracción $\sim dt/\tau$: por eso usamos $dt=\tau/100$.)
#
# ## Experimento 2 — Conductores no uniformes y cargas superficiales
#
# Con la herramienta `resolver_conduccion` resolvemos $\nabla\cdot(\sigma\nabla\phi)=0$. La densidad de carga es $\rho=-\nabla^2\phi/4\pi$ (el laplaciano de siempre, sin $\sigma$).
#
# **(a) Dos materiales en serie.** Una barra con $\sigma_1=1$ en la mitad izquierda y $\sigma_2=0.25$ en la derecha, a potencial $1$ a la izquierda y $0$ a la derecha, con los costados aislantes.
#
# ### Predecí
# ¿Cómo es el potencial a lo largo de la barra? ¿Dónde hay carga, y de qué signo?

# %%
Nx, Ny, h = 101, 21, 0.01
sig = np.where(np.arange(Nx)[:, None] < Nx // 2, 1.0, 0.25) * np.ones((Nx, Ny))
fijo = np.zeros((Nx, Ny), dtype=bool); fijo[0, :] = fijo[-1, :] = True
val = np.zeros((Nx, Ny)); val[0, :] = 1.0
phi = resolver_conduccion(sig, fijo, val, h=h)
x = np.arange(Nx) * h
lap = np.zeros_like(phi); lap[1:-1, :] = (phi[2:, :] - 2 * phi[1:-1, :] + phi[:-2, :]) / h**2
rho = -lap / (4 * np.pi)
fig, axs = plt.subplots(1, 2, figsize=(12, 3.8))
axs[0].plot(x, phi[:, Ny // 2], color=COLORES[0]); axs[0].axvline(x[Nx // 2] - h / 2, color="0.6", ls=":")
axs[0].set(xlabel="x", ylabel="φ", title="potencial: dos rectas")
axs[1].plot(x, rho[:, Ny // 2] * h, color=COLORES[1]); axs[1].set(xlabel="x", ylabel="carga por unidad de área (ρh)", title="carga solo en la interfaz")
plt.tight_layout(); plt.show()

J = -1.0 * (phi[11, Ny // 2] - phi[10, Ny // 2]) / h      # densidad de corriente, leída en el tramo izquierdo
sigma_s = np.sum(rho[1:-1, Ny // 2]) * h
verificar("carga superficial en la interfaz = (J/4π)(1/σ₂ − 1/σ₁)", sigma_s, J / (4 * np.pi) * (1 / 0.25 - 1 / 1.0), tol=1e-6)
verificar("corriente igual en los dos tramos (J₁ = J₂)", -1.0 * (phi[11, 10] - phi[10, 10]) / h, -0.25 * (phi[81, 10] - phi[80, 10]) / h, tol=1e-6)

# %% [markdown]
# **(b) Un cable doblado.** Un cable en forma de L (ancho $0.2$) dentro de un aislante ($\sigma=10^{-6}$), con un extremo a potencial $1$ y el otro a $0$. Además del potencial, miramos dónde está la carga.
#
# ### Predecí
# ¿Dónde aparece carga? ¿Hay campo eléctrico afuera del cable?

# %%
N = 161; h = 2.0 / (N - 1); xs = np.arange(N) * h
X, Y = np.meshgrid(xs, xs, indexing="ij")
cable = ((X > 0.2) & (X < 1.6) & (Y > 0.2) & (Y < 0.4)) | ((X > 1.4) & (X < 1.6) & (Y > 0.2) & (Y < 1.6))
sig = np.where(cable, 1.0, 1e-6)
fijo = (cable & (X < 0.25)) | (cable & (Y > 1.55))
val = np.where(cable & (X < 0.25), 1.0, 0.0)
phi = resolver_conduccion(sig, fijo, val, h=h)
lap = np.zeros_like(phi)
lap[1:-1, 1:-1] = (phi[2:, 1:-1] + phi[:-2, 1:-1] + phi[1:-1, 2:] + phi[1:-1, :-2] - 4 * phi[1:-1, 1:-1]) / h**2
rho = -lap / (4 * np.pi)
Ex, Ey = np.gradient(-phi, h, h)

fig, axs = plt.subplots(1, 2, figsize=(12, 5.4))
im = axs[0].contourf(X, Y, phi, levels=21, cmap="viridis")
axs[0].contour(X, Y, cable.astype(float), levels=[0.5], colors="w", linewidths=1.5)
Jx = np.ma.masked_where(~cable, Ex).T; Jy = np.ma.masked_where(~cable, Ey).T
axs[0].streamplot(xs, xs, Jx, Jy, color="w", density=2.0, linewidth=0.9)
Ox = np.ma.masked_where(cable, Ex).T; Oy = np.ma.masked_where(cable, Ey).T
axs[0].streamplot(xs, xs, Ox, Oy, color="0.85", density=0.8, linewidth=0.4, arrowsize=0.6)
axs[0].set(aspect="equal", title="φ, corriente (blanco) y campo afuera (gris)"); axs[0].grid(False); fig.colorbar(im, ax=axs[0])
cerca_electrodo = np.zeros_like(fijo)
for d in range(-3, 4):
    cerca_electrodo |= np.roll(fijo, d, 0) | np.roll(fijo, d, 1)
lim = np.abs(rho[~cerca_electrodo]).max()
im = axs[1].pcolormesh(X, Y, rho, cmap="RdBu_r", vmin=-lim, vmax=lim, shading="auto")
axs[1].contour(X, Y, cable.astype(float), levels=[0.5], colors="k", linewidths=0.8)
axs[1].set(aspect="equal", title="ρ = −∇²φ/4π (escala de los costados)"); axs[1].grid(False); fig.colorbar(im, ax=axs[1])
guardar(fig, "nb15_cable"); plt.show()

# carga superficial a lo largo de los dos bordes del cable (sumando ρh en una franja de ±3 nodos a través del borde)
def borde(tramos):
    largo, carga = [], []; acum = 0.0
    for (fijo_eje, valor, desde, hasta) in tramos:
        for u in np.arange(desde, hasta, h):
            if fijo_eje == "y":
                i, j = int(round(u / h)), int(round(valor / h)); banda = rho[i, j - 3:j + 4]
            else:
                i, j = int(round(valor / h)), int(round(u / h)); banda = rho[i - 3:i + 4, j]
            largo.append(acum); carga.append(banda.sum() * h); acum += h
    return np.array(largo), np.array(carga)
l_ext, s_ext = borde([("y", 0.2, 0.3, 1.6), ("x", 1.6, 0.2, 1.5)])
l_int, s_int = borde([("y", 0.4, 0.3, 1.4), ("x", 1.4, 0.4, 1.5)])
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.plot(l_ext, s_ext, color=COLORES[0], label="borde exterior")
ax.plot(l_int, s_int, color=COLORES[1], label="borde interior")
ax.axvline(1.3, color="0.6", ls=":", lw=0.8); ax.axvline(1.1, color="0.6", ls=":", lw=0.8)
ax.axhline(0, color="0.6", lw=0.8)
ax.set(xlabel="distancia a lo largo del borde, desde el extremo a potencial 1", ylabel="carga superficial",
       title="cargas superficiales sobre el cable (líneas punteadas: esquinas)"); ax.legend()
plt.show()

adentro = cable.copy()
for d in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
    adentro &= np.roll(cable, d, axis=(0, 1))
adentro &= ~fijo & ~np.roll(fijo, 1, 0) & ~np.roll(fijo, -1, 0) & ~np.roll(fijo, 1, 1) & ~np.roll(fijo, -1, 1)
verificar("carga en el interior del cable (lejos de la superficie) / carga máxima", np.abs(rho[adentro]).max() / np.abs(rho).max(), 0.0, tol=1e-6)
j1 = np.sum((sig * Ex)[int(0.8 / h), :]) * h            # corriente por una sección del tramo horizontal
j2 = np.sum((sig * Ey)[:, int(1.0 / h)]) * h            # y por una del tramo vertical
verificar("la misma corriente en los dos tramos", j2, j1, tol=2e-2)
print(f"campo eléctrico afuera del cable, junto a la esquina exterior: |E| = {np.hypot(Ex, Ey)[int(1.65 / h), int(0.15 / h)]:.3f}")

# %% [markdown]
# ### ¿Qué pasó?
# En la barra, el potencial es una recta en cada material, con más pendiente donde la conductividad es menor; la corriente es la misma, y en la interfaz hay una carga positiva $\frac{J}{4\pi}\left(\frac1{\sigma_2}-\frac1{\sigma_1}\right)$, exactamente. En el cable doblado, la carga está **toda en la superficie** (adentro es cero, a la precisión de la máquina). A lo largo de cada borde la carga superficial cambia de signo, y los dos bordes tienen cargas distintas: esa diferencia produce adentro un campo transversal que, junto con el longitudinal, hace que la corriente siga al cable. Cerca de la esquina interior la carga es máxima, con signos opuestos a cada lado: ahí es donde la corriente tiene que doblar. El patrón exacto depende de lo que rodea al cable (dónde están los electrodos, el resto del circuito); la ley que lo fija es siempre la misma: $J_n=0$ en la superficie. Afuera hay campo eléctrico, y se ve en las líneas grises.
#
# ## Experimento 3 — La resistencia de un conductor coaxial
#
# Dos cilindros coaxiales (radios $a=0.3$ y $b=1$) con un medio de conductividad $\sigma=1$ entre ellos. Calculamos la corriente por unidad de largo y la resistencia, y comparamos con $R=\frac{\ln(b/a)}{2\pi\sigma}$ (por unidad de largo), que las notas obtienen de $RC=1/4\pi\sigma$.
#
# ### Predecí
# Si duplicás $\sigma$, ¿qué pasa con $R$? ¿Y con $C$?

# %%
N = 241; xs = np.linspace(-1.02, 1.02, N); h = xs[1] - xs[0]
X, Y = np.meshgrid(xs, xs, indexing="ij"); s = np.hypot(X, Y)
fijo = (s <= 0.3) | (s >= 1.0)
val = np.where(s <= 0.3, 1.0, 0.0)
phi = resolver_conduccion(np.ones((N, N)), fijo, val, h=h)
# corriente que sale del electrodo interior: suma de los flujos por las caras de un cuadrado que lo rodea
k = int(np.argmin(np.abs(xs + 0.6))); m = N - 1 - k        # el cuadrado va de xs[k] ≈ −0.6 a xs[m] ≈ +0.6
I = (np.sum(phi[k, k:m + 1] - phi[k - 1, k:m + 1]) + np.sum(phi[m, k:m + 1] - phi[m + 1, k:m + 1])
     + np.sum(phi[k:m + 1, k] - phi[k:m + 1, k - 1]) + np.sum(phi[k:m + 1, m] - phi[k:m + 1, m + 1]))
R_num = 1.0 / I
verificar("R por unidad de largo = ln(b/a)/2πσ", R_num, np.log(1 / 0.3) / (2 * np.pi), tol=3e-2)

# %% [markdown]
# ### ¿Qué pasó?
# La resistencia numérica coincide con la fórmula (la diferencia de un par por ciento es por los electrodos escalonados de la grilla). La resistencia es inversa a $\sigma$ y la capacidad no depende de $\sigma$: el producto $RC=1/4\pi\sigma$ es una propiedad del material, no de la geometría.
#
# ## Experimento 4 — La relajación de la carga
#
# Un disco conductor (radio $0.7$, $\sigma=1$) dentro de un aislante, con una nube de carga descentrada en su interior. En cada paso calculamos $\phi$ con $\nabla^2\phi=-4\pi\rho$ y actualizamos $\rho$ con la continuidad, $\partial_t\rho=\nabla\cdot(\sigma\nabla\phi)$. Las notas predicen que en cada punto del interior $\rho\propto e^{-4\pi\sigma t}$.
#
# ### Predecí
# ¿Cómo llega la carga a la superficie? ¿Se mueve la nube?

# %%
N = 61; xs = np.linspace(-1, 1, N); h = xs[1] - xs[0]
X, Y = np.meshgrid(xs, xs, indexing="ij")
disco = np.hypot(X, Y) < 0.7
sig = np.where(disco, 1.0, 0.0)
caja = np.zeros((N, N), dtype=bool); caja[0, :] = caja[-1, :] = caja[:, 0] = caja[:, -1] = True
rho = np.where(disco, np.exp(-((X - 0.2)**2 + (Y - 0.1)**2) / 0.02), 0.0)
Q0 = rho.sum() * h**2
tasa = 4 * np.pi * 1.0
dt = 0.01 / tasa; pasos = 300
i0, j0 = np.argmin(np.abs(xs - 0.2)), np.argmin(np.abs(xs - 0.1))
r0 = rho[i0, j0]; serie, fotos = [], {}
for n in range(pasos + 1):
    if n in (0, 30, 100, 300):
        fotos[n] = rho.copy()
    if n == pasos:
        break
    phi = resolver_conduccion(np.ones((N, N)), caja, np.zeros((N, N)), fuente=4 * np.pi * rho, h=h)
    rho = rho + dt * divergencia_flujo(sig, phi, h)
    serie.append(rho[i0, j0])
tt = np.arange(1, pasos + 1) * dt * tasa

fig, axs = plt.subplots(1, 4, figsize=(15, 3.6))
for ax, (n, foto) in zip(axs, fotos.items()):
    ax.pcolormesh(X, Y, foto, cmap="magma", shading="auto", vmin=0, vmax=max(foto.max(), 1e-12))
    ax.add_patch(plt.Circle((0, 0), 0.7, fill=False, color="w", lw=0.8))
    ax.set(aspect="equal", title=f"4πσt = {n * dt * tasa:.1f}"); ax.grid(False); ax.set_xticks([]); ax.set_yticks([])
plt.show()
fig, ax = plt.subplots(figsize=(6.4, 4))
ax.semilogy(tt, np.array(serie) / r0, color=COLORES[0], label="ρ en el centro de la nube")
ax.semilogy(tt, np.exp(-tt), "k--", label="e^{−4πσt}")
ax.set(xlabel="4πσ t", ylabel="ρ / ρ₀", title="la carga decae en el lugar"); ax.legend()
plt.show()

verificar("ρ(centro de la nube) en 4πσt = 3: e^{−3}", serie[-1] / r0, np.exp(-3), tol=3e-2)
verificar("carga total conservada", rho.sum() * h**2, Q0, tol=1e-10)
borde = disco & ~(np.roll(disco, 1, 0) & np.roll(disco, -1, 0) & np.roll(disco, 1, 1) & np.roll(disco, -1, 1))
print(f"fracción de la carga en los nodos del borde del disco al final: {rho[borde].sum() / rho.sum():.3f}")

# %% [markdown]
# ### ¿Qué pasó?
# La nube **no se desplaza**: decae en el lugar, como $e^{-4\pi\sigma t}$, y al mismo tiempo la carga aparece en el borde del disco, distribuida sobre toda la superficie (más cerca de donde estaba la nube). La carga total se conserva. (Con $\Delta t$ finito, el decaimiento discreto es $(1-4\pi\sigma\Delta t)^n$, que difiere de la exponencial en menos del 2 %.)
#
# ## Explorá
# 1. **El efecto Hall.** Agregá a la simulación de Drude un campo magnético $B\hat{\mathbf z}$ (la fuerza $\frac qc\mathbf v\times\mathbf B$) y un campo $E_y$. Buscá el $E_y$ que anula la corriente media según $y$, y compará con $E_y=JB/nqc$.
# 2. **Drude con campo alterno.** Usá $E(t)=E_0\cos\omega t$ y mirá la amplitud y la fase de $\langle v_x\rangle$ en función de $\omega\tau$. (Lo vamos a necesitar en la Clase 26.)
# 3. **Un resistor con forma rara.** Usá `resolver_conduccion` con una placa con un agujero, o con un estrangulamiento, y calculá la resistencia. ¿Dónde se disipa más potencia? (Calculá $\sigma|\nabla\phi|^2$.)
# 4. **Dos electrodos cualesquiera.** Elegí dos electrodos de formas arbitrarias y verificá numéricamente $RC=1/4\pi\sigma$: $C$ es la carga de un electrodo (de $\oint\mathbf E\cdot d\mathbf a/4\pi$) sobre $V$, con el mismo potencial.
