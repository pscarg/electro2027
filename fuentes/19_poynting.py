# %% [markdown]
# # Clase 19 — Leyes de conservación: energía, momento y momento angular del campo
#
# **Objetivos**
# - Seguir el flujo de energía en un circuito coaxial: de la pila a la resistencia, por el vacío.
# - Comprobar el teorema de Poynting en una simulación: el trabajo de una fuente termina como energía del campo.
# - Calcular la fuerza entre dos hilos con el tensor de tensiones, y el momento del campo de una carga que se mueve.
# - Ver el momento angular del campo pasar a la materia, y calcular el de una carga y un monopolo.
#
# **Material relacionado:** notas de la Clase 19. Guía 7: problemas 3, 4 y 1(b).
#
# **Unidades.** Gaussianas, adimensionales, con $c=1$.

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
from scipy.integrate import quad, dblquad, cumulative_trapezoid

# %% [markdown]
# ## Experimento 1 ★ — De la pila a la resistencia, por el vacío
#
# El circuito coaxial de las notas: conductor interno de radio $a=1$ con resistencia $R$, externo perfecto de radio $b$, largo $L=10$, pila $V_0=1$ en $z=0$ y tapa conductora en $z=L$; $I=V_0/R=1$. Las notas dan los campos exactos y la función de flujo de la energía,
# $$\Psi=\frac{IV_0}{2\pi}\left[\left(1-\frac zL\right)\frac{\ln(s/a)}{\ln(b/a)}+\frac zL\right]\qquad(a<s<b),$$
# con $S_s=-\frac1s\partial_z\Psi$ y $S_z=\frac1s\partial_s\Psi$. Entre dos curvas de $\Psi$ constante viaja siempre la misma potencia. Adentro del conductor con resistencia no hay función de flujo, porque $\nabla\cdot\mathbf S=-\mathbf J\cdot\mathbf E\neq0$: ahí $\mathbf S$ es radial y la energía se va depositando (flechas).
#
# ### Predecí
# ¿Las líneas de energía van por adentro del conductor interno, por el vacío o por el conductor externo? ¿Dónde entran al conductor con resistencia?

# %%
a, L, V0, I = 1.0, 10.0, 1.0, 1.0

def Psi(s, z, b):
    """Función de flujo de la energía entre los conductores (NaN adentro del conductor interno)."""
    s = np.abs(s)
    afuera = I * V0 / (2 * np.pi) * ((1 - z / L) * np.log(np.maximum(s, a) / a) / np.log(b / a) + z / L)
    return np.where(s < a, np.nan, afuera)

def S_hueco(s, z, b):
    """(S_s, S_z) entre los conductores, con E y B de las notas (c = 1)."""
    lnba = np.log(b / a)
    Ez = V0 / L * np.log(b / s) / lnba; Es = V0 * (1 - z / L) / (s * lnba); Bphi = 2 * I / s
    return -Ez * Bphi / (4 * np.pi), Es * Bphi / (4 * np.pi)

def mostrar(b=3.0):
    z = np.linspace(0, L, 400); s = np.linspace(-b, b, 401)
    Z, Ss = np.meshgrid(z, s)
    P = np.where(np.abs(Ss) <= b, Psi(Ss, Z, b), np.nan)
    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.fill_between([0, L], -a, a, color=COLORES[3], alpha=0.25, lw=0)
    ax.contour(Z, Ss, P, levels=np.linspace(0, I * V0 / (2 * np.pi), 13)[1:-1], colors=COLORES[0], linewidths=1.2)
    zq = np.linspace(0.5, L - 0.5, 10)                                  # adentro: S radial, |S| ∝ s
    for sg in (1, -1):
        ax.quiver(zq, sg * 0.95 * a + 0 * zq, 0 * zq, -sg * 0.5 + 0 * zq, color=COLORES[1], scale=1, scale_units="xy", angles="xy", width=0.003)
    ax.plot([0, L], [b, b], "k", lw=3); ax.plot([0, L], [-b, -b], "k", lw=3)
    ax.plot([L, L], [-b, b], "k", lw=3); ax.plot([0, 0], [a, b], color=COLORES[1], lw=5); ax.plot([0, 0], [-b, -a], color=COLORES[1], lw=5)
    ax.text(-0.15, (a + b) / 2, "pila", ha="right", va="center", color=COLORES[1])
    ax.text(L / 2, 0, "conductor con resistencia", ha="center", va="center", fontsize=9)
    ax.set(xlabel="z", ylabel="s", title="líneas del flujo de energía (Ψ constante)", xlim=(-1.2, L + 0.3)); ax.grid(False)
    plt.show()

interactuar(mostrar, b=deslizador("b", 3.0, 1.5, 5.0, 0.25))

b = 3.0
costado = 2 * np.pi * a * quad(lambda z: -S_hueco(a, z, b)[0], 0, L)[0]
pila = 2 * np.pi * quad(lambda s: S_hueco(s, 0.0, b)[1] * s, a, b)[0]
verificar("potencia que entra por el costado = I²R", costado, I**2 * (V0 / I), tol=1e-10)
verificar("potencia que sale de la pila = IV₀", pila, I * V0, tol=1e-10)
h, s0, z0 = 1e-5, 1.7, 3.3
Ss_psi = -(Psi(s0, z0 + h, b) - Psi(s0, z0 - h, b)) / (2 * h) / s0
Sz_psi = (Psi(s0 + h, z0, b) - Psi(s0 - h, z0, b)) / (2 * h) / s0
verificar("S_s = −(1/s)∂Ψ/∂z", Ss_psi, S_hueco(s0, z0, b)[0], tol=1e-6)
verificar("S_z = (1/s)∂Ψ/∂s", Sz_psi, S_hueco(s0, z0, b)[1], tol=1e-6)
div = ((s0 + h) * S_hueco(s0 + h, z0, b)[0] - (s0 - h) * S_hueco(s0 - h, z0, b)[0]) / (2 * h) / s0 \
      + (S_hueco(s0, z0 + h, b)[1] - S_hueco(s0, z0 - h, b)[1]) / (2 * h)
verificar("∇·S = 0 entre los conductores", div, 0.0, tol=1e-8)

# %% [markdown]
# **Un capacitor que se carga.** Placas de radio $a=1$ separadas $h=0.2$, con corriente $I=1$: $E=4Q/a^2$. Calculamos $B_\varphi$ en el borde con Ampère–Maxwell (la derivada temporal del flujo de $\mathbf E$, numérica) y comparamos el flujo de Poynting que entra por el borde con la derivada de la energía del hueco.

# %%
hc, dt = 0.2, 1e-4
E_de = lambda t: 4 * (I * t) / a**2                     # Q = I t
flujoE = lambda t: E_de(t) * np.pi * a**2
t0 = 0.7
B_borde = (flujoE(t0 + dt) - flujoE(t0 - dt)) / (2 * dt) / (2 * np.pi * a)   # 2πa B = (1/c) dΦ_E/dt
entra = E_de(t0) * B_borde / (4 * np.pi) * 2 * np.pi * a * hc
U = lambda t: E_de(t)**2 / (8 * np.pi) * np.pi * a**2 * hc
verificar("capacitor: flujo de S por el borde = dU/dt", entra, (U(t0 + dt) - U(t0 - dt)) / (2 * dt), tol=1e-8)

# %% [markdown]
# ### ¿Qué pasó?
# Las líneas de energía salen de la pila (el anillo naranja), viajan por el vacío entre los conductores y van entrando de costado en el conductor con resistencia, una por una: entre dos líneas viaja la misma potencia, y cada tramo del conductor recibe $I^2R/L$ por unidad de largo. Ninguna línea entra al conductor externo, que no tiene resistencia. La potencia que sale de la pila y la que entra por el costado son ambas $IV_0=I^2R$. En el capacitor, lo mismo: la energía entra por el borde del hueco.
#
# ## Experimento 2 — El teorema de Poynting en una simulación
#
# La lámina de la Clase 18, con $K(t)=e^{-(t-4)^2}$, simulada con `yee1d`. Llevamos tres cuentas:
# - el trabajo del agente que mantiene la corriente, $W(t)=-\int_0^t K\,E_y(0,t')\,dt'$;
# - la energía del campo, $U(t)=\int\frac{E^2+B^2}{8\pi}dx$;
# - la energía que cruzó $x=\pm5$, integrando $S_x=\frac{c}{4\pi}E_yB_z$ en el tiempo.
#
# Las notas predicen $\frac{\pi}{c}\int K^2dt$ hacia cada lado, en total $\frac{2\pi}{c}\int K^2dt=2\pi\sqrt{\pi/2}$.
#
# ### Predecí
# ¿Cuándo empieza a cruzar energía por $x=5$? ¿Cuánto vale $U$ al final?

# %%
dx = 0.02
x = np.arange(-30, 30 + dx / 2, dx); xm = 0.5 * (x[1:] + x[:-1]); i0 = np.argmin(np.abs(x))
K = lambda t: np.exp(-(t - 4.0)**2)
def J_lamina(n):
    j = np.zeros_like(x); j[i0] = K((n + 0.5) * dx) / dx
    return j
pasos = int(round(20 / dx))
Es, Bs, ts = yee1d(0 * x, 0 * xm, pasos, S=1.0, dx=dx, J=J_lamina, cada=1)
Kn = K((np.arange(pasos) + 0.5) * dx)
W = np.concatenate([[0], np.cumsum(-Kn * 0.5 * (Es[:-1, i0] + Es[1:, i0]) * dx)])
Bt = 0.5 * (Bs[:-1] + Bs[1:])                                  # B en los tiempos de E (promedio de medio paso)
Ut = np.concatenate([[0], (np.sum(Es[1:]**2, axis=1) + np.sum(Bt**2, axis=1)) * dx / (8 * np.pi)])
i5 = np.argmin(np.abs(x - 5))
Sx5 = Es[1:, i5] * 0.5 * (Bt[:, i5] + Bt[:, i5 - 1]) / (4 * np.pi)
cruzo = np.concatenate([[0], np.cumsum(Sx5) * dx])

fig, ax = plt.subplots(figsize=(7.5, 4.2))
ax.plot(ts, W, color=COLORES[0], label="trabajo del agente W(t)")
ax.plot(ts, Ut, "--", color=COLORES[1], label="energía del campo U(t)")
ax.plot(ts, 2 * cruzo, color=COLORES[2], label="energía que cruzó x = ±5")
ax.axhline(2 * np.pi * np.sqrt(np.pi / 2), color="0.6", ls=":", label="(2π/c)∫K²dt")
ax.set(xlabel="t", ylabel="energía"); ax.legend(fontsize=9)
plt.show()

esperado = 2 * np.pi * np.sqrt(np.pi / 2)
verificar("trabajo del agente = (2π/c)∫K²dt", W[-1], esperado, tol=1e-6)
verificar("energía del campo = trabajo del agente", Ut[-1], W[-1], tol=1e-3)
verificar("energía que cruzó x = 5 = (π/c)∫K²dt", cruzo[-1], esperado / 2, tol=1e-3)

# %% [markdown]
# ### ¿Qué pasó?
# El agente trabaja contra el campo de la lámina mientras $K\neq0$; ese trabajo aparece como energía del campo, que se va en dos pulsos. Por $x=5$ empieza a pasar energía recién cuando llega el pulso, con retardo $5/c$, y en total pasa la mitad. El código solo tiene las ecuaciones de rotor: la conservación de la energía con $u=\frac{E^2+B^2}{8\pi}$ y $\mathbf S=\frac{c}{4\pi}\mathbf E\times\mathbf B$ sale sola.
#
# ## Experimento 3 — El tensor de tensiones y el momento del campo
#
# **Dos hilos.** Hilos en $x=\mp1$ (separados $d=2$) con corrientes $I_1=1$ e $I_2$. Integramos $T_{xx}=\frac{1}{8\pi}(B_x^2-B_y^2)$ sobre el plano medio: la fuerza sobre el hilo de la derecha es $F_x=-\int T_{xx}\,dy$. Las notas predicen $-\frac{2I_1I_2}{c^2d}$.
#
# ### Predecí
# Con corrientes opuestas, ¿las líneas cruzan el plano medio o corren a lo largo de él?

# %%
d = 2.0
def B_hilos(X, Y, I2):
    Bx = By = 0.0
    for x0, Ik in ((-d / 2, 1.0), (d / 2, I2)):
        r2 = (X - x0)**2 + Y**2
        Bx = Bx - 2 * Ik * Y / r2; By = By + 2 * Ik * (X - x0) / r2
    return Bx, By

def fuerza_tensor(I2):
    Txx = lambda y: (B_hilos(0.0, y, I2)[0]**2 - B_hilos(0.0, y, I2)[1]**2) / (8 * np.pi)
    return -quad(Txx, -np.inf, np.inf)[0]

def mostrar_hilos(I2=1.0):
    X, Y = np.meshgrid(np.linspace(-4, 4, 300), np.linspace(-4, 4, 300))
    Az = -2 * (np.log(np.hypot(X + d / 2, Y)) + I2 * np.log(np.hypot(X - d / 2, Y)))   # función de flujo: B = ∇ × (A_z ẑ)
    y = np.linspace(-4, 4, 400); Bx, By = B_hilos(0.0, y, I2)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
    axs[0].contour(X, Y, Az, levels=30, colors=COLORES[0], linewidths=0.8, linestyles="solid")
    axs[0].axvline(0, color=COLORES[1], ls="--"); axs[0].plot([-d / 2, d / 2], [0, 0], "ko")
    axs[0].set(aspect="equal", title="líneas de B y el plano medio"); axs[0].grid(False)
    axs[1].plot(y, (Bx**2 - By**2) / (8 * np.pi), color=COLORES[1]); axs[1].axhline(0, color="0.6", lw=0.8)
    axs[1].set(xlabel="y (sobre el plano medio)", ylabel="T_xx", title=f"F_x = −∫T_xx dy = {fuerza_tensor(I2):.4f}")
    plt.show()

interactuar(mostrar_hilos, I2=deslizador("I₂", 1.0, -1.0, 1.0, 0.25))

verificar("tensor sobre el plano: F = −2I₁I₂/c²d (paralelas)", fuerza_tensor(1.0), -2 * 1.0 / d, tol=1e-8)
verificar("tensor sobre el plano: F = +2I₁I₂/c²d (opuestas)", fuerza_tensor(-1.0), 2 * 1.0 / d, tol=1e-8)

# %% [markdown]
# **El problema de $\frac43$.** Una cáscara de radio $R=1$ y carga $q=1$ que se mueve con $\mathbf v=v\hat{\mathbf z}$: $\mathbf B=\frac{\mathbf v}{c}\times\mathbf E$ y $g_z=\frac{v}{4\pi c^2}(E^2-E_z^2)$. Integramos en $r>R$ y $\theta$, y comparamos con $U/c^2$, $U=q^2/2R$.

# %%
v = 1.0
gz = lambda th, r: v / (4 * np.pi) * (1 / r**4) * (1 - np.cos(th)**2) * 2 * np.pi * r**2 * np.sin(th)
P = dblquad(gz, 1.0, np.inf, 0, np.pi)[0]
print(f"P_campo = {P:.6f},  U v/c² = {0.5 * v:.6f},  cociente = {P / (0.5 * v):.6f}")
verificar("P_campo = (4/3) U v/c²", P, 4 / 3 * 0.5 * v, tol=1e-8)

# %% [markdown]
# ### ¿Qué pasó?
# Con corrientes paralelas las líneas cruzan el plano medio, $T_{xx}>0$, y la tensión tira de los hilos uno hacia el otro; con corrientes opuestas corren a lo largo del plano, $T_{xx}<0$, y la presión los separa. La integral da exactamente la fuerza de la Clase 12. El momento del campo de la cáscara es $\frac43$ de $U\mathbf v/c^2$: el $\frac23$ del promedio angular de $E^2-E_z^2$ por el $2$ de $\int E^2=8\pi U$.
#
# ## Experimento 4 — El momento angular del campo
#
# **El cilindro alrededor del solenoide.** Línea con $\lambda=1$ en el eje, solenoide de radio $a=1$ con flujo $\Phi(t)$, cáscara de radio $b=2$ con $-\lambda$. Apagamos el flujo suavemente, $\Phi(t)=\Phi_0\,\frac12\left(1+\cos\frac{\pi t}{T}\right)$ entre $t=0$ y $T=10$. A cada tiempo integramos el momento angular del campo, $\int_0^a s\,g_\varphi\,2\pi s\,ds$, y acumulamos el torque sobre la cáscara, $\frac{\lambda}{2\pi c}\dot\Phi$.
#
# ### Predecí
# ¿Cuánto momento angular tiene la cáscara al final? ¿Hacia qué lado gira?

# %%
lam, a_s, b_c, Phi0, T = 1.0, 1.0, 2.0, 1.0, 10.0
Phi = lambda t: Phi0 * 0.5 * (1 + np.cos(np.pi * np.clip(t, 0, T) / T))
dPhi = lambda t: np.where((t > 0) & (t < T), -Phi0 * 0.5 * np.pi / T * np.sin(np.pi * t / T), 0.0)
def L_campo(t):
    B = Phi(t) / (np.pi * a_s**2)
    g_phi = lambda s: -(2 * lam / s) * B / (4 * np.pi)          # (E × B)_φ / 4πc, con E = 2λ/s ŝ
    return quad(lambda s: s * g_phi(s) * 2 * np.pi * s, 0, a_s)[0]
tt = np.linspace(0, 12, 601)
Lc = np.array([L_campo(t) for t in tt])
Lm = cumulative_trapezoid(lam / (2 * np.pi) * dPhi(tt), tt, initial=0.0)

fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
axs[0].plot(tt, Lc, color=COLORES[0], label="L del campo")
axs[0].plot(tt, Lm, color=COLORES[1], label="L de la cáscara")
axs[0].plot(tt, Lc + Lm, "k--", lw=1, label="total")
axs[0].set(xlabel="t", ylabel="L_z por unidad de largo", title="el momento angular pasa a la materia"); axs[0].legend(fontsize=9)
th = np.linspace(0, 2 * np.pi, 12, endpoint=False); rr = np.array([0.3, 0.6, 0.9])
R_, TH = np.meshgrid(rr, th); X, Y = R_ * np.cos(TH), R_ * np.sin(TH)
axs[1].quiver(X, Y, np.sin(TH), -np.cos(TH), color=COLORES[0], scale=12)       # dirección de g: −φ̂ (su módulo va como 1/s)
axs[1].add_patch(plt.Circle((0, 0), a_s, fill=False, color="k", ls="--")); axs[1].add_patch(plt.Circle((0, 0), b_c, fill=False, color=COLORES[1], lw=2))
axs[1].set(aspect="equal", xlim=(-2.3, 2.3), ylim=(-2.3, 2.3), title="dirección de g antes de apagar (|g| ∝ 1/s)"); axs[1].grid(False)
plt.show()

verificar("L del campo inicial = −λΦ/2πc", Lc[0], -lam * Phi0 / (2 * np.pi), tol=1e-8)
verificar("L de la cáscara al final = L inicial del campo", Lm[-1], Lc[0], tol=1e-4)
verificar("total constante (máx |ΔL|)", np.max(np.abs(Lc + Lm - Lc[0])), 0.0, tol=1e-4)

# %% [markdown]
# **Una carga y un monopolo.** Carga $e$ en el origen y monopolo $g$ en $(0,0,d)$. Por simetría solo hay $L_z$; con $\mathbf E=e\mathbf r/r^3$ y $\mathbf B=g(\mathbf r-\mathbf d)/|\mathbf r-\mathbf d|^3$, $[\mathbf r\times(\mathbf E\times\mathbf B)]_z=\frac{eg\,d\,s^2}{r^3r'^3}$, y
# $$L_z=\frac{eg\,d}{2c}\int_0^\infty\!\!\int_{-\infty}^{\infty}\frac{s^3\,dz\,ds}{r^3\,r'^3},\qquad r'=|\mathbf r-\mathbf d| .$$
# Las notas dicen que da $eg/c$ para cualquier $d$.

# %%
def L_monopolo(dd):
    f = lambda z, s: s**3 / ((s * s + z * z)**1.5 * (s * s + (z - dd)**2)**1.5)
    return dd / 2 * dblquad(f, 0, np.inf, -np.inf, np.inf, epsabs=1e-11, epsrel=1e-9)[0]
for dd in (0.5, 2.0):
    verificar(f"carga y monopolo: L_z = eg/c (d = {dd:g})", L_monopolo(dd), 1.0, tol=1e-6)

# %% [markdown]
# ### ¿Qué pasó?
# Antes de apagar el solenoide nada se mueve, pero el campo tiene momento angular, con $\mathbf g$ girando alrededor del eje adentro del solenoide. Mientras el flujo baja, el campo inducido hace girar la cáscara, y el momento angular del campo pasa, sin pérdida, a la materia. El de la carga y el monopolo no depende de la distancia: con la cuantización del momento angular, es la condición de Dirac (Clase 2).
#
# ## Explorá
#
# 1. **Guía 7, P3.** Como en el Experimento 3, elegí una superficie que encierre un pedazo de la pared de un solenoide y calculá la fuerza con $T_{ij}$. ¿Por qué no hace falta integrar adentro del solenoide?
# 2. **Guía 7, P4.** Escribí, como en el Experimento 3, la integral del momento del campo $\frac{1}{4\pi c}\int\mathbf E\times\mathbf B$ para un capacitor de placas en un campo $\mathbf B$ uniforme, y compará el impulso que reciben las placas en las dos maneras de apagar el sistema.
# 3. En el Experimento 1, dale también resistencia al conductor externo (su potencial baja linealmente desde $0$ en $z=L$ hasta $-V_1$ en $z=0$). Escribí el potencial entre los conductores como combinación de $\ln(b/s)$ y $\ln(s/a)$, y dibujá las nuevas líneas de energía: ¿entra energía también al conductor externo?
# 4. En el Experimento 2, reemplazá el borde absorbente de la derecha por un espejo ($E_y=0$ en el último nodo) y calculá la fuerza por unidad de área sobre el espejo, $-T_{xx}=\frac{E^2+B^2}{8\pi}$, mientras refleja el pulso. Compará el impulso total con el doble del momento del pulso, $2U/c$: es la presión de radiación de la Clase 20.
