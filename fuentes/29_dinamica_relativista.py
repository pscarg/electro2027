# %% [markdown]
# # Clase 29 — Dinámica relativista: cuadrimomento, fuerza y principio de acción
#
# **Objetivos**
# - Integrar la ecuación de movimiento relativista de una carga con un integrador de Boris relativista, y compararla con la de Newton.
# - Ver cómo se mueve una carga en un campo eléctrico uniforme (movimiento hiperbólico) y en campos cruzados con $E<B$ y con $E>B$.
# - Comprobar numéricamente que la trayectoria real hace estacionaria la acción relativista, y que la de Newton no.
# - Usar el cuadrimomento: la masa invariante de los productos de un decaimiento y sus energías en el laboratorio.
#
# **Material relacionado:** notas de la Clase 29. Guía 12: problemas 3 y 4.
#
# **Unidades.** $c=1$; en los experimentos 1 y 2, $q/m=1$ (las velocidades se miden en unidades de $c$ y $u=\gamma v$ también); en el 3, energías en MeV.

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
from scipy.integrate import trapezoid, cumulative_trapezoid

# %% [markdown]
# ## Experimento 1 ★ — Un integrador relativista
#
# La ecuación de movimiento de una carga es $\frac{d}{dt}(\gamma m\mathbf v)=q\left(\mathbf E+\frac{\mathbf v}{c}\times\mathbf B\right)$ (secciones 2 y 4 de las notas). Con $\mathbf u=\gamma\mathbf v$, es $\frac{d\mathbf u}{dt}=\frac qm\left(\mathbf E+\frac{\mathbf u\times\mathbf B}{\gamma c}\right)$, con $\gamma=\sqrt{1+u^2/c^2}$. El método de Boris de la Clase 14 se generaliza así: medio impulso eléctrico, una rotación con el $\gamma$ de ese momento, y otro medio impulso. Como la rotación conserva $|\mathbf u|$, con $\mathbf E=0$ la energía se conserva exactamente. Comparamos con el Boris de Newton de la Clase 14.
#
# Primero, un campo eléctrico uniforme $E_0\hat{\mathbf x}$, desde el reposo: las notas dan $x(t)=\frac{mc^2}{qE_0}\left(\sqrt{1+(qE_0t/mc)^2}-1\right)$. Después, campos cruzados $\mathbf E=E\hat{\mathbf y}$ y $\mathbf B=B\hat{\mathbf z}$: las notas muestran que $p_x-\frac{qB}{c}y$ y $\mathcal E-qEy$ se conservan, y que el movimiento es acotado si $E<B$.
#
# ### Predecí
# 1. Con $E_0$ uniforme, ¿qué da Newton para la velocidad después de un tiempo $5mc/qE_0$?
# 2. Con campos cruzados y $E>B$, la deriva de la Clase 14 daría $v_d=cE/B>c$. ¿Qué hace la carga?

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

# herramienta: boris_rel v1 (NB29)
def boris_rel(x0, u0, campos, q_M=1.0, c=1.0, dt=1e-2, pasos=1000):
    """Integra d(γv)/dt = (q/m)(E + v×B/c) con el método de Boris relativista, en términos de u = γv.

    campos(x, t) devuelve (E, B) en la posición x (arreglos de 3 componentes).
    Las posiciones quedan en los tiempos n·dt y los u en (n + ½)·dt; γ = √(1 + u²/c²).
    Con E = 0, el paso es una rotación exacta de u: conserva |u| (y la energía) a precisión de máquina.
    """
    x = np.array(x0, float); u = np.array(u0, float)
    X = np.empty((pasos + 1, 3)); U = np.empty((pasos, 3)); X[0] = x
    for n in range(pasos):
        E, B = campos(x, n * dt)
        u_menos = u + 0.5 * dt * q_M * np.asarray(E)
        g = np.sqrt(1 + u_menos @ u_menos / c**2)
        t = 0.5 * dt * q_M * np.asarray(B) / (g * c)
        s = 2 * t / (1 + t @ t)
        u_prima = u_menos + np.cross(u_menos, t)
        u = u_menos + np.cross(u_prima, s) + 0.5 * dt * q_M * np.asarray(E)
        x = x + dt * u / np.sqrt(1 + u @ u / c**2)
        X[n + 1] = x; U[n] = u
    return X, U
# fin herramienta

def campos_uniformes(E, B):
    E = np.array(E, float); B = np.array(B, float)
    return lambda x, t: (E, B)

def mostrar_campo_uniforme(exportar=False):
    dt, T = 1e-3, 5.0; n = int(T / dt)
    f = campos_uniformes([1.0, 0, 0], [0, 0, 0])
    Xr, Ur = boris_rel([0, 0, 0], [0, 0, 0], f, dt=dt, pasos=n)
    Xn, Vn = boris([0, 0, 0], [0, 0, 0], f, dt=dt, pasos=n)
    t = dt * np.arange(n + 1); th = dt * (np.arange(n) + 0.5)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    a1.plot(Xr[:, 0], t, color=COLORES[0], label="relativista")
    a1.plot(Xn[:, 0], t, "--", color=COLORES[1], label="Newton")
    a1.plot(t, t, ":", color="k", lw=1, label="luz")
    a1.set(xlabel="x (unidades de mc²/qE₀)", ylabel="ct", xlim=(0, 6), aspect="equal", title="línea de mundo desde el reposo")
    a1.legend(fontsize=9)
    vr = Ur[:, 0] / np.sqrt(1 + Ur[:, 0]**2)
    a2.plot(th, vr, color=COLORES[0], label="relativista: v = u/γ")
    a2.plot(th, Vn[:, 0], "--", color=COLORES[1], label="Newton: v = qE₀t/m")
    a2.axhline(1, color="k", lw=1)
    a2.set(xlabel="t (unidades de mc/qE₀)", ylabel="v / c", ylim=(0, 2.5), title="campo eléctrico uniforme")
    a2.legend(fontsize=9)
    fig.tight_layout()
    if exportar:
        guardar(fig, "nb29_uniforme")
    plt.show()

mostrar_campo_uniforme(exportar=bool(CARPETA_FIGURAS))

def cruzados(E=0.5, T=60.0, dt=5e-3):
    return boris_rel([0, 0, 0], [0, 0, 0], campos_uniformes([0, E, 0], [0, 0, 1.0]), dt=dt, pasos=int(T / dt))

def mostrar_cruzados(E=0.5, exportar=False):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, e in zip(axs, (E, 1.6)):
        X, U = cruzados(e, T=30.0 if e < 1 else 12.0)
        ax.plot(X[:, 0], X[:, 1], color=COLORES[0] if e < 1 else COLORES[1])
        g = np.sqrt(1 + (U**2).sum(axis=1))
        ax.set(xlabel="x  (deriva según E×B)", ylabel="y  (según E)",
               title=f"E/B = {e:g}: γ máximo = {g.max():.1f}" + ("  (acotado)" if e < 1 else "  (sin límite)"))
    fig.tight_layout()
    if exportar:
        guardar(fig, "nb29_cruzados")
    plt.show()

interactuar(mostrar_cruzados, E=deslizador("E/B", 0.5, 0.05, 0.95, 0.05))
if CARPETA_FIGURAS:
    mostrar_cruzados(0.5, exportar=True)

# Comprobaciones
dt, T = 1e-3, 5.0
u_ini = [-0.5 * dt, 0, 0]                                # u en t = −dt/2: la velocidad va medio paso atrás
Xr, Ur = boris_rel([0, 0, 0], u_ini, campos_uniformes([1.0, 0, 0], [0, 0, 0]), dt=dt, pasos=int(T / dt))
verificar("campo uniforme: x(T) = √(1 + T²) − 1 (movimiento hiperbólico, T = 5mc/qE₀)", Xr[-1, 0], np.sqrt(1 + T**2) - 1, tol=1e-6)
th = dt * (np.arange(len(Ur)) + 0.5); gam = np.sqrt(1 + Ur[:, 0]**2)
tau = cumulative_trapezoid(1 / gam, th, initial=0) + th[0] / gam[0]
verificar("la rapidez crece linealmente con el tiempo propio: ζ = qE₀τ/mc", np.polyfit(tau, np.arcsinh(Ur[:, 0]), 1)[0], 1.0, tol=1e-6)
X, U = boris_rel([0, 0, 0], [0.3, 2.0, -0.7], campos_uniformes([0, 0, 0], [0.2, -0.4, 1.0]), dt=0.05, pasos=20000)
verificar("solo B: |u| (y la energía) se conserva en 20000 pasos", np.ptp(np.linalg.norm(U, axis=1)), 0.0, tol=1e-12)
for e in (0.5, 1.6):
    X, U = cruzados(e, T=60.0 if e < 1 else 12.0, dt=5e-3)
    yh = 0.5 * (X[1:, 1] + X[:-1, 1])                      # y en los mismos tiempos que u
    g = np.sqrt(1 + (U**2).sum(axis=1))
    verificar(f"E/B = {e}: p_x − (qB/c) y se conserva (variación relativa)", np.ptp(U[:, 0] - yh) / np.abs(U[:, 0]).max(), 0.0, tol=1e-3)
    verificar(f"E/B = {e}: 𝓔 − qE y se conserva (variación relativa)", np.ptp(g - e * yh) / g.max(), 0.0, tol=1e-3)
X, U = cruzados(0.5, T=30.0, dt=1e-3)
verificar("E/B = 0.5 desde el reposo: la altura máxima es y = 2mc²E/q(B² − E²) = 4/3", X[:, 1].max(), 2 * 0.5 / (1 - 0.25), tol=1e-4)
X, U = cruzados(0.5, T=400.0, dt=1e-2)
t = 1e-2 * np.arange(len(X))
verificar("E/B = 0.5: la deriva media es cE/B (acotado; la transformación de campos de la Clase 30 lo explica)", np.polyfit(t, X[:, 0], 1)[0], 0.5, tol=1e-2)

# %% [markdown]
# ### ¿Qué pasó?
# Con un campo uniforme, Newton da $v=qE_0t/m$, que supera a $c$ en $t=mc/qE_0$: a $t=5mc/qE_0$ va a $5c$. La ecuación relativista da $\gamma m\mathbf v=q\mathbf E_0t$: el momento crece sin límite, pero la velocidad tiende a $c$, y la línea de mundo es una hipérbola que se acerca a la diagonal de la luz. La rapidez crece a ritmo constante en el tiempo propio de la carga, como el cohete de la Clase 28.
#
# En campos cruzados con $E<B$, la carga gira y avanza en la dirección $\mathbf E\times\mathbf B$ con velocidad media $cE/B$, la deriva de la Clase 14, y su energía queda acotada. Con $E>B$ no hay deriva posible ($cE/B>c$): la carga gana energía sin límite y se aleja casi a $c$. En los dos casos se conservan $p_x-\frac{qB}{c}y$ y $\mathcal E-qEy$; la diferencia es que, si $E>B$, la combinación $\mathcal E-\frac EBcp_x$ ya no acota a $\mathcal E$ (sección 2 de las notas).

# %% [markdown]
# ## Experimento 2 — La acción es estacionaria en la trayectoria real
#
# En una dimensión, con un campo uniforme $E_0$ ($\phi=-E_0x$), la acción relativista es (sección 4 de las notas)
# $$S[x]=\int_0^T\left(-mc^2\sqrt{1-\dot x^2/c^2}+qE_0x\right)dt .$$
# Tomamos la trayectoria real, el movimiento hiperbólico desde el reposo, que llega a $X=x(T)$, y otra con los mismos extremos: la parábola de Newton, $x=v_0t+\frac12\frac{qE_0}{m}t^2$, con $v_0$ elegida para llegar a $X$ en $T$. A cada una le sumamos $\epsilon\,\eta(t)$, con $\eta=\sin(\pi t/T)$, que se anula en los extremos, y calculamos $S(\epsilon)$.
#
# ### Predecí
# ¿Cuál de las dos curvas $S(\epsilon)$ tiene su mínimo en $\epsilon=0$?

# %%
T_acc = 0.8                                             # en unidades de mc/qE₀; Newton no llega a c
t_acc = np.linspace(0, T_acc, 20001)
X_fin = np.sqrt(1 + T_acc**2) - 1
caminos = {
    "real (hiperbólico)": (np.sqrt(1 + t_acc**2) - 1, t_acc / np.sqrt(1 + t_acc**2)),
    "Newton, mismos extremos": None,
}
v0 = (X_fin - 0.5 * T_acc**2) / T_acc
caminos["Newton, mismos extremos"] = (v0 * t_acc + 0.5 * t_acc**2, v0 + t_acc)
eta = np.sin(np.pi * t_acc / T_acc); deta = (np.pi / T_acc) * np.cos(np.pi * t_acc / T_acc)

def accion(x, v):
    return trapezoid(-np.sqrt(1 - v**2) + x, t_acc)

def S_eps(nombre, eps):
    x, v = caminos[nombre]
    return accion(x + eps * eta, v + eps * deta)

eps = np.linspace(-0.06, 0.06, 121)
fig, ax = plt.subplots(figsize=(7.5, 4.2))
for (nombre, _), c in zip(caminos.items(), (COLORES[0], COLORES[1])):
    S = np.array([S_eps(nombre, e) for e in eps])
    ax.plot(eps, S - S_eps(nombre, 0.0), color=c, label=nombre)
ax.axvline(0, color="k", lw=0.8)
ax.set(xlabel="ε (amplitud de la variación)", ylabel="S(ε) − S(0)", title="la acción relativista alrededor de dos caminos")
ax.legend(); plt.show()

# Comprobaciones
h = 1e-4
d1 = {n: (S_eps(n, h) - S_eps(n, -h)) / (2 * h) for n in caminos}
verificar("trayectoria real: dS/dε = 0 (la acción es estacionaria)", d1["real (hiperbólico)"], 0.0, tol=1e-7)
print(f"   (para la parábola de Newton, dS/dε = {d1['Newton, mismos extremos']:.2e}: no es estacionaria)")
x, v = caminos["real (hiperbólico)"]
d2 = (S_eps("real (hiperbólico)", h) - 2 * S_eps("real (hiperbólico)", 0) + S_eps("real (hiperbólico)", -h)) / h**2
verificar("d²S/dε² = ∫ mγ³ η̇² dt > 0: es un mínimo", d2, trapezoid((1 - v**2)**-1.5 * deta**2, t_acc), tol=1e-4)

# %% [markdown]
# ### ¿Qué pasó?
# Para la trayectoria real, $S(\epsilon)$ tiene un mínimo en $\epsilon=0$: cualquier variación chica que respete los extremos aumenta la acción, en primer orden nada y en segundo orden $\frac12\epsilon^2\int m\gamma^3\dot\eta^2dt$. La parábola de Newton, que resuelve otra ecuación, no es un extremo de esta acción: su $S(\epsilon)$ tiene pendiente en $\epsilon=0$. Las ecuaciones de Euler–Lagrange de esta acción son $\frac{d}{dt}(\gamma m\dot x)=qE_0$, la ecuación relativista.

# %% [markdown]
# ## Experimento 3 — El cuadrimomento en un decaimiento
#
# Un pion cargado decae en un muon y un neutrino (de masa despreciable): $\pi^+\to\mu^+\nu$. En el reposo del pion, la conservación del cuadrimomento da (sección 1 de las notas) $\mathcal E_\mu^*=\frac{m_\pi^2+m_\mu^2}{2m_\pi}c^2=109.8$ MeV y $|\mathbf p^*|c=\frac{m_\pi^2-m_\mu^2}{2m_\pi}c^2=29.8$ MeV, en una dirección al azar. Simulamos muchos decaimientos de piones de $1$ GeV, transformamos los cuadrimomentos al laboratorio con un boost, y miramos las energías de los muones.
#
# ### Predecí
# En el laboratorio, ¿los muones tienen todos la misma energía? ¿Cuál es la masa invariante del par muon–neutrino?

# %%
M_PI, M_MU = 139.570, 105.658                          # MeV/c²
E_star = (M_PI**2 + M_MU**2) / (2 * M_PI); p_star = (M_PI**2 - M_MU**2) / (2 * M_PI)

def boost4(beta):
    """Transformación de Lorentz 4 × 4 que lleva al sistema que se mueve con velocidad beta·c según z (la inversa: −beta)."""
    g = 1 / np.sqrt(1 - beta**2); L = np.eye(4)
    L[0, 0] = L[3, 3] = g; L[0, 3] = L[3, 0] = -g * beta
    return L

def decaimientos(E_pi=1000.0, N=20000, semilla=3):
    rng = np.random.default_rng(semilla)
    ct = np.concatenate([[-1.0, 1.0], rng.uniform(-1, 1, N - 2)]); ph = rng.uniform(0, 2 * np.pi, N)
    st = np.sqrt(1 - ct**2)
    pmu = np.stack([np.full(N, E_star), p_star * st * np.cos(ph), p_star * st * np.sin(ph), p_star * ct])   # (𝓔/c, p) en MeV
    pnu = np.stack([np.full(N, p_star), -pmu[1], -pmu[2], -pmu[3]])
    beta = np.sqrt(1 - (M_PI / E_pi)**2)
    L = boost4(-beta)                                    # del reposo del pion al laboratorio
    return L @ pmu, L @ pnu, beta

def masa(p):
    return np.sqrt(np.maximum(p[0]**2 - p[1]**2 - p[2]**2 - p[3]**2, 0))

def mostrar_decaimientos(E_pi=1000.0):
    pmu, pnu, beta = decaimientos(E_pi)
    g = 1 / np.sqrt(1 - beta**2)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.hist(pmu[0], bins=60, color=COLORES[0], alpha=0.8)
    for lim in (g * (E_star - beta * p_star), g * (E_star + beta * p_star)):
        ax.axvline(lim, color="k", ls="--", lw=1)
    ax.set(xlabel="energía del muon en el laboratorio (MeV)", ylabel="cuentas", title=f"π⁺ → μ⁺ν con piones de {E_pi:g} MeV")
    plt.show()

interactuar(mostrar_decaimientos, E_pi=deslizador("𝓔_π (MeV)", 1000.0, 150.0, 5000.0, 50.0))

# Comprobaciones
pmu, pnu, beta = decaimientos()
verificar("la masa invariante del par μν en el laboratorio es m_π (error relativo máximo)", np.abs(masa(pmu + pnu) - M_PI).max() / M_PI, 0.0, tol=1e-12)
verificar("el muon conserva su masa al cambiar de sistema", np.abs(masa(pmu) - M_MU).max() / M_MU, 0.0, tol=1e-9)
verificar("el neutrino sin masa tiene 𝓔 = |p|c en el laboratorio", np.abs(pnu[0] - np.linalg.norm(pnu[1:], axis=0)).max() / pnu[0].max(), 0.0, tol=1e-12)
g = 1 / np.sqrt(1 - beta**2)
verificar("la energía máxima del muon es γ(𝓔* + βp*c)", pmu[0].max(), g * (E_star + beta * p_star), tol=1e-12)

# %% [markdown]
# ### ¿Qué pasó?
# En el reposo del pion, todos los muones tienen la misma energía, $109.8$ MeV, y salen en cualquier dirección. En el laboratorio, la energía depende del ángulo, $\mathcal E_\mu=\gamma(\mathcal E^*+\beta p^*c\cos\theta^*)$, y como $\cos\theta^*$ es uniforme, la distribución es plana entre $\gamma(\mathcal E^*\mp\beta p^*c)$. La masa invariante del par, $\sqrt{(p_\mu+p_\nu)^2}$, es la del pion en cualquier sistema: así se descubren partículas que decaen antes de llegar al detector, buscando picos en la masa invariante de sus productos.

# %% [markdown]
# ## Explorá
#
# 1. **Guía 12, problema 3.** Usá `boris_rel` con solo un campo magnético uniforme y distintos $|\mathbf u_0|$, medí el período y el radio de la órbita, y comparalos con tus resultados. ¿Qué pasa cuando $|\mathbf u_0|\ll c$? Compará con el ciclotrón de la Clase 14.
# 2. **Guía 12, problema 4.** `boost4(beta)` transforma cuadrivectores $(A^0,A^x,A^y,A^z)$ a un sistema que se mueve según $z$. Aplicala a la cuadricorriente de una densidad de carga quieta, $(c\rho_0,0,0,0)$, y a la de un hilo neutro con corriente según $z$, y compará con tu interpretación.
# 3. **El cable de la Clase 12.** Con $\lambda'=\lambda_0(\gamma_u-1/\gamma_u)$ y $F'=2q\lambda'/s$, comprobá para $u$ entre $0.1\,c$ y $0.99\,c$ que $F'=\gamma_uF_{\text{lab}}$, con $F_{\text{lab}}=2q\lambda_0u^2/c^2s$, como dice la transformación de la fuerza de la sección 2 de las notas.
