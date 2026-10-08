# %% [markdown]
# # Clase 27 — Ondas en medios: reflexión, refracción, conductores y dispersión
#
# **Objetivos**
# - Resolver numéricamente las condiciones de borde en una interfaz y ver cuánto se refleja para cada polarización y cada ángulo: el ángulo de Brewster y la reflexión total.
# - Ver los frentes de onda que se quiebran al cruzar una interfaz, y la onda evanescente de la reflexión total.
# - Calcular la reflexión de capas delgadas con la matriz de transferencia: una capa antirreflejo, un espejo de Bragg, el túnel óptico y una película metálica.
# - Seguir pulsos en el tiempo: uno que llega a un vidrio y uno que viaja en un plasma con la velocidad de grupo.
# - Comparar la velocidad de fase con la de grupo, y ver cómo se ensancha un paquete en un medio dispersivo.
#
# **Material relacionado:** notas de la Clase 27. Guía 11: problemas 3, 4 y 6.
#
# **Unidades.** Gaussianas, adimensionales: $c=1$; en los experimentos 1 y 2 las longitudes van en unidades de $\lambda_0/2\pi=c/\omega$ (la del vacío), y en el 3 y el 4, en unidades de $c/\omega_p$ cuando hay plasma.

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
from scipy.signal import hilbert

# %% [markdown]
# ## Experimento 1 ★ — Fresnel: cuánto se refleja
#
# La interfaz es el plano $z=0$; el medio 1 (índice $n_1$) está en $z<0$ y el 2 ($n_2$) en $z>0$, con $\mu=1$. El plano de incidencia es el $yz$, como en las notas. Cada onda tiene su $\mathbf k$ (las tres con la misma frecuencia y la misma $k_y$, sección 2 de las notas) y su campo eléctrico, transversal a $\mathbf k$, se escribe con dos componentes: según $\hat{\mathbf x}$ (polarización **s** o TE, perpendicular al plano de incidencia) y según $\hat{\mathbf p}=\hat{\mathbf x}\times\mathbf k/(nk_0)$ (polarización **p** o TM, en el plano). Con $\mathbf H=\mathbf B=\frac{c}{\omega}\mathbf k\times\mathbf E$, las cuatro condiciones de borde ($E_x$, $E_y$, $H_x$ y $H_y$ continuas) son cuatro ecuaciones lineales para las cuatro amplitudes desconocidas, la reflejada y la transmitida en cada polarización. `fresnel_numerico` las resuelve, sin usar ninguna fórmula de Fresnel, y calcula $R$ y $T$ con el flujo de $\langle\mathbf S\rangle$ a través de la interfaz. Si $n_2<n_1\sin\theta$, $k_{2z}$ sale imaginario: la onda evanescente.
#
# ### Predecí
# 1. Del aire al vidrio ($n=1.5$), ¿cómo cambia $R$ con el ángulo para cada polarización? ¿Alguna llega a cero?
# 2. Del vidrio al aire, ¿qué pasa a partir de cierto ángulo?

# %%
def fresnel_numerico(n1, n2, theta, pol=(1.0, 0.0)):
    """Resuelve las condiciones de borde en z = 0 para una onda que llega desde el medio 1 (z < 0) con ángulo theta.
    pol = (a, b): componentes del E incidente según x̂ (s) y según p̂ = x̂ × k/(n k0) (p).
    Para un medio con ε complejo, pasar n2 = np.sqrt(ε + 0j).
    Devuelve las amplitudes (r_s, r_p, t_s, t_p) en las bases de cada onda, y R y T (flujos de energía según z)."""
    k0 = 1.0
    ky = n1 * k0 * np.sin(theta); kz1 = n1 * k0 * np.cos(theta)
    kz2 = np.sqrt(complex(n2**2 * k0**2 - ky**2))          # Im > 0: si es evanescente, decae hacia z > 0
    kI = np.array([0, ky, kz1], complex); kR = np.array([0, ky, -kz1], complex); kT = np.array([0, ky, kz2], complex)
    x = np.array([1, 0, 0], complex)
    p = lambda k, n: np.cross(x, k) / (n * k0)
    H = lambda k, E: np.cross(k, E) / k0                     # H = B = (c/ω) k × E, con c/ω = 1/k0
    EI = pol[0] * x + pol[1] * p(kI, n1)
    columnas = []
    for vec, k, signo in ((x, kR, 1), (p(kR, n1), kR, 1), (x, kT, -1), (p(kT, n2), kT, -1)):
        E = signo * vec; Hv = H(k, E)
        columnas.append([E[0], E[1], Hv[0], Hv[1]])
    HI = H(kI, EI)
    rs, rp, ts, tp = np.linalg.solve(np.array(columnas).T, -np.array([EI[0], EI[1], HI[0], HI[1]]))
    ER = rs * x + rp * p(kR, n1); ET = ts * x + tp * p(kT, n2)
    Sz = lambda E, k: 0.5 * np.real(np.cross(E, np.conj(H(k, E))))[2]
    return rs, rp, ts, tp, -Sz(ER, kR) / Sz(EI, kI), Sz(ET, kT) / Sz(EI, kI)

def curvas_R(n1, n2, th):
    Rs = np.array([fresnel_numerico(n1, n2, t, (1, 0))[4] for t in th])
    Rp = np.array([fresnel_numerico(n1, n2, t, (0, 1))[4] for t in th])
    return Rs, Rp

def mostrar_fresnel(n=1.5, exportar=False):
    th = np.radians(np.linspace(0, 89.9, 400))
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.3))
    for ax, (n1, n2) in zip(axs, ((1.0, n), (n, 1.0))):
        Rs, Rp = curvas_R(n1, n2, th)
        ax.plot(np.degrees(th), Rs, color=COLORES[0], label="R_s (E perpendicular al plano)")
        ax.plot(np.degrees(th), Rp, color=COLORES[1], label="R_p (E en el plano)")
        tb = np.degrees(np.arctan(n2 / n1))
        ax.axvline(tb, color=COLORES[1], ls=":", lw=1.2); ax.text(tb + 1, 0.9, f"Brewster\n{tb:.1f}°", color=COLORES[1], fontsize=9)
        if n2 < n1:
            tc = np.degrees(np.arcsin(n2 / n1))
            ax.axvline(tc, color="k", ls="--", lw=1); ax.text(tc + 1, 0.5, f"crítico\n{tc:.1f}°", fontsize=9)
        ax.set(xlabel="ángulo de incidencia θ (grados)", ylabel="R", ylim=(0, 1.02), xlim=(0, 90), title=f"de n₁ = {n1:g} a n₂ = {n2:g}")
        ax.legend(loc="center left", fontsize=9)
    if exportar:
        guardar(fig, "nb27_fresnel")
    plt.show()

interactuar(mostrar_fresnel, n=deslizador("n", 1.5, 1.05, 4.0, 0.05))
if CARPETA_FIGURAS:
    mostrar_fresnel(1.5, exportar=True)

# %% [markdown]
# Ahora miramos el campo. Para la polarización s, $E_x(y,z,0)$ es la parte real de $e^{i(k_yy+k_{1z}z)}+r_se^{i(k_yy-k_{1z}z)}$ en el medio 1 y de $t_se^{i(k_yy+k_{2z}z)}$ en el 2.

# %%
def campo_s(n1, n2, theta, Y, Z):
    rs, _, ts, _, _, _ = fresnel_numerico(n1, n2, theta, (1, 0))
    ky = n1 * np.sin(theta); kz1 = n1 * np.cos(theta); kz2 = np.sqrt(complex(n2**2 - ky**2))
    E1 = np.exp(1j * (ky * Y + kz1 * Z)) + rs * np.exp(1j * (ky * Y - kz1 * Z))
    E2 = ts * np.exp(1j * (ky * Y + kz2 * Z))
    return np.real(np.where(Z < 0, E1, E2))

def dibujar_campo(ax, n1, n2, grados):
    th = np.radians(grados)
    y = np.linspace(-15, 15, 301); z = np.linspace(-15, 15, 301)
    Y, Z = np.meshgrid(y, z, indexing="ij")
    ax.pcolormesh(Y, Z, campo_s(n1, n2, th, Y, Z), cmap="RdBu_r", vmin=-2, vmax=2, shading="auto")
    ax.axhline(0, color="k", lw=1.2)
    ax.annotate("", xy=(-8 + 5 * np.sin(th), -8 + 5 * np.cos(th)), xytext=(-8, -8), arrowprops=dict(arrowstyle="->", lw=2))
    ky = n1 * np.sin(th)
    if ky < n2:
        tt = np.arcsin(ky / n2)
        ax.annotate("", xy=(5 + 5 * np.sin(tt), 5 + 5 * np.cos(tt)), xytext=(5, 5), arrowprops=dict(arrowstyle="->", lw=2))
        tit = f"n₁ = {n1:g} → n₂ = {n2:g}, θ = {grados:g}°: refractada a {np.degrees(tt):.1f}°"
    else:
        tit = f"n₁ = {n1:g} → n₂ = {n2:g}, θ = {grados:g}°: reflexión total"
    ax.set(aspect="equal", xlabel="y (unidades de c/ω)", ylabel="z", title=tit)
    ax.grid(False)

def mostrar_campo(theta=50.0, n1=1.0, n2=1.5):
    fig, ax = plt.subplots(figsize=(6, 6))
    dibujar_campo(ax, n1, n2, theta)
    plt.show()

interactuar(mostrar_campo, theta=deslizador("θ (grados)", 50.0, 0.0, 89.0, 1.0), n1=deslizador("n₁", 1.0, 1.0, 3.0, 0.1), n2=deslizador("n₂", 1.5, 1.0, 3.0, 0.1))
if CARPETA_FIGURAS:
    fig, axs = plt.subplots(1, 2, figsize=(11, 5.6))
    dibujar_campo(axs[0], 1.0, 1.5, 50.0); dibujar_campo(axs[1], 1.5, 1.0, 50.0)
    fig.tight_layout(); guardar(fig, "nb27_campos"); plt.close(fig)

# Comprobaciones
n1, n2 = 1.0, 1.5
err = 0.0
for th in np.radians([0.0, 20.0, 45.0, 80.0]):
    ct = np.sqrt(1 - (n1 * np.sin(th) / n2)**2)
    rs_formula = (n1 * np.cos(th) - n2 * ct) / (n1 * np.cos(th) + n2 * ct)
    err = max(err, abs(fresnel_numerico(n1, n2, th, (1, 0))[0] - rs_formula))
verificar("r_s numérico = (n₁cos θ₁ − n₂cos θ₂)/(n₁cos θ₁ + n₂cos θ₂) (4 ángulos, error máximo)", err, 0.0, tol=1e-12)
suma = max(abs(sum(fresnel_numerico(a, b, np.radians(t), pol)[4:]) - 1) for a, b in ((1, 1.5), (1.5, 1)) for t in (10, 35, 40) for pol in ((1, 0), (0, 1)))
verificar("R + T = 1 en las dos polarizaciones y en los dos sentidos", suma, 0.0, tol=1e-12)
verificar("R_p = 0 en el ángulo de Brewster, tan θ_B = n₂/n₁", fresnel_numerico(1.0, 1.5, np.arctan(1.5), (0, 1))[4], 0.0, tol=1e-12)
verificar("R_s en el ángulo de Brewster no es cero: vale sen²(θ_B − θ_T) = 0.148", fresnel_numerico(1.0, 1.5, np.arctan(1.5), (1, 0))[4],
          np.sin(np.arctan(1.5) - np.arctan(1 / 1.5))**2, tol=1e-12)
th = np.radians(60.0); rs = fresnel_numerico(1.5, 1.0, th, (1, 0))[0]
a, beta = 1.5 * np.cos(th), np.sqrt(1.5**2 * np.sin(th)**2 - 1)
verificar("reflexión total (vidrio → aire, 60°): |r_s| = 1", abs(rs), 1.0, tol=1e-12)
verificar("su fase es −2 arctan(β/a), con a = n₁cos θ₁ y β = √(n₁²sen²θ₁ − n₂²)", np.angle(rs), -2 * np.arctan(beta / a), tol=1e-12)
verificar("incidencia normal en un vidrio: R = ((n − 1)/(n + 1))² = 4 %", fresnel_numerico(1.0, 1.5, 0.0, (1, 0))[4], 0.04, tol=1e-12)

# %% [markdown]
# ### ¿Qué pasó?
# Del aire al vidrio, $R_s$ crece con el ángulo, pero $R_p$ **se anula** en el ángulo de Brewster, $\tan\theta_B=n_2/n_1$ ($56.3^\circ$ para $n=1.5$): la luz reflejada a ese ángulo queda polarizada con $\mathbf E$ perpendicular al plano de incidencia. Así funcionan los anteojos polarizados, que bloquean el reflejo del agua o de la ruta. En incidencia rasante, las dos tienden a 1.
#
# Del vidrio al aire, a partir del ángulo crítico, $\sin\theta_c=n_2/n_1$ ($41.8^\circ$), **todo se refleja**, en las dos polarizaciones. En el mapa del campo, del aire al vidrio los frentes de onda se quiebran y la onda transmitida **se acerca a la normal**: la longitud de onda es menor en el vidrio, y los frentes tienen que empalmar en la interfaz. Es al revés que las líneas de campo estáticas de la Clase 25. Con reflexión total, del otro lado queda una onda que viaja a lo largo de la interfaz y decae en $z$: la onda evanescente.

# %% [markdown]
# ## Experimento 2 — Capas delgadas: la matriz de transferencia
#
# Una pila de capas planas, con índices $n_j$ y espesores $d_j$, entre un medio de entrada $n_0$ y un sustrato $n_s$. Para la polarización s, en cada capa $E_x=Ae^{ik_zz}+Be^{-ik_zz}$ (por el factor común $e^{i(k_yy-\omega t)}$), con $k_z=k_0q$ y $q=\sqrt{n_j^2-n_0^2\sin^2\theta}$. Las condiciones de borde dicen que $E_x$ y $H_y\propto\partial_zE_x$ son continuos (sección 2 de las notas), así que conviene seguir el vector $(E_x,\ \partial_zE_x/k_0)$, que no salta en las interfaces. Al atravesar una capa de espesor $d$, con $\varphi=k_0qd$,
# $$\begin{pmatrix}E\\ E'/k_0\end{pmatrix}_{z+d}=\begin{pmatrix}\cos\varphi&\sin\varphi/q\\ -q\sin\varphi&\cos\varphi\end{pmatrix}\begin{pmatrix}E\\ E'/k_0\end{pmatrix}_z ,$$
# que se comprueba con las dos soluciones $\cos(k_0qz)$ y $\sin(k_0qz)/q$. El producto de las matrices de todas las capas lleva el vector de la primera interfaz a la última. Adelante, $E=1+r$ y $E'/k_0=iq_0(1-r)$; atrás, $E=t$ y $E'/k_0=iq_st$: dos ecuaciones lineales para $r$ y $t$. Con $q$ complejo, la misma cuenta sirve para capas evanescentes y absorbentes.
#
# ### Predecí
# 1. ¿Se puede eliminar el 4 % que refleja un vidrio poniéndole encima una capa delgada de otro material?
# 2. Dos prismas de vidrio, con la luz llegando con un ángulo mayor que el crítico, separados por un espacio de aire mucho menor que $\lambda$: ¿se refleja todo?

# %%
def reflexion_capas(n, d, n0=1.0, ns=1.5, theta=0.0, k0=1.0):
    """r, t, R, T (polarización s) de una pila de capas con índices n[j] y espesores d[j], entre n0 y ns.
    Los índices pueden ser complejos (n = np.sqrt(ε + 0j))."""
    s2 = (n0 * np.sin(theta))**2
    q = lambda nj: np.sqrt(complex(nj**2 - s2))
    M = np.eye(2, dtype=complex)
    for nj, dj in zip(n, d):
        qj = q(nj); f = k0 * qj * dj
        M = np.array([[np.cos(f), np.sin(f) / qj], [-qj * np.sin(f), np.cos(f)]]) @ M
    q0, qs = q(n0), q(ns)
    # M [1 + r, i q0 (1 − r)] = [t, i qs t]  →  incógnitas (r, t)
    A = np.array([[M[0, 0] - 1j * q0 * M[0, 1], -1], [M[1, 0] - 1j * q0 * M[1, 1], -1j * qs]])
    b = -np.array([M[0, 0] + 1j * q0 * M[0, 1], M[1, 0] + 1j * q0 * M[1, 1]])
    r, t = np.linalg.solve(A, b)
    return r, t, abs(r)**2, np.real(qs) / np.real(q0) * abs(t)**2

def figura_capas(N=6, exportar=False):
    ng = 1.5; nc = np.sqrt(ng); nH, nL = 2.3, 1.38
    lam = np.linspace(0.4, 2.0, 600)                                # λ/λ0
    k0 = 1.0 / lam                                                   # en unidades de 2π/λ0 → fase k0 n d con d en λ0/2π
    fig, axs = plt.subplots(2, 2, figsize=(12, 7.5))
    ax = axs[0, 0]
    ax.plot(lam, [reflexion_capas([], [], ns=ng, k0=k)[2] for k in k0], color=COLORES[0], label="vidrio solo")
    ax.plot(lam, [reflexion_capas([nc], [np.pi / (2 * nc)], ns=ng, k0=k)[2] for k in k0], color=COLORES[1], label=f"con capa n = √1.5, d = λ₀/4n")
    ax.set(xlabel="λ / λ₀", ylabel="R", title="capa antirreflejo"); ax.legend(fontsize=9)
    ax = axs[0, 1]
    capas = [nH, nL] * N; esp = [np.pi / (2 * nH), np.pi / (2 * nL)] * N
    ax.plot(lam, [reflexion_capas(capas, esp, ns=ng, k0=k)[2] for k in k0], color=COLORES[2])
    ax.set(xlabel="λ / λ₀", ylabel="R", title=f"espejo de Bragg: {N} pares de capas λ₀/4 (n = {nH}, {nL})")
    ax = axs[1, 0]
    th = np.radians(50.0); dd = np.linspace(0, 3.0, 300)
    ax.semilogy(dd / (2 * np.pi), [reflexion_capas([1.0], [x], n0=1.5, ns=1.5, theta=th)[3] for x in dd], color=COLORES[3])
    ax.set(xlabel="espesor del aire / λ₀", ylabel="T", title="túnel óptico: vidrio | aire | vidrio a 50° (> 41.8°)")
    ax = axs[1, 1]
    nm = np.sqrt(1 + 4j * np.pi * 1e3)                               # un buen conductor: ε = 1 + 4πiσ/ω con σ/ω = 1000
    dm = np.linspace(0, 0.2, 200)
    ax.semilogy(dm / (2 * np.pi), [reflexion_capas([nm], [x], ns=1.0)[3] for x in dm], color=COLORES[4])
    ax.set(xlabel="espesor / λ₀", ylabel="T", title="película de un buen conductor (σ/ω = 1000)")
    fig.tight_layout()
    if exportar:
        guardar(fig, "nb27_capas")
    plt.show()

interactuar(figura_capas, N=IntSlider(value=6, min=1, max=15, description="pares", continuous_update=False))
if CARPETA_FIGURAS:
    figura_capas(6, exportar=True)

# Comprobaciones
th = np.radians(30.0)
verificar("sin capas, la matriz de transferencia da el r_s de una interfaz", abs(reflexion_capas([], [], n0=1.0, ns=1.5, theta=th)[0] - fresnel_numerico(1.0, 1.5, th, (1, 0))[0]), 0.0, tol=1e-12)
nc = np.sqrt(1.5)
verificar("capa antirreflejo n = √n_vidrio, d = λ₀/4n: R = 0 en λ₀", reflexion_capas([nc], [np.pi / (2 * nc)], ns=1.5)[2], 0.0, tol=1e-12)
rng = np.random.default_rng(1)
_, _, R, T = reflexion_capas(rng.uniform(1, 3, 8), rng.uniform(0, 3, 8), ns=1.7, theta=0.4)
verificar("ocho capas transparentes al azar: R + T = 1", R + T, 1.0, tol=1e-12)
th = np.radians(50.0); kap = np.sqrt(1.5**2 * np.sin(th)**2 - 1)
dd = np.array([10.0, 14.0]); lnT = np.log([reflexion_capas([1.0], [x], n0=1.5, ns=1.5, theta=th)[3] for x in dd])
verificar("túnel óptico: para un espacio grueso, T ∝ e^{−2κd}, con κ = k₀√(n₁²sen²θ − 1)", -(lnT[1] - lnT[0]) / (dd[1] - dd[0]), 2 * kap, tol=1e-3)
nm = np.sqrt(1 + 4j * np.pi * 1e3); delta = 1 / np.sqrt(2 * np.pi * 1e3)    # δ = c/√(2πσω), con c = ω = 1
dm = np.array([0.1, 0.15]); lnT = np.log([reflexion_capas([nm], [x], ns=1.0)[3] for x in dm])
verificar("película conductora: T ∝ e^{−2d/δ}, con δ = c/√(2πσω)", -(lnT[1] - lnT[0]) / (dm[1] - dm[0]), 2 / delta, tol=2e-3)

# %% [markdown]
# ### ¿Qué pasó?
# - **Antirreflejo.** Una capa de índice $\sqrt{n_{\text{vidrio}}}$ y espesor $\lambda_0/4n$ anula la reflexión en $\lambda_0$: las ondas reflejadas en sus dos caras salen con la misma amplitud y desfasadas $\pi$ (sección 2 de las notas). En otras longitudes de onda la cancelación es parcial; por eso los lentes con capa antirreflejo se ven violáceos: reflejan algo de rojo y de azul.
# - **Espejo de Bragg.** Muchas capas de $\lambda_0/4$ alternadas suman en fase las reflexiones de todas las interfaces, y alrededor de $\lambda_0$ aparece una banda donde $R\to1$, sin metal y sin absorción. Así son los espejos de los láseres y los colores de las alas de algunas mariposas.
# - **Túnel óptico.** Con reflexión total, la onda evanescente llega al otro lado si el espacio es más chico que $1/\kappa$, y una parte de la energía pasa: $T$ cae como $e^{-2\kappa d}$. Es la versión óptica del efecto túnel.
# - **Película conductora.** La onda decae en la profundidad de penetración $\delta$: casi toda se refleja, y lo que entra se atenúa como $e^{-d/\delta}$ en amplitud.

# %% [markdown]
# ## Experimento 3 — Pulsos: una interfaz y un plasma
#
# Seguimos pulsos en el tiempo con el esquema de Yee de la Clase 18, en la versión 2 de la Clase 26, que admite un medio ($\varepsilon$ en los nodos, y electrones de Drude). Primero, un pulso gaussiano que llega desde el vacío a un vidrio con $\varepsilon=2.25$ ($n=1.5$), en incidencia normal. Después, un pulso con una portadora de frecuencia $\omega_0=2\omega_p$ que entra a un plasma: según las notas (sección 5), la envolvente viaja con la velocidad de grupo $v_g=c\sqrt{1-\omega_p^2/\omega_0^2}$, y las crestas con la de fase, $v_\varphi=c^2/v_g$.
#
# ### Predecí
# 1. ¿Con qué amplitud vuelve el pulso reflejado en el vidrio, y con qué signo? ¿El transmitido es más largo o más corto?
# 2. En el plasma, ¿las crestas avanzan más rápido o más despacio que la envolvente?

# %%
# herramienta: fdtd1d v2 (NB26)
def yee1d(E, B, pasos, S=1.0, dx=1.0, J=None, cada=1, absorbente=True, eps=None, wp2=None, gamma=0.0):
    """Ecuaciones de Maxwell en 1D con el esquema de Yee, en unidades con c = 1, con un medio opcional.

    E: E_y en los nodos x_i = i dx, en t = 0 (N valores).
    B: B_z en los puntos medios x_{i+1/2}, en t = −dt/2 (N − 1 valores).
    dt = S dx, con S ≤ 1 (número de Courant). J(n): J_y externa en los nodos en t = (n + 1/2) dt (opcional).
    Medio (v2): eps = permitividad en los nodos (número o arreglo; 1 si se omite); wp2 = ωp² de los electrones
    libres en los nodos y gamma su amortiguamiento (números o arreglos). La corriente de los electrones, Je,
    en t = (n + 1/2) dt, cumple dJe/dt = −γ Je + (ωp²/4π) E (diferencias centradas). Con plasma, el esquema
    es estable si S² + (ωp dt/2)² ≤ 1.
    Actualiza ∂B_z/∂t = −∂E_y/∂x y ε ∂E_y/∂t = −∂B_z/∂x − 4π (J_y + Je), con bordes absorbentes de Mur
    (exactos si S = 1; el medio no debe llegar a los bordes).
    Devuelve las historias (Es, Bs, ts), guardadas cada `cada` pasos (t es el tiempo de E; B va medio paso atrás).
    """
    E = np.array(E, float); B = np.array(B, float); dt = S * dx
    k = (S - 1) / (S + 1)
    eps = np.ones_like(E) * (1.0 if eps is None else eps)
    if wp2 is not None:
        g = np.ones_like(E) * gamma
        a1 = (1 - g * dt / 2) / (1 + g * dt / 2); a2 = dt / (4 * np.pi) * (np.ones_like(E) * wp2) / (1 + g * dt / 2)
        Je = np.zeros_like(E)
    Es, Bs, ts = [E.copy()], [B.copy()], [0.0]
    for n in range(pasos):
        B -= S * (E[1:] - E[:-1])
        E0, E1, Ef, Ef1 = E[0], E[1], E[-1], E[-2]
        if wp2 is not None:
            Je = a1 * Je + a2 * E
            E -= 4 * np.pi * dt * Je / eps
        E[1:-1] -= S * (B[1:] - B[:-1]) / eps[1:-1]
        if J is not None:
            E -= 4 * np.pi * dt * J(n) / eps
        if absorbente:
            E[0] = E1 + k * (E[1] - E0)
            E[-1] = Ef1 + k * (E[-2] - Ef)
        if (n + 1) % cada == 0:
            Es.append(E.copy()); Bs.append(B.copy()); ts.append((n + 1) * dt)
    return np.array(Es), np.array(Bs), np.array(ts)
# fin herramienta

def pulso_vidrio(n=1.5, ancho=2.0, tfin=50.0):
    """Un pulso gaussiano que viaja hacia +x llega a un vidrio en x > 60 (S = 1, c = 1). Devuelve x y las historias."""
    dx = 0.05; x = np.arange(0, 120, dx)
    f = lambda u: np.exp(-((u - 30) / ancho)**2)
    eps = np.where(x >= 60, n**2, 1.0)
    Es, _, ts = yee1d(f(x), f(x[1:]), int(tfin / dx), S=1.0, dx=dx, cada=int(5 / dx), eps=eps)
    return x, Es, ts

def pulso_plasma(w0=2.0, tau=6.0, tfin=200.0, cada_t=0.4):
    """Un pulso con portadora w0 (en unidades de ωp) que entra desde el vacío a un plasma en x > 40."""
    S = 0.9; dx = 2 * np.pi / (40 * w0); dt = S * dx
    x = np.arange(0, 260, dx); xm = x[:-1] + dx / 2
    f = lambda u: np.exp(-(u - 20)**2 / (2 * tau**2)) * np.cos(w0 * (u - 20))
    Es, _, ts = yee1d(f(x), f(xm + dt / 2), int(tfin / dt), S=S, dx=dx, cada=max(1, int(cada_t / dt)), wp2=np.where(x > 40, 1.0, 0.0))
    return x, Es, ts

def mostrar_pulsos(exportar=False):
    x, Es, ts = pulso_vidrio()
    xp, Ep, tp = pulso_plasma()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    for j, c in zip((0, 4, 7, 10), RAMPA):
        a1.plot(x, Es[j], color=c, lw=1.4, label=f"t = {ts[j]:.0f}")
    a1.axvspan(60, 120, color=COLORES[2], alpha=0.12, lw=0); a1.text(62, 0.9, "vidrio, n = 1.5")
    a1.set(xlabel="x", ylabel="E", xlim=(10, 100), title="un pulso que llega a un vidrio"); a1.legend(fontsize=8, loc="center left")
    vg = np.sqrt(1 - 1 / 4.0); t1, t2 = 100.0, 136.0
    st = (tp >= t1) & (tp <= t2); sx = (xp > 95) & (xp < 160)
    a2.pcolormesh(tp[st], xp[sx], Ep[np.ix_(st, sx)].T, cmap="RdBu_r", vmin=-1, vmax=1, shading="auto", rasterized=True)
    tt = np.array([t1, t2]); xc = 40 + vg * (tt - 20)                 # el centro entra a x = 40 en t = 20
    a2.plot(tt, xc, "k--", lw=1.8, label=f"envolvente: v_g = {vg:.3f} c")
    j = np.argmin(abs(tp - t1)); cerca = abs(xp - xc[0]) < 2.0
    xcr = xp[cerca][np.argmax(Ep[j][cerca])]                           # una cresta cerca del centro
    a2.plot(tt, xcr + (tt - t1) / vg, "k:", lw=2.2, label=f"una cresta: v_φ = {1 / vg:.3f} c")
    a2.set(xlabel="t (unidades de 1/ωp)", ylabel="x (unidades de c/ωp)", xlim=(t1, t2), ylim=(95, 160), title="un pulso en un plasma (ω₀ = 2ωp): E(x, t)")
    a2.legend(fontsize=8, loc="upper left", framealpha=0.9); a2.grid(False)
    fig.tight_layout()
    if exportar:
        guardar(fig, "nb27_pulsos")
    plt.show()

mostrar_pulsos(exportar=bool(CARPETA_FIGURAS))

# Comprobaciones
x, Es, ts = pulso_vidrio()
fin = Es[-1]; aire = x < 60; vid = x >= 60
n = 1.5
verificar("pulso reflejado: amplitud r = (1 − n)/(1 + n) = −0.2", fin[aire].min(), (1 - n) / (1 + n), tol=3e-3)
verificar("pulso transmitido: amplitud t = 2/(1 + n) = 0.8", fin[vid].max(), 2 / (1 + n), tol=3e-3)
ancho = lambda u, E: np.sqrt(trapezoid((u - trapezoid(u * E**2, u) / trapezoid(E**2, u))**2 * E**2, u) / trapezoid(E**2, u))
verificar("el pulso transmitido es n veces más corto (viaja a c/n)", ancho(x[vid], fin[vid]) / ancho(x, Es[0]), 1 / n, tol=5e-3)
xp, Ep, tp = pulso_plasma()
pico = np.array([xp[np.argmax(np.where(xp > 50, abs(hilbert(E)), 0))] for E in Ep])
sel = tp > 80
verificar("en el plasma, la envolvente viaja con v_g = c√(1 − ωp²/ω₀²)", np.polyfit(tp[sel], pico[sel], 1)[0], np.sqrt(1 - 1 / 4.0), tol=2e-3)

# %% [markdown]
# ### ¿Qué pasó?
# El pulso reflejado en el vidrio vuelve **invertido**, con amplitud $-0.2$: $r=(n_1-n_2)/(n_1+n_2)$ es negativo si la onda va hacia el medio de mayor índice, como en el espejo metálico de la Clase 20 ($r=-1$). El transmitido tiene amplitud $0.8$ y es $1.5$ veces más corto, porque viaja a $c/n$. (Las amplitudes no suman 1: lo que se conserva es la energía, $R+T=0.04+0.96$.)
#
# En el diagrama espacio-tiempo del plasma, las crestas (las franjas) son más empinadas que la envolvente: **las crestas avanzan más rápido que el pulso**, con $v_\varphi>c$, y lo atraviesan de atrás hacia adelante. La envolvente, que lleva la energía, viaja con $v_g<c$, y $v_gv_\varphi=c^2$.

# %% [markdown]
# ## Experimento 4 — Velocidad de fase y velocidad de grupo
#
# Un paquete es una suma de ondas planas con distintos $k$. Si $\omega(k)$ no es proporcional a $k$, cada una viaja con su velocidad de fase, y el paquete cambia. Retomamos un ejercicio de guías anteriores: el tren de pulsos
# $$f(x,t)=\sum_{k=5}^{25}\cos\left(kx-\omega(k)t\right)e^{-(k-15)^2/20},$$
# con tres relaciones de dispersión: $\omega=k$ (sin dispersión), $\omega=k+10$ y $\omega=\sqrt k$. La envolvente es $|F|$, con $F=\sum_ke^{i(kx-\omega t)}e^{-(k-15)^2/20}$.
#
# ### Predecí
# Con $\omega=k+10$, ¿con qué velocidad avanza la envolvente? ¿Y las crestas? ¿Y con $\omega=\sqrt k$?

# %%
KS = np.arange(5, 26); PESOS = np.exp(-(KS - 15)**2 / 20)
DISPERSION = {"ω = k": lambda k: k * 1.0, "ω = k + 10": lambda k: k + 10.0, "ω = √k": lambda k: np.sqrt(k)}

def tren(x, t, w):
    """F(x, t) = Σ_k e^{i(kx − ω(k)t)} e^{−(k−15)²/20}; el campo es su parte real y la envolvente, su módulo."""
    return (PESOS[None, :] * np.exp(1j * (KS[None, :] * x[:, None] - w(KS)[None, :] * t))).sum(axis=1)

def mostrar_trenes(t=1.0):
    x = np.linspace(-np.pi, np.pi, 1500)
    fig, axs = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    for ax, (nombre, w) in zip(axs, DISPERSION.items()):
        F0 = tren(x, 0.0, w); F = tren(x, t, w)
        ax.plot(x, F0.real, color="gray", lw=0.8, alpha=0.6, label="t = 0")
        ax.plot(x, F.real, color=COLORES[0], lw=1.2, label=f"t = {t:g}")
        ax.plot(x, abs(F), color=COLORES[1], lw=1.5)
        vg = (w(15.0 + 1e-4) - w(15.0 - 1e-4)) / 2e-4; vf = w(15.0) / 15.0
        ax.set(ylabel="f", title=f"{nombre}:  v_φ = {vf:.3f},  v_g = {vg:.3f}  (en k = 15)")
        ax.legend(loc="upper right", fontsize=8)
    axs[-1].set_xlabel("x")
    fig.tight_layout(); plt.show()

interactuar(mostrar_trenes, t=deslizador("t", 1.0, 0.0, 6.0, 0.05))

# %% [markdown]
# Ahora un paquete gaussiano en un plasma, $\omega=\sqrt{\omega_p^2+c^2k^2}$ (con $c=\omega_p=1$), centrado en $k_0=\sqrt3$ ($\omega_0=2$), con $A(k)=e^{-(k-k_0)^2\ell^2/2}$ y $\ell=8$. Lo sumamos exactamente, y comparamos su ancho con el de las notas (sección 5): $|E|^2\propto e^{-\xi^2/w^2}$, con $w(t)^2=\ell^2+(\omega''t/\ell)^2$ y $\omega''=c^2\omega_p^2/\omega_0^3$.

# %%
def paquete_plasma(t, sigma=8.0, k0=np.sqrt(3.0)):
    k = np.linspace(k0 - 8 / sigma, k0 + 8 / sigma, 1201)
    A = np.exp(-(k - k0)**2 * sigma**2 / 2); w = np.sqrt(1 + k**2)
    vg = k0 / np.sqrt(1 + k0**2)
    xi = np.linspace(-120, 120, 3001); x = vg * t + xi
    F = trapezoid(A[None, :] * np.exp(1j * (k[None, :] * x[:, None] - w[None, :] * t)), k, axis=1)
    return xi, F

def ancho_paquete(xi, F):
    I = abs(F)**2; m = trapezoid(xi * I, xi) / trapezoid(I, xi)
    return np.sqrt(2 * trapezoid((xi - m)**2 * I, xi) / trapezoid(I, xi))     # para e^{−ξ²/w²}, la varianza es w²/2

fig, ax = plt.subplots(figsize=(9, 3.6))
for t, c in zip((0, 300, 600), RAMPA[1:]):
    xi, F = paquete_plasma(t)
    ax.plot(xi, abs(F)**2 / abs(paquete_plasma(0)[1]).max()**2, color=c, label=f"t = {t}")
ax.set(xlabel="ξ = x − v_g t", ylabel="|E|² (relativo)", title="un paquete en un plasma se ensancha (ℓ = 8, ω₀ = 2ωp)")
ax.legend(); plt.show()

# Comprobaciones
x = np.linspace(-np.pi, np.pi, 20001)
t = 0.5
verificar("ω = k + 10: la envolvente avanza con v_g = dω/dk = 1 (exacto: |F(x, t)| = |F(x − t, 0)|)", x[np.argmax(abs(tren(x, t, DISPERSION["ω = k + 10"])))] / t, 1.0, tol=1e-3)
vg = 1 / (2 * np.sqrt(15)); t = 0.5 / vg
verificar("ω = √k: la envolvente avanza aproximadamente con v_g = 1/(2√15) (el espectro tiene un ancho)", x[np.argmax(abs(tren(x, t, DISPERSION["ω = √k"])))] / t, vg, tol=3e-2)
for t in (300, 600):
    xi, F = paquete_plasma(t)
    verificar(f"plasma, t = {t}: ancho del paquete = √(ℓ² + (ω''t/ℓ)²)", ancho_paquete(xi, F), np.sqrt(64 + (t / 8 / 8)**2), tol=1e-2)

# %% [markdown]
# ### ¿Qué pasó?
# Sin dispersión ($\omega=k$), todo avanza junto y el tren no cambia de forma. Con $\omega=k+10$, la envolvente avanza con $v_g=1$ y las crestas, más rápido, con $v_\varphi=25/15$: aparecen por atrás del paquete y desaparecen por adelante. Con $\omega=\sqrt k$ pasa al revés ($v_g=v_\varphi/2$, como en las olas de agua profunda): las crestas nacen adelante y mueren atrás, y como $\omega''\neq0$ el paquete se deforma. Lo que viaja con el paquete, la energía y la información, lo hace con $v_g$.
#
# El paquete en el plasma se ensancha como dice la cuenta de segundo orden de las notas: en $t=600$, de 8 a unos 12.

# %% [markdown]
# ## Explorá
#
# 1. **Guía 11, problema 3.** Deducí $r_p$ y $t_p$ y compará con `fresnel_numerico(n1, n2, theta, (0, 1))`. Cuidado con la convención: acá el $\mathbf E$ de cada onda en polarización p es un múltiplo de $\hat{\mathbf p}=\hat{\mathbf x}\times\mathbf k/(nk_0)$, que para la onda reflejada apunta distinto que para la incidente. Comprobá también que tu $R_p+T_p=1$.
# 2. **Guía 11, problema 4.** Una lámina de vidrio en el vacío es una sola capa: `reflexion_capas([n], [d], n0=1.0, ns=1.0)`. Graficá $T$ en función de $d$ y comparalo con tu fórmula. ¿Para qué espesores $T=1$?
# 3. **Guía 11, problema 6.** Con `fresnel_numerico(1.0, np.sqrt(eps + 0j), 0.0)` podés calcular $R$ en incidencia normal para $\varepsilon$ real positivo, real negativo, y para un buen conductor, $\varepsilon=1+4\pi i\sigma/\omega$. Compará con tus resultados, y fijate qué pasa con $R$ para un conductor cuando crece $\sigma/\omega$.
# 4. **Ondas en un medio absorbente (sección 4 de las notas).** Con $\sqrt{\varepsilon}=n'+in''$ complejo, la amplitud decae como $e^{-n''\omega x/c}$. Usá `yee1d` con un plasma con `gamma` > 0 (como en la Clase 26), medí cuánto decae la amplitud en una longitud de onda y compará con el $n''$ que da la $\varepsilon(\omega)$ de Drude.
