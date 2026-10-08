# %% [markdown]
# # Clase 26 — Medios magnéticos y modelos de medios: ε(ω), plasma y energía
#
# **Objetivos**
# - Ver al electrón ligado como un oscilador forzado y amortiguado, y obtener de él $\varepsilon(\omega)$: la resonancia, la dispersión anómala y la absorción.
# - Mandar una onda a un plasma y ver que por debajo de $\omega_p$ no entra, y que por encima se propaga con $k^2=(\omega^2-\omega_p^2)/c^2$.
# - Dibujar las líneas de $\mathbf B$ y de $\mathbf H$ de un imán de barra, calculado con sus corrientes de magnetización.
# - Comprobar con una órbita que, al prender $\mathbf B$, el momento magnético cambia siempre en contra del campo: el diamagnetismo.
#
# **Material relacionado:** notas de la Clase 26. Guía 11: problemas 1, 2 y 5.
#
# **Unidades.** Gaussianas, adimensionales: en los experimentos 1 y 4, carga del electrón $-e$ con $e=1$, masa $m_e=1$ y $\omega_0=1$; en el 2, $c=1$ y $\omega_p=1$; en el 3, $M=1$ y el radio del imán es 1.

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
from scipy.integrate import solve_ivp, trapezoid, cumulative_trapezoid
from scipy.optimize import brentq

# %% [markdown]
# ## Experimento 1 ★ — El oscilador de Lorentz
#
# Un electrón ligado (carga $-e$, masa $m_e$) en el campo de una onda, $E(t)=E_0\cos\omega t$, uniforme sobre el átomo porque el átomo es mucho más chico que la longitud de onda:
# $$m_e\ddot x=-m_e\omega_0^2x-m_e\gamma\dot x-eE_0\cos\omega t .$$
# Integramos esta ecuación desde el reposo, esperamos que el transitorio se apague (decae como $e^{-\gamma t/2}$, Clase 17) y medimos la amplitud compleja del dipolo $p=-ex$. Las notas (sección 4) dan
# $$p=\mathrm{Re}\left[\alpha(\omega)E_0e^{-i\omega t}\right],\qquad\alpha(\omega)=\frac{e^2/m_e}{\omega_0^2-\omega^2-i\gamma\omega},\qquad\varepsilon(\omega)=1+4\pi n\alpha=1+\frac{\omega_p^2}{\omega_0^2-\omega^2-i\gamma\omega},$$
# con $\omega_p^2=4\pi ne^2/m_e$. Para dibujar $\varepsilon$ tomamos $\omega_p^2=0.3\,\omega_0^2$.
#
# ### Predecí
# 1. Justo en la resonancia, $\omega=\omega_0$, ¿el electrón se mueve en fase con la fuerza, en contrafase o desfasado $90^\circ$? ¿Y muy por encima de $\omega_0$?
# 2. ¿Cuánto vale $\varepsilon'$ en $\omega=\omega_0$: es máximo, vale 1 o es infinito?

# %%
W0, WP2 = 1.0, 0.3                                   # ω0 y ωp² (en unidades de ω0 y ω0²)

def alfa(w, g, w0=W0):
    """Polarizabilidad del oscilador de Lorentz, con e = m_e = 1."""
    return 1.0 / (w0**2 - w**2 - 1j * g * w)

def epsilon_lorentz(w, g, wp2=WP2, w0=W0):
    """ε(ω) = 1 + ωp²/(ω0² − ω² − iγω)."""
    return 1 + wp2 * alfa(w, g, w0)

def oscilador(w, g, w0=W0, E0=1.0, tfin=None):
    """Integra x'' = −ω0² x − γ x' − E0 cos ωt (electrón con e = m_e = 1) desde el reposo."""
    if tfin is None:
        tfin = 2 * np.log(1e7) / g + 4 * 2 * np.pi / w       # el transitorio cae a 1e-7
    f = lambda t, y: [y[1], -w0**2 * y[0] - g * y[1] - E0 * np.cos(w * t)]
    return solve_ivp(f, (0, tfin), [0.0, 0.0], method="DOP853", rtol=1e-10, atol=1e-12, dense_output=True)

def medir(w, g, w0=W0, nper=4):
    """En los últimos nper períodos: amplitud compleja de p = −x, potencia media <F v> y energía media del oscilador."""
    sol = oscilador(w, g, w0)
    T = 2 * np.pi / w
    t = np.linspace(sol.t[-1] - nper * T, sol.t[-1], 400 * nper + 1)
    x, v = sol.sol(t)
    p_hat = 2 / (nper * T) * trapezoid(-x * np.exp(1j * w * t), t)
    potencia = trapezoid(-np.cos(w * t) * v, t) / (nper * T)
    energia = trapezoid(0.5 * v**2 + 0.5 * w0**2 * x**2, t) / (nper * T)
    return p_hat, potencia, energia

def banda_anomala(ax, ws, eps_r):
    """Sombrea donde ε' baja al aumentar ω (dispersión anómala)."""
    baja = np.gradient(eps_r, ws) < 0
    ax.fill_between(ws, 0, 1, where=baja, transform=ax.get_xaxis_transform(), color=COLORES[3], alpha=0.15, lw=0)

def mostrar_oscilador(w=1.0, g=0.1):
    sol = oscilador(w, g, tfin=max(8 / g, 6 * 2 * np.pi / w))
    t = np.linspace(0, sol.t[-1], 4000)
    x = sol.sol(t)[0]
    p_hat = medir(w, g)[0]
    ws = np.linspace(0.02, 2.5, 2000); eps = epsilon_lorentz(ws, g)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.3))
    esc = np.max(abs(x[len(x) // 2:]))
    a1.plot(t, -np.cos(w * t) * esc, color=COLORES[1], lw=1, label="fuerza −eE (escalada)")
    a1.plot(t, x, color=COLORES[0], lw=1.3, label="x(t)")
    a1.set(xlabel="t ω₀", ylabel="x", title=f"ω/ω₀ = {w:g}: x atrasa {np.degrees(np.angle(p_hat)):.0f}° respecto de la fuerza")
    a1.legend(loc="upper left", fontsize=9, framealpha=0.9)
    banda_anomala(a2, ws, eps.real)
    a2.plot(ws, eps.real, color=COLORES[0], label="ε'")
    a2.plot(ws, eps.imag, color=COLORES[1], label="ε''")
    e_med = 1 + WP2 * p_hat
    a2.plot([w], [e_med.real], "o", color=COLORES[0]); a2.plot([w], [e_med.imag], "o", color=COLORES[1])
    a2.axhline(1, color="gray", lw=0.8)
    a2.set(xlabel="ω / ω₀", ylim=(-0.6 * WP2 / g - 0.5, 1.1 * WP2 / g + 1.5), title="ε(ω); puntos: medido; sombreado: dispersión anómala")
    a2.legend(loc="upper right")
    plt.show()

interactuar(mostrar_oscilador, w=deslizador("ω/ω₀", 1.0, 0.1, 2.5, 0.05), g=deslizador("γ/ω₀", 0.1, 0.02, 0.5, 0.02))

# %%
def figura_lorentz(g=0.1, exportar=False):
    ws = np.linspace(0.01, 2.5, 3000); eps = epsilon_lorentz(ws, g)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.3))
    banda_anomala(a1, ws, eps.real)
    a1.plot(ws, eps.real, color=COLORES[0], label="ε'")
    a1.plot(ws, eps.imag, color=COLORES[1], label="ε''")
    a1.axhline(1, color="gray", lw=0.8); a1.axvline(W0, color="gray", lw=0.8, ls=":")
    a1.set(xlabel="ω / ω₀", title=f"ε(ω) con ωp² = {WP2:g} ω₀² y γ = {g:g} ω₀")
    a1.legend()
    wl = np.logspace(-2, 2, 2000)
    a2.loglog(wl, wl**4 * abs(alfa(wl, g))**2, color=COLORES[2], label="oscilador de Lorentz")
    a2.loglog(wl[wl < 0.6], wl[wl < 0.6]**4, "--", color=COLORES[0], label="Rayleigh: (ω/ω₀)⁴")
    a2.loglog(wl[wl > 2], np.ones((wl > 2).sum()), "--", color=COLORES[1], label="Thomson: 1")
    a2.set(xlabel="ω / ω₀", ylabel="σ / σ_T", ylim=(1e-8, 3e3), title="sección eficaz de un electrón ligado")
    a2.legend(loc="lower right", fontsize=9)
    if exportar:
        guardar(fig, "nb26_lorentz")
    plt.show()

figura_lorentz(exportar=bool(CARPETA_FIGURAS))

# Comprobaciones
g = 0.1
errores = [abs(medir(w, g)[0] / alfa(w, g) - 1) for w in (0.5, 1.0, 1.6)]
verificar("la amplitud compleja medida de p es α(ω)E₀ (ω/ω₀ = 0.5, 1 y 1.6; error máximo)", max(errores), 0.0, tol=1e-6)
p0, pot0, _ = medir(W0, g)
verificar("en ω = ω₀, ε' medido = 1: toda la respuesta está en ε''", (1 + WP2 * p0).real, 1.0, tol=1e-6)
verificar("en ω = ω₀, la potencia media <F·v> = ½ ω Im α E₀²", pot0, 0.5 * W0 * alfa(W0, g).imag, tol=1e-6)
Pabs = lambda w: w * alfa(w, g).imag
wm = brentq(lambda w: Pabs(w) - Pabs(W0) / 2, 0.5, W0); wM = brentq(lambda w: Pabs(w) - Pabs(W0) / 2, W0, 1.5)
verificar("el ancho de la absorción, ω₊ − ω₋, es exactamente γ (como en la Clase 17)", wM - wm, g, tol=1e-9)
ws = np.linspace(0.8, 1.2, 400001); ep = epsilon_lorentz(ws, g).real
verificar("ε' es máximo en ω₀ − γ/2 (aproximado, para γ ≪ ω₀)", ws[np.argmax(ep)], W0 - g / 2, tol=3e-3)
verificar("ε' es mínimo en ω₀ + γ/2 (aproximado)", ws[np.argmin(ep)], W0 + g / 2, tol=3e-3)
verificar("ω ≪ ω₀: σ/σ_T = (ω/ω₀)⁴ (Rayleigh), en ω = 0.01 ω₀", 0.01**4 * abs(alfa(0.01, g))**2 / 0.01**4, 1.0, tol=1e-3)
verificar("ω ≫ ω₀: σ/σ_T = 1 (Thomson), en ω = 100 ω₀", 100.0**4 * abs(alfa(100.0, g))**2, 1.0, tol=1e-3)
for w in (0.5, 2.0):
    energia = medir(w, 0.02)[2]
    verificar(f"ω = {w} ω₀, γ chico: energía media del oscilador = ¼(e²/m_e)(ω² + ω₀²)/(ω₀² − ω²)² E₀² (sección 6)",
              energia, 0.25 * (w**2 + W0**2) / (W0**2 - w**2)**2, tol=1e-3)

# %% [markdown]
# ### ¿Qué pasó?
# Lejos de la resonancia, el electrón sigue a la fuerza: en fase por debajo de $\omega_0$ (como un resorte que se estira con la fuerza) y en contrafase por encima (domina la inercia). Justo en $\omega_0$ está desfasado $90^\circ$: la velocidad está en fase con la fuerza y la potencia que absorbe es máxima. Por eso **en $\omega_0$, $\varepsilon'=1$**: la parte en fase del dipolo se anula y toda la respuesta está en $\varepsilon''$, que mide la absorción.
#
# Alrededor de la resonancia, en una banda de ancho $\gamma$, $\varepsilon'$ **baja** al aumentar $\omega$: es la dispersión anómala (sombreada), y coincide con la zona de absorción. Fuera de ella, $\varepsilon'$ crece con $\omega$ (dispersión normal): un vidrio, cuyas resonancias están en el ultravioleta, desvía más el azul que el rojo. Muy por encima de $\omega_0$, $\varepsilon<1$: el electrón responde como si estuviera libre (Experimento 2).
#
# La sección eficaz une los dos casos de la Clase 23: $\sigma=\sigma_T\,\omega^4/[(\omega_0^2-\omega^2)^2+\gamma^2\omega^2]$ es la de Rayleigh, $\propto\omega^4$, por debajo de la resonancia, la de Thomson por encima, y en la resonancia es $(\omega_0/\gamma)^2$ veces mayor que la de Thomson. La energía media del oscilador, sumada a la del campo, da la fórmula de la sección 6 con $d(\omega\varepsilon)/d\omega$.

# %% [markdown]
# ## Experimento 2 — Una onda que llega a un plasma
#
# Para un electrón libre, $\omega_0=0$, y el modelo de Lorentz da el de Drude con campos que oscilan (sección 5 de las notas): $\varepsilon(\omega)=1-\omega_p^2/(\omega^2+i\gamma\omega)$. Sin choques, $\varepsilon=1-\omega_p^2/\omega^2$, y una onda plana en el medio cumple $k^2=\varepsilon\omega^2/c^2=(\omega^2-\omega_p^2)/c^2$.
#
# Lo comprobamos sin usar $\varepsilon$: resolvemos las ecuaciones de Maxwell en 1D con el esquema de Yee de la Clase 18, más la corriente de los electrones, que cumple $\partial_tJ_e=-\gamma J_e+(\omega_p^2/4\pi)E$ (la ecuación de movimiento de Drude multiplicada por $-ne$). Es la versión 2 de la herramienta `fdtd1d`: la de la Clase 18 más un medio. Con plasma, el esquema es estable si $S^2+(\omega_p\,\Delta t/2)^2\le1$; usamos $S=0.9$. Una fuente en $x=10$ emite una onda de frecuencia $\omega$ hacia un plasma que ocupa $x>30$; al final, una capa con $\gamma$ creciente absorbe lo que entra, para simular un plasma muy largo. Medimos la amplitud compleja $\hat E(x)$ ajustando $E(x,t)=\mathrm{Re}[\hat E(x)e^{-i\omega t}]$ en los últimos cuatro períodos.
#
# ### Predecí
# Con $\omega=\omega_p/2$, ¿qué hace la onda dentro del plasma? ¿Se propaga con otra longitud de onda, decae, o se absorbe? ¿Y con $\omega=1.5\,\omega_p$: la longitud de onda adentro es mayor o menor que en el vacío?

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

def onda_en_plasma(w, wp=1.0, gamma=0.0, x0=30.0, L=100.0, tfin=200.0, S=0.9):
    """Una fuente en x = 10 emite una onda de frecuencia w hacia un plasma en x > x0 (c = 1).
    Devuelve x, el campo E(x) al final y la amplitud compleja Ê(x)."""
    dx = 2 * np.pi / (40 * max(w, wp)); dt = S * dx
    x = np.arange(0, L, dx); N = len(x)
    wp2 = np.where(x > x0, wp**2, 0.0)
    g = gamma + np.where(x > 0.8 * L, 2.0 * ((x - 0.8 * L) / (0.2 * L))**2, 0.0)   # capa absorbente al final
    i_f = int(round(10 / dx)); T = 2 * np.pi / w; Tr = 6 * T
    def J(n):                                   # la fuente se prende suavemente en seis períodos
        t = (n + 0.5) * dt
        j = np.zeros(N); j[i_f] = np.sin(w * t) * np.sin(0.5 * np.pi * min(t / Tr, 1))**2 / dx
        return j
    cada = max(1, int(T / dt / 20))
    Es, Bs, ts = yee1d(np.zeros(N), np.zeros(N - 1), int(tfin / dt), S=S, dx=dx, J=J, cada=cada, wp2=wp2, gamma=g)
    nT = int(round(4 * T / (dt * cada)))
    base = np.stack([np.cos(w * ts[-nT:]), np.sin(w * ts[-nT:])], axis=1)     # E = a cos ωt + b sin ωt
    a, b = np.linalg.lstsq(base, Es[-nT:], rcond=None)[0]
    return x, Es[-1], a + 1j * b

def dibujar_plasma(ax, w, gamma=0.0):
    x, Ef, A = onda_en_plasma(w, gamma=gamma)
    ax.axvspan(30, 100, color=COLORES[2], alpha=0.10, lw=0); ax.axvspan(80, 100, color="gray", alpha=0.12, lw=0)
    ax.plot(x, Ef, color=COLORES[0], lw=1, label="E(x) al final")
    ax.plot(x, abs(A), color=COLORES[1], lw=1.8, label="amplitud |Ê(x)|")
    if w < 1:
        kap = np.sqrt(1 - w**2); xs = x[(x > 30) & (x < 30 + 4 / kap)]
        ax.plot(xs, abs(A[np.searchsorted(x, 30)]) * np.exp(-kap * (xs - 30)), "--", color="k", lw=1, label="∝ exp[−κ(x − 30)]")
        texto = f"ω = {w:g} ωp < ωp: decae en c/√(ωp² − ω²) = {1 / kap:.2f} c/ωp"
    else:
        texto = f"ω = {w:g} ωp: λ adentro = {2 * np.pi / np.sqrt(w**2 - 1):.2f}, en el vacío = {2 * np.pi / w:.2f} (c/ωp)"
    ax.set(xlim=(0, 100), xlabel="x  (unidades de c/ωp)", ylabel="E", title=texto)
    ax.text(31, 0.9 * ax.get_ylim()[1], "plasma", fontsize=10); ax.text(81, 0.9 * ax.get_ylim()[1], "capa absorbente", fontsize=9)
    ax.legend(loc="lower left", fontsize=8)

def mostrar_plasma(w=0.5):
    fig, ax = plt.subplots(figsize=(12, 3.6))
    dibujar_plasma(ax, w)
    plt.show()

interactuar(mostrar_plasma, w=deslizador("ω/ωp", 0.5, 0.3, 2.5, 0.05))
if CARPETA_FIGURAS:
    fig, axs = plt.subplots(2, 1, figsize=(11, 6.4), sharex=True)
    dibujar_plasma(axs[0], 0.5); dibujar_plasma(axs[1], 1.5); axs[0].set_xlabel("")
    fig.tight_layout(); guardar(fig, "nb26_plasma"); plt.close(fig)

# Comprobaciones
dx = 0.05; xv = np.arange(0, 20, dx); f = lambda u: np.exp(-((u - 5) / 0.5)**2)
Es, _, _ = yee1d(f(xv), f(xv[1:]), 100, S=1.0, dx=dx, cada=100)       # B = E[1:]: pulso hacia +x
verificar("sin medio, la versión 2 traslada un pulso exactamente con S = 1 (como la de la Clase 18)", np.max(abs(Es[-1] - f(xv - 100 * dx))), 0.0, tol=1e-12)
x, _, A = onda_en_plasma(0.5)
kap = np.sqrt(1 - 0.5**2); sel = (x > 31) & (x < 31 + 3 / kap)
verificar("ω = ωp/2: la amplitud decae como e^{−κx}, con κ = √(ωp² − ω²)/c", -np.polyfit(x[sel], np.log(abs(A[sel])), 1)[0], kap, tol=5e-3)
x, _, A = onda_en_plasma(1.5)
sel = (x > 33) & (x < 75)
verificar("ω = 1.5 ωp: la fase avanza con k = √(ω² − ωp²)/c", np.polyfit(x[sel], np.unwrap(np.angle(A[sel])), 1)[0], np.sqrt(1.5**2 - 1), tol=5e-3)

# %% [markdown]
# ### ¿Qué pasó?
# Por debajo de $\omega_p$, la onda **no se propaga** en el plasma: $k^2<0$, $k=i\kappa$, y la amplitud decae como $e^{-\kappa x}$ en una distancia $c/\sqrt{\omega_p^2-\omega^2}$, sin oscilar en el espacio. Sin choques no hay absorción ($\varepsilon''=0$): la energía no puede quedarse en el plasma, y la onda vuelve. A la izquierda se ve una onda estacionaria: la incidente y la que vuelve. (Cómo se calculan las amplitudes reflejada y transmitida en una interfaz lo vemos en la Clase 27.)
#
# Por encima de $\omega_p$ la onda entra, con una longitud de onda **mayor** que en el vacío, $2\pi c/\sqrt{\omega^2-\omega_p^2}$: la velocidad de fase $\omega/k$ es mayor que $c$. Es lo que pasa con la ionosfera ($f_p\sim9$ MHz): la radio de AM rebota, y la FM y los satélites pasan. Y por eso la onda de 30 Hz de un púlsar (Clase 23) no llega a nosotros.

# %% [markdown]
# ## Experimento 3 — Las líneas de B y de H de un imán de barra
#
# Un cilindro de radio $R=1$ y largo $L$, con magnetización uniforme $\mathbf M=M\hat{\mathbf z}$. Adentro $\nabla\times\mathbf M=0$, así que la única corriente de magnetización es la de la superficie lateral, $\mathbf K=c\mathbf M\times\hat{\mathbf n}=cM\hat{\boldsymbol\varphi}$ (sección 1 de las notas): **el imán es un solenoide** con $nI=cM$. Calculamos $\mathbf B$ con la herramienta de Biot–Savart de la Clase 12, apilando espiras con $I/c=M\,dz$, y después $\mathbf H=\mathbf B-4\pi\mathbf M$.
#
# Por la simetría de revolución, las líneas de $\mathbf B$ son curvas de nivel del flujo por un disco de radio $s$, $\Psi_B(s,z)=2\pi\int_0^sB_z\,s'ds'$ (la función de flujo de la Clase 3). Para $\mathbf H$ restamos el flujo de $4\pi\mathbf M$, que solo hay dentro del imán: $\Psi_H=\Psi_B-4\pi^2M\min(s,R)^2$ en la franja $|z|<L/2$, y $\Psi_B-4\pi^2MR^2$ fuera de ella. $\Psi_H$ salta en las tapas, donde están las ``cargas magnéticas'' $\mathbf M\cdot\hat{\mathbf n}=\pm M$: ahí nacen y mueren las líneas de $\mathbf H$. Dibujamos cada región por separado. Elegimos los niveles para que, donde el campo es uniforme, las líneas queden equiespaciadas en el dibujo.
#
# ### Predecí
# Dentro del imán, ¿hacia dónde apunta $\mathbf B$? ¿Y $\mathbf H$? ¿Las líneas de $\mathbf H$ son cerradas?

# %%
# herramienta: biot_savart v1 (NB12)
def campo_poligonal(puntos, vertices, I_c=1.0, cerrada=True, bloque=400):
    """Campo B en `puntos` (M × 3) de una corriente que recorre la poligonal de `vertices` (N × 3).

    Usa la fórmula exacta de cada segmento recto: con a = P1 − r y b = P2 − r,
        B = (I/c) (a × b) (|a| + |b|) / (|a| |b| (|a| |b| + a·b)).
    I_c = I/c. Si cerrada=True, une el último vértice con el primero.
    """
    puntos = np.atleast_2d(np.asarray(puntos, float)); V = np.asarray(vertices, float)
    P1 = V; P2 = np.roll(V, -1, axis=0) if cerrada else V[1:]
    P1 = P1 if cerrada else V[:-1]
    B = np.zeros_like(puntos)
    for i in range(0, len(puntos), bloque):                     # de a bloques, para no llenar la memoria
        r = puntos[i:i + bloque, None, :]
        a = P1[None, :, :] - r; b = P2[None, :, :] - r
        na = np.linalg.norm(a, axis=2); nb = np.linalg.norm(b, axis=2)
        den = na * nb * (na * nb + np.sum(a * b, axis=2))
        with np.errstate(divide="ignore", invalid="ignore"):
            f = np.where(den > 1e-300, (na + nb) / den, 0.0)
        B[i:i + bloque] = I_c * np.sum(np.cross(a, b) * f[..., None], axis=1)
    return B

def espira(R=1.0, N=360, z=0.0):
    """Vértices de una espira circular de radio R en el plano z (polígono de N lados, corriente antihoraria vista desde +z)."""
    t = np.linspace(0, 2 * np.pi, N, endpoint=False)
    return np.stack([R * np.cos(t), R * np.sin(t), np.full(N, z)], axis=1)

def campo_espiras(puntos, lista_z, R=1.0, N=120, I_c=1.0):
    """Campo de varias espiras coaxiales (un solenoide como pila de espiras)."""
    return sum(campo_poligonal(puntos, espira(R, N, z), I_c) for z in lista_z)
# fin herramienta

RI, MI = 1.0, 1.0                                     # radio y magnetización del imán

def B_iman(P, L=4.0, Nz=None):
    """B de un imán de barra de largo L: un solenoide de Nz espiras con I/c = M dz cada una (K = cM)."""
    Nz = Nz or max(120, int(30 * L))
    dz = L / Nz; zs = -L / 2 + dz * (np.arange(Nz) + 0.5)
    return campo_espiras(P, zs, R=RI, N=48, I_c=MI * dz)

def adentro(P, L):
    P = np.atleast_2d(P)
    return (np.hypot(P[:, 0], P[:, 1]) < RI) & (abs(P[:, 2]) < L / 2)

def H_iman(P, L=4.0):
    """H = B − 4πM (M solo adentro del imán)."""
    P = np.atleast_2d(np.asarray(P, float))
    return B_iman(P, L) - 4 * np.pi * MI * adentro(P, L)[:, None] * np.array([0, 0, 1.0])

def flujos_iman(L=4.0, zmax=3.5):
    """Funciones de flujo de B y de H en el plano (s, z), con s > 0 (las de s < 0 son iguales)."""
    ds = RI / 16
    s = ds * (np.arange(48) + 0.5); z = np.arange(0, max(zmax, L / 2 + 1.5) + 1e-9, ds)   # la pared queda entre dos nodos
    S_, Z_ = np.meshgrid(s, z, indexing="ij")
    Bz = B_iman(np.stack([S_.ravel(), 0 * S_.ravel(), Z_.ravel()], 1), L)[:, 2].reshape(S_.shape)
    Bz = np.concatenate([Bz[:, :0:-1], Bz], axis=1); z = np.concatenate([-z[:0:-1], z])   # simetría z → −z
    S_, Z_ = np.meshgrid(s, z, indexing="ij")
    psiB = 2 * np.pi * (cumulative_trapezoid(Bz * S_, s, axis=0, initial=0) + Bz[0] * s[0]**2 / 2)
    franja = abs(Z_) < L / 2
    psiH = psiB - 4 * np.pi**2 * MI * np.where(franja, np.minimum(S_, RI)**2, RI**2)
    return S_, Z_, psiB, psiH, franja

def niveles(escala, n=11):
    k = np.arange(1, 8 * n)
    return np.sort(np.concatenate([-escala * (k / n)**2, [0.0], escala * (k / n)**2]))

def flechas(ax, L, campo):
    pts = np.array([[0, 0, 0], [0.55, 0, 0.6 * L / 2], [-0.55, 0, -0.6 * L / 2], [2.0, 0, 0], [-2.0, 0, 0], [0, 0, L / 2 + 1.0], [0, 0, -L / 2 - 1.0]])
    F = campo(pts, L); n = np.hypot(F[:, 0], F[:, 2])
    ax.quiver(pts[:, 0], pts[:, 2], F[:, 0] / n, F[:, 2] / n, color="k", scale=14, width=0.008, zorder=5)

def mostrar_iman(L=4.0, exportar=False):
    S_, Z_, psiB, psiH, franja = flujos_iman(L)
    zmax = Z_.max()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 6.2))
    lev = niveles(0.9 * psiB.max())                    # los mismos niveles para B y H: la densidad de líneas se compara
    for sg in (1, -1):
        a1.contour(sg * S_, Z_, psiB, levels=lev, colors=[COLORES[0]], linewidths=0.9, linestyles="solid")
        a2.contour(sg * S_, Z_, np.ma.array(psiH, mask=(S_ < RI) & ~franja), levels=lev, colors=[COLORES[1]], linewidths=0.9, linestyles="solid")
        a2.contour(sg * S_, Z_, np.ma.array(psiH, mask=franja), levels=lev, colors=[COLORES[1]], linewidths=0.9, linestyles="solid")
    flechas(a1, L, B_iman); flechas(a2, L, H_iman)
    for ax, tit in ((a1, "líneas de B: cerradas"), (a2, "líneas de H: nacen y mueren en las tapas")):
        ax.add_patch(plt.Rectangle((-RI, -L / 2), 2 * RI, L, color=COLORES[2], alpha=0.15, lw=0))
        ax.set(aspect="equal", xlim=(-3, 3), ylim=(-zmax, zmax), xlabel="x / R", ylabel="z / R", title=tit)
        ax.grid(False)
    fig.suptitle(f"imán de barra, L = {L:g} R, M según +z")
    if exportar:
        guardar(fig, "nb26_iman")
    plt.show()

interactuar(mostrar_iman, L=deslizador("L / R", 4.0, 0.5, 8.0, 0.5))
if CARPETA_FIGURAS:
    mostrar_iman(4.0, exportar=True)

# Comprobaciones
def H_cargas(P, L, nr=200, nf=256):
    """H de las cargas magnéticas ±M de las tapas, sumando Coulomb sobre los dos discos."""
    r = (np.arange(nr) + 0.5) / nr * RI; f = (np.arange(nf) + 0.5) / nf * 2 * np.pi
    rr, ff = np.meshgrid(r, f, indexing="ij"); da = (RI / nr) * (2 * np.pi / nf) * rr
    H = np.zeros((len(P), 3))
    for zc, sig in ((L / 2, MI), (-L / 2, -MI)):
        fuente = np.stack([rr * np.cos(ff), rr * np.sin(ff), np.full_like(rr, zc)], -1).reshape(-1, 3)
        d = P[:, None, :] - fuente[None]
        H += np.sum((sig * da).ravel()[None, :, None] * d / np.linalg.norm(d, axis=2)[..., None]**3, axis=1)
    return H
P = np.array([[0, 0, 0.0], [0.5, 0, 0.7], [1.6, 0, 1.2], [0.8, 0, 2.6], [0.3, 0, -1.5]])
verificar("H = B − 4πM coincide con el campo de las cargas ±M de las tapas (5 puntos, error relativo máximo)",
          np.max(np.linalg.norm(H_iman(P) - H_cargas(P, 4.0), axis=1) / np.linalg.norm(H_cargas(P, 4.0), axis=1)), 0.0, tol=3e-3)
n = 200; u = (np.arange(n) + 0.5) / n
lados = [((0.5, 1), (1.5, 1)), ((1.5, 1), (1.5, -1)), ((1.5, -1), (0.5, -1)), ((0.5, -1), (0.5, 1))]   # normal +φ
circ_B = circ_H = 0.0
for (a, b) in lados:
    Pc = np.stack([a[0] + (b[0] - a[0]) * u, 0 * u, a[1] + (b[1] - a[1]) * u], 1); t = np.array([b[0] - a[0], 0, b[1] - a[1]])
    circ_B += np.sum(B_iman(Pc) @ t) / n; circ_H += np.sum(H_iman(Pc) @ t) / n
verificar("∮B·dl por un rectángulo que corta la pared = (4π/c) × corriente de magnetización = 4πM ℓ (ℓ = 2)", circ_B, 4 * np.pi * MI * 2, tol=2e-3)
verificar("∮H·dl por el mismo rectángulo = 0: no hay corrientes libres", circ_H, 0.0, tol=1e-4)
r = 40.0; Pf = np.array([[r * np.sin(0.6), 0, r * np.cos(0.6)]]); m = MI * np.pi * RI**2 * 4.0
rh = Pf[0] / r; Bd = (3 * m * rh[2] * rh - np.array([0, 0, m])) / r**3
verificar("lejos, B es el de un dipolo m = M × volumen (r = 40R, error relativo)", np.linalg.norm(B_iman(Pf)[0] - Bd) / np.linalg.norm(Bd), 0.0, tol=5e-3)
B0 = B_iman(np.zeros((1, 3)), L=40.0, Nz=800)[0, 2]
verificar("barra larga (L = 40R): en el centro B = 4πM (Clase 13 con nI/c = M), y H = B − 4πM → 0", B0, 4 * np.pi * MI, tol=2e-3)

# %% [markdown]
# ### ¿Qué pasó?
# Las líneas de $\mathbf B$ son cerradas: salen por la tapa de arriba, vuelven por afuera y suben por dentro del imán, **en el sentido de $\mathbf M$**. Afuera, $\mathbf H=\mathbf B$. Adentro, $\mathbf H=\mathbf B-4\pi\mathbf M$ apunta **al revés**: las líneas de $\mathbf H$ nacen en la tapa de arriba ($\mathbf M\cdot\hat{\mathbf n}=+M$) y mueren en la de abajo, adentro y afuera, como las de $\mathbf E$ en la barra polarizada de la Clase 25 (pregunta 25-a). Las comprobaciones lo confirman: $\mathbf H$ es exactamente el campo de esas cargas magnéticas. Su rotor es nulo (no hay corrientes libres), y su divergencia no: $\nabla\cdot\mathbf H=-4\pi\nabla\cdot\mathbf M$. Si el imán es largo, adentro $\mathbf H\to0$ y $\mathbf B\to4\pi\mathbf M$: las tapas quedan lejos.

# %% [markdown]
# ## Experimento 4 — Diamagnetismo: una órbita en un campo que se prende
#
# Un electrón (carga $-e$, masa $m_e$, con $e=m_e=c=1$) gira en una órbita circular de radio $r=1$ alrededor de un centro de fuerza. Prendemos un campo $B(t)\hat{\mathbf z}$ que sube de 0 a $B_f$ en un tiempo $T$. Mientras sube, hay un campo eléctrico inducido, $\mathbf E=-\frac{1}{2c}\dot B\,\hat{\mathbf z}\times\mathbf r$ ($\mathbf E=-\frac1c\partial_t\mathbf A$ con $\mathbf A=\frac12\mathbf B\times\mathbf r$, Clases 13 y 18), y la fuerza es $-e(\mathbf E+\mathbf v\times\mathbf B/c)$. Integramos el movimiento y comparamos el momento magnético $m_z=\frac{-e}{2c}(xv_y-yv_x)$, promediado en el tiempo después de la rampa, con el de antes. Las notas (sección 3) dan
# $$\Delta m_z=-\frac{e^2r^2}{4m_ec^2}B_f,$$
# para cualquier sentido de giro.
#
# ### Predecí
# Si el electrón gira en un sentido, su momento inicial es $+\mathbf m$; si gira en el otro, $-\mathbf m$. ¿El campo cambia los dos momentos en el mismo sentido o en sentidos opuestos? ¿Importa si el campo se prende rápido o despacio?

# %%
def orbita(Bf=0.02, T=20.0, sentido=1, fuerza="armonica", tfin=None):
    """Electrón (q = −1, m = 1, c = 1) en una órbita circular de radio 1 con velocidad angular 1, mientras B sube de 0 a Bf en un tiempo T.
    fuerza: "armonica" (−r) o "coulomb" (−r̂/r²). Devuelve la solución y la función m_z(t)."""
    q = -1.0
    Bt = lambda t: Bf * (min(t / T, 1) - np.sin(2 * np.pi * min(t / T, 1)) / (2 * np.pi))
    dB = lambda t: Bf * (1 - np.cos(2 * np.pi * t / T)) / T if t < T else 0.0
    def f(t, y):
        x, yy, vx, vy = y
        b, db = Bt(t), dB(t)
        Ex, Ey = 0.5 * db * yy, -0.5 * db * x              # E = −(1/2c) dB/dt ẑ × r
        r3 = (x * x + yy * yy)**1.5 if fuerza == "coulomb" else 1.0
        return [vx, vy, -x / r3 + q * (Ex + vy * b), -yy / r3 + q * (Ey - vx * b)]
    tfin = tfin or T + 40 * np.pi
    sol = solve_ivp(f, (0, tfin), [1.0, 0.0, 0.0, float(sentido)], method="DOP853", rtol=1e-11, atol=1e-12, dense_output=True)
    def mz(t):
        x, yy, vx, vy = sol.sol(t)
        return q / 2 * (x * vy - yy * vx)
    return sol, mz

def delta_m(**kw):
    T = kw.get("T", 20.0); sol, mz = orbita(**kw)
    t = np.linspace(T, sol.t[-1], 4001)
    return trapezoid(mz(t), t) / (t[-1] - T) - mz(0.0)

def mostrar_orbitas(Bf=0.02, T=20.0):
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for s, c, ls, lw in ((1, COLORES[0], "solid", 3.0), (-1, COLORES[1], "dashed", 1.6)):
        sol, mz = orbita(Bf, T, s)
        t = np.linspace(0, sol.t[-1], 3000)
        ax.plot(t, (mz(t) - mz(0.0)) / (Bf / 4), color=c, lw=lw, ls=ls, label=f"giro {'antihorario' if s > 0 else 'horario'}: m_z(0) = {mz(0.0):+.1f}")
    ax.axhline(-1, color="k", ls=":", lw=1.2, label="−e²r²B/4m_ec²")
    ax.axvspan(0, T, color="gray", alpha=0.12, lw=0)
    ax.set(xlabel="t", ylabel="Δm_z / (B_f/4)", ylim=(-1.6, 0.4), title="cambio del momento magnético (sombreado: mientras B sube)")
    ax.legend(fontsize=9, loc="upper right")
    plt.show()

interactuar(mostrar_orbitas, Bf=deslizador("B_f", 0.02, 0.005, 0.1, 0.005), T=deslizador("T", 20.0, 2.0, 60.0, 2.0))

# Comprobaciones
r, Bf = 1.0, 0.02
esperado = -r**2 * Bf / 4                       # −e²r²B/4m_ec² con e = m_e = c = 1
for s in (1, -1):
    verificar(f"giro {'antihorario' if s > 0 else 'horario'}: Δm_z = −e²r²B/4m_ec²", delta_m(sentido=s), esperado, tol=1e-3)
verificar("fuerza de Coulomb y campo que se prende rápido (T = 2): el mismo Δm_z", delta_m(fuerza="coulomb", T=2.0), esperado, tol=1e-3)

# %% [markdown]
# ### ¿Qué pasó?
# Los dos electrones, que empiezan con momentos opuestos, cambian su momento **en el mismo sentido: en contra de $\mathbf B$**. El campo eléctrico inducido frena a uno y acelera al otro, y en los dos casos el cambio de momento es $-e^2r^2B/4m_ec^2$. No importa cuán rápido se prende el campo, porque el impulso que da $\mathbf E$ depende solo del cambio total de flujo (Faraday), ni cómo está ligado el electrón, mientras el radio no cambie (la fuerza magnética $ev B/c$ aporta justo la fuerza centrípeta extra que hace falta, sección 3). Un átomo con muchos electrones, que antes del campo tenía momento total cero, queda con un momento opuesto a $\mathbf B$: el material es **diamagnético**, y lo repele un imán.

# %% [markdown]
# ## Explorá
#
# 1. **Guía 11, problema 1.** En un problema con simetría cilíndrica, $\mathbf M=M_\varphi(s)\hat{\boldsymbol\varphi}$ da una corriente de magnetización $J_z=\frac{c}{s}\frac{d}{ds}(sM_\varphi)$ en el volumen y $K_z=-cM_\varphi(R)$ en la superficie $s=R$ (con $\hat{\mathbf n}=\hat{\mathbf s}$, $\hat{\boldsymbol\varphi}\times\hat{\mathbf s}=-\hat{\mathbf z}$). Con tu $M_\varphi(s)$, usá `corriente_magnetizacion` y comprobá que la corriente de magnetización total es cero. Después sumá la corriente libre y calculá $B_\varphi(s)$ con `ampere_cilindrico`: tiene que dar tu $\mathbf B$.
# 2. **Guía 11, problema 2.** El enunciado da la ecuación de $\rho_l(t)$. Integrala con `solve_ivp`, como el oscilador del Experimento 1, para $\gamma/\omega_p=0.1$ y $\gamma/\omega_p=10$, y compará con tus resultados en los dos límites.
# 3. **Guía 11, problema 5.** La función `epsilon_lorentz` con `w0=0` da el $\varepsilon$ de los electrones libres. Escribí tu $\varepsilon_{ef}=\varepsilon+4\pi i\sigma/\omega$ con la $\sigma(\omega)$ de Drude de la sección 5 de las notas, y compará las dos para varias frecuencias.
# 4. **La forma del imán.** Calculá $H_z$ en el centro del imán con `H_iman` para $L/R$ entre $0.05$ y $40$. ¿A qué tiende para un disco chato? ¿Y para una barra larga? Compará con la esfera uniformemente magnetizada de las notas, que tiene $\mathbf H=-\frac{4\pi}{3}\mathbf M$ adentro, y mirá las líneas con el deslizador del Experimento 3. Abajo está el cálculo para $L/R=4$.

# %%
C_LUZ = 1.0                                     # en esta celda, c = 1

def corriente_magnetizacion(s, M_phi):
    """J_z = (c/s) d(s M_φ)/ds en los puntos s (numérico) y K_z = −c M_φ en el último punto (la superficie)."""
    Jz = C_LUZ / s * np.gradient(s * M_phi, s)
    return Jz, -C_LUZ * M_phi[-1]

def ampere_cilindrico(s, Jz, K=0.0):
    """B_φ(s) = (4π/cs) × (corriente que atraviesa el disco de radio s), con J_z(s) y una corriente superficial K_z en el borde s[-1]."""
    I = cumulative_trapezoid(Jz * 2 * np.pi * s, s, initial=0)
    B = 4 * np.pi / (C_LUZ * s) * I
    B[-1] += 4 * np.pi / (C_LUZ * s[-1]) * K * 2 * np.pi * s[-1]
    return B

# Ejemplo (no es el problema): una magnetización inventada, M_φ = s², en un cilindro de radio 1.
s = np.linspace(1e-3, 1.0, 2001)
Jz, Kz = corriente_magnetizacion(s, s**2)
print(f"corriente de magnetización: volumen = {trapezoid(Jz * 2 * np.pi * s, s):.4f}, superficie = {Kz * 2 * np.pi * s[-1]:.4f}")

L = 4.0
print(f"L/R = {L:g}: H_z(centro) / 4πM = {H_iman(np.zeros((1, 3)), L)[0, 2] / (4 * np.pi * MI):+.3f}")
