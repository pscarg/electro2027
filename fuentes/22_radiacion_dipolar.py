# %% [markdown]
# # Clase 22 — Radiación: zonas, dipolo eléctrico, Larmor y el átomo clásico
#
# **Objetivos**
# - Ver cómo las líneas de campo de un dipolo que oscila se cierran y se desprenden: la radiación.
# - Medir las tres zonas del campo ($1/r^3$, $1/r^2$, $1/r$) y comprobar que la potencia media es la misma a través de cualquier esfera.
# - Calcular el diagrama de radiación dipolar y la potencia total para cualquier $\ddot{\mathbf p}$.
# - Simular el colapso del átomo clásico y compararlo con la fórmula de Larmor.
#
# **Material relacionado:** notas de la Clase 22. Guía 9: problemas 1 a 3.
#
# **Unidades.** Gaussianas, adimensionales, con $c=1$. Para el dipolo, $p_0=1$ y $k=\omega=1$ (la longitud de onda es $\lambda=2\pi$).

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
from scipy.integrate import trapezoid, solve_ivp
from matplotlib import animation

# %% [markdown]
# ## Experimento 1 ★ — Las líneas de campo del dipolo de Hertz
#
# Un dipolo $\mathbf p=p_0\hat{\mathbf z}\,e^{-i\omega t}$ en el origen. Las notas (sección 5) dan los campos exactos, con $k=\omega/c$:
# $$E_r=2p_0\cos\theta\left(\frac{1}{r^3}-\frac{ik}{r^2}\right)e^{ikr},\qquad E_\theta=p_0\sin\theta\left(\frac{1}{r^3}-\frac{ik}{r^2}-\frac{k^2}{r}\right)e^{ikr},\qquad B_\varphi=-k^2p_0\sin\theta\left(1+\frac{i}{kr}\right)\frac{e^{ikr}}{r},$$
# y la función de flujo de la Clase 3, $\Psi=-p_0\sin^2\theta\left(\frac1r-ik\right)e^{ikr}$, con $\mathbf E=\frac1s\hat{\boldsymbol\varphi}\times\nabla\Psi$. Las líneas de $\mathbf E$ en el instante $t$ son las curvas de nivel de $\operatorname{Re}\left[\Psi e^{-i\omega t}\right]$.
#
# ### Predecí
# Cerca del dipolo ($r\ll\lambda$), ¿cómo son las líneas? ¿Y lejos? ¿Qué pasa con las líneas cuando el dipolo pasa por cero?

# %%
def campos_hertz(r, th, k=1.0, p0=1.0):
    """Amplitudes complejas (sin e^{−iωt}) de E_r, E_θ y B_φ del dipolo p0 ẑ e^{−iωt}, con c = 1 y ω = k."""
    e = np.exp(1j * k * r)
    Er = 2 * p0 * np.cos(th) * (1 / r**3 - 1j * k / r**2) * e
    Et = p0 * np.sin(th) * (1 / r**3 - 1j * k / r**2 - k**2 / r) * e
    Bp = -k**2 * p0 * np.sin(th) * (1 + 1j / (k * r)) * e / r
    return Er, Et, Bp

def flujo_hertz(r, th, k=1.0, p0=1.0):
    """Función de flujo Ψ (amplitud compleja): E = (1/s) φ̂ × ∇Ψ."""
    return -p0 * np.sin(th)**2 * (1 / r - 1j * k) * np.exp(1j * k * r)

L = 12.0
xg = np.linspace(-L, L, 321); X, Z = np.meshgrid(xg, xg)
Rg = np.hypot(X, Z); THg = np.arctan2(np.abs(X), Z)
Psi = flujo_hertz(np.maximum(Rg, 1e-3), THg)
mascara = Rg < 0.6
niveles = np.concatenate([-np.arange(0.15, 2.6, 0.3)[::-1], np.arange(0.15, 2.6, 0.3)])   # sin el nivel 0, que incluye el eje

def dibujar(ax, fase):
    ax.clear()
    lineas = np.ma.array(np.real(Psi * np.exp(-1j * fase)), mask=mascara)
    ax.contour(X, Z, lineas, levels=niveles, colors=[COLORES[0]], linewidths=1.0, linestyles="solid")
    ax.add_patch(plt.Circle((0, 0), 0.6, color="0.85"))
    for n in (1, 2):
        ax.add_patch(plt.Circle((0, 0), n * np.pi, fill=False, ls=":", color="0.6", lw=0.8))
    ax.set(aspect="equal", xlim=(-L, L), ylim=(-L, L), xlabel="x  (en unidades de 1/k)", ylabel="z",
           title=f"líneas de E, ωt = {np.degrees(fase) % 360:.0f}°  (círculos: r = λ/2, λ)")
    ax.grid(False)

fig, ax = plt.subplots(figsize=(6.2, 6.2))
fases = np.linspace(0, 2 * np.pi, 3 if PRUEBA else 32, endpoint=False)
anim = animation.FuncAnimation(fig, lambda n: dibujar(ax, fases[n]), frames=len(fases), interval=120)
plt.close(fig)
display(HTML(anim.to_jshtml()))

fig, ax = plt.subplots(figsize=(6.2, 6.2)); dibujar(ax, 0.3); guardar(fig, "nb22_hertz"); plt.close(fig)

# Comprobaciones: E sale de Ψ, Faraday se cumple, y el límite estático es el dipolo de la Clase 6
r0, t0, h = 1.3, 0.7, 1e-5
Er, Et, Bp = campos_hertz(r0, t0)
dPsi_dth = (flujo_hertz(r0, t0 + h) - flujo_hertz(r0, t0 - h)) / (2 * h)
dPsi_dr = (flujo_hertz(r0 + h, t0) - flujo_hertz(r0 - h, t0)) / (2 * h)
verificar("E_r = −(1/r² sinθ) ∂Ψ/∂θ", abs(-dPsi_dth / (r0**2 * np.sin(t0)) - Er) / abs(Er), 0.0, tol=1e-8)
verificar("E_θ = (1/r sinθ) ∂Ψ/∂r", abs(dPsi_dr / (r0 * np.sin(t0)) - Et) / abs(Et), 0.0, tol=1e-8)
rEt = lambda r: r * campos_hertz(r, t0)[1]
rot_E = ((rEt(r0 + h) - rEt(r0 - h)) / (2 * h) - (campos_hertz(r0, t0 + h)[0] - campos_hertz(r0, t0 - h)[0]) / (2 * h)) / r0
verificar("Faraday: (∇×E)_φ = ik B_φ", abs(rot_E - 1j * Bp) / abs(Bp), 0.0, tol=1e-7)
Ers, Ets, _ = campos_hertz(r0, t0, k=1e-5)
verificar("k → 0: E_r = 2p cosθ/r³ (dipolo estático)", Ers.real, 2 * np.cos(t0) / r0**3, tol=1e-6)

# %% [markdown]
# ### ¿Qué pasó?
# Cerca del dipolo, las líneas son las del dipolo estático de la Clase 6 con el $p$ de cada instante: salen de la carga positiva y vuelven a la negativa, y se encogen y se invierten con $p(t)$. Cuando $p$ pasa por cero, las líneas que estaban afuera no pueden volver: se cierran sobre sí mismas y quedan lazos que se alejan a velocidad $c$, separados media longitud de onda. Lejos, el campo es transversal y forma las ``costillas'' de una onda esférica, más intensas en el plano ecuatorial ($\sin\theta$) y nulas sobre el eje. La transición ocurre a $r\sim\lambda/2\pi$ (el primer círculo punteado es $\lambda/2$).
#
# ## Experimento 2 — Las tres zonas y el flujo de energía
#
# En el plano ecuatorial, $E_\theta$ tiene tres términos, $\frac{1}{r^3}$, $\frac{k}{r^2}$ y $\frac{k^2}{r}$, que son iguales en $kr=1$. Las notas muestran que, a pesar de eso, el flujo medio de energía es radial y **el mismo a través de cualquier esfera**: $P=\frac{p_0^2\omega^4}{3c^3}$, que vale $\frac13$ en estas unidades.
#
# ### Predecí
# Muy cerca del dipolo los campos son enormes. ¿El flujo medio de energía a través de una esfera chica es mayor, menor o igual que a través de una esfera lejana?

# %%
kr = np.logspace(-1.5, 1.5, 400)
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
for termino, lab, col in ((1 / kr**3, "1/r³ (cuasiestático)", COLORES[0]), (1 / kr**2, "k/r² (inducción)", COLORES[1]), (1 / kr, "k²/r (radiación)", COLORES[2])):
    axs[0].loglog(kr, termino, color=col, label=lab)
axs[0].axvline(1, color="0.6", lw=0.8); axs[0].set(xlabel="kr", ylabel="|término| de E_θ (θ = 90°)", title="las tres zonas")
axs[0].legend(fontsize=9)
Er, Et, Bp = campos_hertz(kr, np.pi / 2)
desfasaje = np.degrees(np.angle(Et / Bp))
axs[1].semilogx(kr, desfasaje, color=COLORES[3], label="fase de E_θ − fase de B_φ")
axs[1].set(xlabel="kr", ylabel="grados", title="cerca: E y B en cuadratura; lejos: en fase"); axs[1].legend(fontsize=9)
plt.tight_layout(); plt.show()

def potencia_media(r, nth=4001):
    """∮ ⟨S_r⟩ da sobre una esfera de radio r (c = 1): ⟨S_r⟩ = Re(E_θ B_φ*)/8π."""
    th = np.linspace(0, np.pi, nth)
    _, Et_, Bp_ = campos_hertz(r, th)
    return trapezoid(np.real(Et_ * np.conj(Bp_)) / (8 * np.pi) * 2 * np.pi * r**2 * np.sin(th), th)

for r in (0.05, 1.0, 30.0):
    verificar(f"potencia media a través de la esfera kr = {r:g}: p0²ω⁴/3c³", potencia_media(r), 1 / 3, tol=1e-6)
th = np.linspace(0.1, 3.0, 7)
Er_, _, Bp_ = campos_hertz(0.3, th)
verificar("⟨S_θ⟩ = −Re(E_r B_φ*)/8π = 0", np.max(np.abs(np.real(Er_ * np.conj(Bp_)))), 0.0, tol=1e-12)
verificar("lejos (kr = 10⁴): |E_θ| = |B_φ|", abs(campos_hertz(1e4, 1.0)[1] / campos_hertz(1e4, 1.0)[2]), 1.0, tol=1e-3)

# %% [markdown]
# ### ¿Qué pasó?
# Los tres términos se cruzan en $kr=1$, es decir, a $r=\lambda/2\pi$. Cerca, $E$ y $B$ oscilan en cuadratura: la energía va y viene entre el dipolo y el campo cercano, y en promedio no se lleva nada (es la parte imaginaria de $E_\theta B_\varphi^*$, que cae como $1/r^5$). Lejos, $E$ y $B$ oscilan en fase y con el mismo módulo, como en una onda plana. Lo que se lleva energía es la misma cantidad en todas las esferas, $\frac{p_0^2\omega^4}{3c^3}$: la energía que cruza una esfera chica es la misma que llega al infinito.
#
# ## Experimento 3 — El diagrama de radiación dipolar
#
# Lejos, $\mathbf E_{\text{rad}}=\frac{1}{c^2r}\,\hat{\mathbf r}\times(\hat{\mathbf r}\times\ddot{\mathbf p})$ y $\frac{dP}{d\Omega}=\frac{|\hat{\mathbf r}\times\ddot{\mathbf p}|^2}{4\pi c^3}$, con $\ddot{\mathbf p}$ en el tiempo retardado (sección 4 de las notas). Las funciones `campo_rad` y `patron` sirven para cualquier $\ddot{\mathbf p}$.
#
# ### Predecí
# Un dipolo que oscila según $\hat{\mathbf z}$: ¿en qué dirección se radía más? ¿Y si el dipolo está inclinado?

# %%
def campo_rad(pdd, n):
    """c² r E_rad = n × (n × p̈) = (n·p̈) n − p̈, para direcciones n (..., 3)."""
    pdd = np.asarray(pdd, float); n = np.asarray(n, float)
    return (n @ pdd)[..., None] * n - pdd

def patron(pdd, nth=181, nph=361):
    """dP/dΩ = |n × p̈|²/4π (c = 1) sobre una grilla (θ, φ). Devuelve θ, φ, dP/dΩ."""
    th = np.linspace(0, np.pi, nth); ph = np.linspace(0, 2 * np.pi, nph)
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    n = np.stack([np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH), np.cos(TH)], axis=-1)
    E = campo_rad(pdd, n)
    return th, ph, np.sum(E**2, axis=-1) / (4 * np.pi)

def potencia_total(pdd):
    th, ph, dP = patron(pdd)
    return trapezoid(trapezoid(dP, ph, axis=1) * np.sin(th), th)

def mostrar_patron(inclinacion=0.0):
    a = np.radians(inclinacion)
    pdd = np.array([np.sin(a), 0, np.cos(a)])
    th, ph, dP = patron(pdd, 61, 91)
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    fig = plt.figure(figsize=(11, 4.6))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.plot_surface(dP * np.sin(TH) * np.cos(PH), dP * np.sin(TH) * np.sin(PH), dP * np.cos(TH),
                    color=COLORES[0], alpha=0.6, linewidth=0)
    ax.quiver(0, 0, 0, *(0.12 * pdd), color=COLORES[1], linewidth=2)
    m = 0.09; ax.set(xlim=(-m, m), ylim=(-m, m), zlim=(-m, m), title="dP/dΩ en 3D (flecha: p̈)")
    ax2 = fig.add_subplot(1, 2, 2, projection="polar")
    th_p = np.linspace(0, 2 * np.pi, 400)
    n = np.stack([np.sin(th_p), 0 * th_p, np.cos(th_p)], axis=-1)        # corte en el plano xz
    ax2.plot(th_p, np.sum(campo_rad(pdd, n)**2, axis=-1) / (4 * np.pi), color=COLORES[0])
    ax2.set_theta_zero_location("N"); ax2.set_title("corte en el plano xz")
    plt.show()

interactuar(mostrar_patron, inclinacion=deslizador("inclinación de p̈ (°)", 0.0, 0.0, 90.0, 5.0))

pdd = np.array([0.3, -0.5, 0.8])
verificar("P = ∮ dP/dΩ dΩ = 2|p̈|²/3c³ (p̈ en cualquier dirección)", potencia_total(pdd), 2 * (pdd @ pdd) / 3, tol=1e-4)
th, ph, dP = patron([0, 0, 1.0])
verificar("p̈ ∥ ẑ: el máximo está en θ = 90°", np.degrees(th[np.argmax(dP[:, 0])]), 90.0, tol=1e-6)
n = np.array([np.sin(0.4) * np.cos(1.1), np.sin(0.4) * np.sin(1.1), np.cos(0.4)])
verificar("E_rad es transversal: n·E_rad = 0", campo_rad(pdd, n) @ n, 0.0, tol=1e-14)
# Larmor: una carga que oscila, z = d cos ωt, tiene p̈ = −q d ω² cos ωt ẑ
q, d, w = 1.0, 0.1, 2.0
t = np.linspace(0, 2 * np.pi / w, 2001)
P_t = 2 * (q * d * w**2 * np.cos(w * t))**2 / 3
verificar("⟨P⟩ de una carga que oscila = q²d²ω⁴/3c³", trapezoid(P_t, t) / t[-1], q**2 * d**2 * w**4 / 3, tol=1e-6)

# %% [markdown]
# ### ¿Qué pasó?
# El diagrama es una ``rosquilla'' con el eje en la dirección de $\ddot{\mathbf p}$: nada a lo largo del eje y el máximo en el plano perpendicular, con $\sin^2\alpha$, donde $\alpha$ es el ángulo entre $\hat{\mathbf r}$ y $\ddot{\mathbf p}$. Si el dipolo se inclina, la rosquilla se inclina con él, y la potencia total, $\frac{2|\ddot{\mathbf p}|^2}{3c^3}$, no depende de la dirección. El campo radiado es siempre perpendicular a $\hat{\mathbf r}$: es la proyección de $-\ddot{\mathbf p}$ sobre el plano transversal.
#
# ## Experimento 4 — El átomo clásico se cae
#
# Un electrón en una órbita circular alrededor de un protón radía con la potencia de Larmor, $P=\frac{2e^2a^2}{3c^3}=m_e\tau_ea^2$ con $\tau_e=\frac{2e^2}{3m_ec^3}$, y la energía $U=-\frac{e^2}{2r}$ baja. Las notas (sección 7) dan, si la órbita se achica despacio, $r^3=r_0^3-6\tau_e\frac{e^2}{m_e}t$.
#
# Acá simulamos la órbita completa. A la fuerza de Coulomb le sumamos una fuerza de frenado $\mathbf F=m_e\tau_e\dot{\mathbf a}$: su trabajo en un período es $m_e\tau_e\int\dot{\mathbf a}\cdot\mathbf v\,dt=m_e\tau_e[\mathbf a\cdot\mathbf v]-m_e\tau_e\int a^2dt$, y el primer término se anula en un movimiento periódico, así que en promedio saca del movimiento justo la potencia de Larmor. Como $\tau_e$ es chico, en $\dot{\mathbf a}$ usamos la derivada de la aceleración de Coulomb a lo largo de la trayectoria. Unidades: $e^2/m_e=1$, $r_0=1$; el $\tau_e$ real ($\sim10^{-7}$ del período) se exagera para ver la espiral.
#
# ### Predecí
# Mientras el electrón pierde energía y la órbita se achica, ¿su rapidez aumenta o disminuye?

# %%
def orbita_con_frenado(tau_e, ley="coulomb", r0=1.0, w0=1.0, t_max=None, r_min=0.25):
    """Integra a = a_0 + τ_e da_0/dt para una fuerza central a_0 = −r/r³ (Coulomb, e²/m = 1) o −ω0² r (elástica).

    Empieza en una órbita circular de radio r0. Devuelve (t, x, y, vx, vy).
    """
    def f(t, Y):
        r, v = Y[:2], Y[2:]
        R = np.hypot(*r)
        if ley == "coulomb":
            a0 = -r / R**3; da0 = -v / R**3 + 3 * (r @ v) * r / R**5
        else:
            a0 = -w0**2 * r; da0 = -w0**2 * v
        return np.concatenate([v, a0 + tau_e * da0])
    v0 = 1 / np.sqrt(r0) if ley == "coulomb" else w0 * r0
    choque = lambda t, Y: np.hypot(*Y[:2]) - r_min
    choque.terminal = True
    if t_max is None:
        t_max = r0**3 / (6 * tau_e) if ley == "coulomb" else 5 / (tau_e * w0**2)
    sol = solve_ivp(f, (0, t_max), [r0, 0, 0, v0], rtol=1e-10, atol=1e-12, events=choque, max_step=0.05)
    return sol.t, *sol.y

tau_e = 2e-3
t, x, y, vx, vy = orbita_con_frenado(tau_e)
r = np.hypot(x, y); v = np.hypot(vx, vy)
fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
axs[0].plot(x, y, color=COLORES[0], lw=0.8); axs[0].set(aspect="equal", title=f"órbita con τ_e = {tau_e:g}", xlabel="x", ylabel="y")
axs[1].plot(t, r**3, color=COLORES[0], label="r³ simulado")
axs[1].plot(t, 1 - 6 * tau_e * t, "k--", lw=1, label="r₀³ − 6τ_e t (notas)")
axs[1].plot(t, v / 4, color=COLORES[1], lw=1, label="rapidez / 4")
axs[1].set(xlabel="t", title="la órbita se achica y el electrón se acelera"); axs[1].legend(fontsize=9)
plt.tight_layout(); plt.show()

i = np.argmin(np.abs(t - (1 - 0.5**3) / (6 * tau_e)))
verificar("r cuando las notas predicen r = 0.5", r[i], 0.5, tol=1e-2)
verificar("la rapidez sigue a v = (e²/m r)^{1/2} (virial)", v[i] * np.sqrt(r[i]), 1.0, tol=1e-2)
vueltas = np.sum(np.diff(np.unwrap(np.arctan2(y, x)))) / (2 * np.pi)
t_c = 1 / (6 * tau_e)
verificar("vueltas hasta r = 0.25: N = (2t_c/T₀)[1 − (r/r₀)^{3/2}] (notas)", vueltas, (2 * t_c / (2 * np.pi)) * (1 - 0.25**1.5), tol=2e-2)

# %% [markdown]
# ### ¿Qué pasó?
# La órbita se achica en espiral, y $r^3$ baja linealmente, como predicen las notas: cada vuelta pierde una fracción chica de la energía, y la órbita sigue siendo casi circular. **El electrón se acelera**: con $U=-\frac{e^2}{2r}$ y la energía cinética $\frac{e^2}{2r}=-U$, perder energía es caer más cerca y moverse más rápido. Con el $\tau_e$ real del hidrógeno, la caída tarda $1.6\times10^{-11}$~s: la física clásica no puede explicar átomos estables.
#
# ## Explorá
#
# 1. **Guía 9, P1.** Para $p(t)=p_0\cos\omega t$, compará tu $\mathbf B(\mathbf r,t)$ exacto con la parte real de `campos_hertz` multiplicada por $e^{-i\omega t}$, en varios puntos y tiempos.
# 2. **Guía 9, P2.** `campo_rad(pdd, n)` da el campo radiado en la dirección `n` para un $\ddot{\mathbf p}$ dado. Armá $\ddot{\mathbf p}(t)$ para el dipolo giratorio, calculá $\frac{dP}{d\Omega}$ promediado en un período con `patron`, y dibujá la curva que recorre $\mathbf E_{\text{rad}}$ en el plano perpendicular a `n` durante un período para distintos $\theta$.
# 3. **Las vueltas del átomo.** En el Experimento 4, cambiá $\tau_e$ y contá las vueltas hasta el colapso. Compará con el número que dan las notas para el hidrógeno, $2t_c/T_0$.
# 4. **Un oscilador que radía.** Con `orbita_con_frenado(tau_e, ley="elastica", w0=...)`, un electrón ligado elásticamente pierde energía por radiación. Mostrá que la energía decae como $e^{-\gamma t}$ con $\gamma=\tau_e\omega_0^2$, y estimá cuánto dura el ``destello'' de un átomo que emite luz visible.
