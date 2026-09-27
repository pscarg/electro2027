# %% [markdown]
# # Clase 3 — El potencial electrostático
#
# **Objetivos**
# - Ver que las equipotenciales son perpendiculares a las líneas de campo, y que las líneas se pueden dibujar como curvas de nivel de una **función de flujo**.
# - Comprobar que el trabajo de $\mathbf E$ entre dos puntos **no depende del camino**, y ver qué falla cuando el rotor se concentra en una línea (el vórtice de la Clase 1).
# - Calcular el potencial de un disco cargado en todo el espacio y reconocer sus límites: plano cerca, carga puntual lejos.
# - Entender por qué el potencial de un hilo infinito necesita una referencia a distancia finita.
#
# **Material relacionado:** notas de la Clase 3. Guía 2: problemas 1 y 3.

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

from scipy.integrate import solve_ivp, trapezoid
from scipy.special import ellipk

# %% [markdown]
# ## Experimento 1 ★ — Líneas de campo y equipotenciales
#
# Ponemos dos cargas sobre el eje $z$: $q_1$ en $z=-d/2$ y $q_2$ en $z=+d/2$. En el plano $xz$ dibujamos:
# - las **equipotenciales** $\phi = \sum_i q_i/|\mathbf r - \mathbf r_i| = $ cte., en naranja;
# - las **líneas de campo**, en gris, como curvas de nivel de la función de flujo $\Psi = \sum_i q_i\cos\theta_i$ (deducida en las notas).
#
# ### Predecí
# 1. Para un dipolo ($q_1=+1$, $q_2=-1$), dibujá a mano algunas equipotenciales. ¿Cómo se ven de lejos?
# 2. Para dos cargas iguales ($q_2=+1$): hay un punto donde $\mathbf E = 0$. ¿Qué forma tiene la equipotencial que pasa por él?

# %%
def potencial_y_flujo(X, Z, qs, zs):
    """φ y Ψ en el plano meridiano (X hace de coordenada radial, con signo) para cargas sobre el eje z."""
    phi = sum(q / np.sqrt(X**2 + (Z - zi)**2) for q, zi in zip(qs, zs))
    psi = sum(q * (Z - zi) / np.sqrt(X**2 + (Z - zi)**2) for q, zi in zip(qs, zs))
    return phi, psi

def campo_meridiano(x, z, qs, zs):
    """Componentes (E_x, E_z) del campo en el plano xz."""
    Ex = sum(q * x / (x**2 + (z - zi)**2)**1.5 for q, zi in zip(qs, zs))
    Ez = sum(q * (z - zi) / (x**2 + (z - zi)**2)**1.5 for q, zi in zip(qs, zs))
    return Ex, Ez

def mapa(q2=-1.0, d=1.0, q1=1.0):
    qs, zs = np.array([q1, q2]), np.array([-d / 2, d / 2])
    x = np.linspace(-3, 3, 600); z = np.linspace(-3, 3, 600)
    X, Z = np.meshgrid(x, z)
    phi, psi = potencial_y_flujo(X, Z, qs, zs)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.contour(X, Z, phi, levels=np.linspace(-3, 3, 31), colors=COLORES[1], linewidths=1.0, linestyles="solid")
    nq = np.abs(qs).sum()
    ax.contour(X, Z, psi, levels=np.linspace(-nq, nq, 33), colors="0.3", linewidths=0.8, linestyles="solid")
    for q, zi in zip(qs, zs):
        ax.plot(0, zi, "o", ms=9 + 3 * abs(q), color=COLORES[7] if q > 0 else COLORES[0], zorder=3)
    ax.set(aspect="equal", xlabel="x", ylabel="z", title=f"q1 = {q1:+.1f}, q2 = {q2:+.1f}   (naranja: φ = cte.; gris: líneas de campo)")
    ax.grid(False); plt.show()

interactuar(mapa, q2=deslizador("q2", -1.0, -2, 2, 0.25), d=deslizador("d", 1.0, 0.4, 2, 0.1))

# %% [markdown]
# **Una verificación independiente.** Integramos numéricamente una línea de campo, resolviendo $d\mathbf r/d\ell = \mathbf E/|\mathbf E|$ desde muy cerca de $q_1$, y comprobamos que $\Psi$ se mantiene constante a lo largo de ella. Esto pone a prueba la deducción de la función de flujo sin usarla para dibujar.

# %%
qs, zs = np.array([2.0, -1.0]), np.array([-0.5, 0.5])

def direccion(l, r):
    Ex, Ez = campo_meridiano(r[0], r[1], qs, zs)
    norma = np.hypot(Ex, Ez)
    return [Ex / norma, Ez / norma]

def cerca_de_q2(l, r):
    return np.hypot(r[0], r[1] - zs[1]) - 0.02
cerca_de_q2.terminal = True

alfa = 0.9                                             # ángulo de salida respecto del eje +z
r0 = [0.01 * np.sin(alfa), zs[0] + 0.01 * np.cos(alfa)]
sol = solve_ivp(direccion, [0, 8], r0, events=cerca_de_q2, rtol=1e-10, atol=1e-12, max_step=0.01)
_, psi_linea = potencial_y_flujo(sol.y[0], sol.y[1], qs, zs)
print(f"Ψ a lo largo de la línea: entre {psi_linea.min():.8f} y {psi_linea.max():.8f}")
verificar("variación de Ψ a lo largo de una línea de campo integrada", psi_linea.max() - psi_linea.min(), 0.0, tol=1e-6)

# %% [markdown]
# ### ¿Qué pasó?
# - Las curvas naranjas y grises se cortan **siempre en ángulo recto**: $\mathbf E = -\nabla\phi$ es normal a las equipotenciales.
# - **Cerca** de cada carga, las equipotenciales son casi esferas. **Lejos** de un dipolo, tienen la forma de la función $\cos\theta/r^2$ (Clase 6).
# - Con dos cargas iguales hay un punto de campo nulo. La equipotencial que pasa por él **se corta a sí misma**, formando un cono. En un punto donde $\mathbf E=0$, el gradiente no define una única normal.
# - La densidad de equipotenciales, con saltos de $\phi$ iguales entre curvas vecinas, indica dónde el campo es intenso: $|\mathbf E| \approx \Delta\phi/\text{separación}$.
#
# ## Experimento 2 — El trabajo no depende del camino (salvo que...)
#
# Tomamos 5 cargas al azar y dos puntos $A$ y $B$. Calculamos $\int_A^B \mathbf E\cdot d\mathbf l$ por un camino recto y por uno ondulado, y lo comparamos con $\phi(A)-\phi(B)$.
#
# ### Predecí
# 1. ¿Dan lo mismo los dos caminos?
# 2. En el vórtice de la Clase 1, $\mathbf F = (-y,x,0)/(x^2+y^2)$, el rotor es nulo fuera del eje. ¿Cuánto vale la circulación de $\mathbf F$ en una circunferencia que rodea al eje? ¿Y en una que no lo rodea?

# %%
rng = np.random.default_rng(7)
q_az = np.array([1.2, -0.8, 0.7, -1.3, 1.0])            # cargas de ambos signos
r_az = rng.uniform(-1, 1, (5, 3))                        # posiciones al azar

def campo_3d(P):
    """Campo de las cargas al azar en los puntos P (N x 3)."""
    E = np.zeros_like(P)
    for q, ri in zip(q_az, r_az):
        R = P - ri
        E += q * R / np.linalg.norm(R, axis=1, keepdims=True)**3
    return E

def pot_3d(P):
    return sum(q / np.linalg.norm(P - ri, axis=-1) for q, ri in zip(q_az, r_az))

def trabajo(camino, t):
    """∫ E·dl a lo largo de camino(t), t en [0, 1]."""
    P = camino(t)
    dP = np.gradient(P, t, axis=0)
    return trapezoid(np.sum(campo_3d(P) * dP, axis=1), t)

A, B = np.array([-2.0, -1.5, 0.5]), np.array([2.0, 1.5, -0.3])
recto = lambda t: A + np.outer(t, B - A)
ondulado = lambda t: A + np.outer(t, B - A) + np.outer(np.sin(np.pi * t), [0, 0, 2.5]) + np.outer(np.sin(3 * np.pi * t), [0.8, -0.8, 0])
t = np.linspace(0, 1, 20001)

W1, W2 = trabajo(recto, t), trabajo(ondulado, t)
print(f"camino recto:    ∫E·dl = {W1:.6f}")
print(f"camino ondulado: ∫E·dl = {W2:.6f}")
verificar("∫E·dl por el camino recto = φ(A) − φ(B)", W1, pot_3d(A) - pot_3d(B), tol=1e-5)
verificar("∫E·dl por el camino ondulado = φ(A) − φ(B)", W2, pot_3d(A) - pot_3d(B), tol=1e-5)

fig, ax = plt.subplots(figsize=(5.5, 4.5))
for camino, nombre, c in [(recto, "recto", COLORES[0]), (ondulado, "ondulado", COLORES[1])]:
    P = camino(t); ax.plot(P[:, 0], P[:, 1], color=c, label=nombre)
ax.scatter(r_az[:, 0], r_az[:, 1], c=[COLORES[7] if q > 0 else COLORES[0] for q in q_az], s=60, zorder=3)
ax.plot(*A[:2], "ks"); ax.plot(*B[:2], "k^")
ax.set(xlabel="x", ylabel="y", title="proyección xy de los caminos y las cargas", aspect="equal"); ax.legend()
plt.show()

# el vórtice de la Clase 1: rotor nulo fuera del eje, pero circulación 2π alrededor de él
s = np.linspace(0, 2 * np.pi, 20001)
def circulacion_vortice(cx, cy, radio):
    x, y = cx + radio * np.cos(s), cy + radio * np.sin(s)
    dx, dy = -radio * np.sin(s), radio * np.cos(s)
    return trapezoid((-y * dx + x * dy) / (x**2 + y**2), s)
verificar("circulación del vórtice en una circunferencia que rodea al eje = 2π", circulacion_vortice(0.3, 0.2, 1.0), 2 * np.pi, tol=1e-8)
verificar("circulación del vórtice en una circunferencia que no rodea al eje = 0", circulacion_vortice(2.0, 0.0, 0.5), 0.0, tol=1e-8)

# %% [markdown]
# ### ¿Qué pasó?
# El trabajo coincide para los dos caminos y es igual a $\phi(A)-\phi(B)$: **$\nabla\times\mathbf E=0$ implica que $\mathbf E$ es un gradiente**.
#
# El vórtice tiene rotor nulo en todos los puntos donde está definido, y sin embargo su circulación alrededor del eje es $2\pi$. La región sin el eje **no es simplemente conexa**: un lazo que rodea el eje no puede achicarse hasta un punto sin cruzarlo. Por eso el teorema de Stokes no se puede aplicar a ese lazo y no existe un potencial univaluado. Una "función ángulo" $\varphi$ cumple $\nabla\varphi = \hat{\boldsymbol\varphi}/s$, pero aumenta en $2\pi$ en cada vuelta. En electrostática esto no pasa, porque $\nabla\times\mathbf E = 0$ vale en todo el espacio, incluso donde hay cargas.
#
# ## Experimento 3 — El disco cargado en todo el espacio
#
# El potencial fuera del eje de un disco de radio $R$ y densidad $\sigma$ se obtiene sumando **anillos**. El potencial de un anillo de carga $Q$ y radio $a$ es
# $$\phi_{\text{anillo}}(s,z) = \frac{2Q}{\pi}\,\frac{K(m)}{\sqrt{(a+s)^2+z^2}}, \qquad m = \frac{4as}{(a+s)^2+z^2},$$
# donde $K$ es la integral elíptica completa de primera especie (la deducción está en las notas). En el eje ($s=0$), $K(0)=\pi/2$ y se recupera $Q/\sqrt{a^2+z^2}$.
#
# ### Predecí
# Sobre el eje, ¿cómo decae $\phi(z)$ cerca del disco ($z\ll R$) y lejos ($z\gg R$)? ¿El potencial es continuo al cruzar el disco? ¿Y el campo?

# %%
def potencial_disco(s, z, R=1.0, sigma=1.0, n=400):
    """Potencial de un disco uniforme, sumando anillos con cuadratura de Gauss-Legendre."""
    x, w = np.polynomial.legendre.leggauss(n)
    a = 0.5 * R * (x + 1); pesos = 0.5 * R * w
    s = np.asarray(s, float)[..., None]; z = np.asarray(z, float)[..., None]
    Q = 2 * np.pi * sigma * a                       # carga por unidad de radio del anillo
    D2 = (a + s)**2 + z**2
    return np.sum(pesos * (2 * Q / np.pi) * ellipk(4 * a * s / D2) / np.sqrt(D2), axis=-1)

phi_eje = lambda z, R=1.0, sigma=1.0: 2 * np.pi * sigma * (np.sqrt(R**2 + z**2) - np.abs(z))

x = np.linspace(-2.5, 2.5, 161); z = np.linspace(-2.5, 2.5, 161)
X, Z = np.meshgrid(x, z)
PHI = potencial_disco(np.abs(X), Z + 1e-9)

fig, axs = plt.subplots(1, 2, figsize=(12, 5))
cs = axs[0].contourf(X, Z, PHI, levels=30, cmap="Blues")
axs[0].contour(X, Z, PHI, levels=15, colors="k", linewidths=0.5)
axs[0].plot([-1, 1], [0, 0], color=COLORES[7], lw=4)
axs[0].set(aspect="equal", xlabel="x", ylabel="z", title="φ del disco (σ = 1, R = 1)"); axs[0].grid(False)
fig.colorbar(cs, ax=axs[0], shrink=0.8)
zz = np.geomspace(1e-3, 1e3, 200)
axs[1].loglog(zz, phi_eje(zz), label="exacto en el eje")
axs[1].loglog(zz, np.pi / zz, "--", lw=1, label="carga puntual Q/z")
axs[1].loglog(zz, 2 * np.pi * (1 - zz), ":", lw=1.5, label="plano: 2πσ(R − z)")
axs[1].set(ylim=(1e-3, 10), xlabel="z / R", ylabel="φ(0, z)", title="límites cercano y lejano"); axs[1].legend()
plt.tight_layout(); guardar(fig, "nb03_disco"); plt.show()

verificar("suma de anillos en el eje (z = 0.5) = 2πσ(√(R²+z²) − z)", potencial_disco(0.0, 0.5), phi_eje(0.5), tol=1e-8)
r_lejos = np.array([30.0, 40.0])
verificar("lejos (r = 50): φ ≈ Q/r", potencial_disco(*r_lejos), np.pi / 50, tol=1e-3)
Ez = lambda z: 2 * np.pi * (np.sign(z) - z / np.sqrt(1 + z**2))   # E_z = −∂φ/∂z en el eje
verificar("salto de E_z al cruzar el disco = 4πσ", Ez(1e-9) - Ez(-1e-9), 4 * np.pi, tol=1e-6)

# %% [markdown]
# ### ¿Qué pasó?
# - **Cerca del disco** ($z\ll R$), $\phi \approx 2\pi\sigma(R-|z|)$ y el campo vale $2\pi\sigma$: el disco se ve como un plano infinito.
# - **Lejos** ($z\gg R$), $\phi \approx Q/z$: se ve como una carga puntual.
# - $\phi$ es **continuo** al cruzar el disco, pero tiene un **quiebre**: $E_z = -\partial_z\phi$ salta en $4\pi\sigma$. Esa es la condición de borde de una superficie cargada.
#
# ## Experimento 4 — El hilo infinito necesita un corte
#
# En el plano medio de un hilo de longitud $2L$ y densidad $\lambda$, a distancia $s$,
# $$\phi_L(s) = \lambda\int_{-L}^{L}\frac{dz'}{\sqrt{s^2+z'^2}} = 2\lambda\,\operatorname{arcsinh}\frac{L}{s}.$$
#
# ### Predecí
# ¿Qué le pasa a $\phi_L(s)$ cuando $L\to\infty$? ¿Y a la **diferencia** $\phi_L(s)-\phi_L(s_0)$?

# %%
phi_hilo = lambda s, L, lam=1.0: 2 * lam * np.arcsinh(L / s)
s = np.geomspace(0.01, 100, 300)
fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
for L, c in zip([10, 100, 1000], RAMPA[1:]):
    axs[0].semilogx(s, phi_hilo(s, L), color=c, label=f"L = {L}")
    axs[1].semilogx(s, phi_hilo(s, L) - phi_hilo(1.0, L), color=c, label=f"L = {L}")
axs[1].semilogx(s, -2 * np.log(s), "k--", lw=1, label="−2λ ln s")
axs[0].set(xlabel="s", ylabel="φ_L(s)", title="el potencial crece sin límite con L"); axs[0].legend()
axs[1].set(xlabel="s", ylabel="φ_L(s) − φ_L(1)", ylim=(-12, 12), title="las diferencias convergen"); axs[1].legend()
plt.tight_layout(); guardar(fig, "nb03_hilo"); plt.show()

verificar("φ_L(3) − φ_L(1) → −2λ ln 3 para L = 10⁴", phi_hilo(3.0, 1e4) - phi_hilo(1.0, 1e4), -2 * np.log(3), tol=1e-7)

# %% [markdown]
# ### ¿Qué pasó?
# $\phi_L(s)\approx 2\lambda\ln(2L/s)$ diverge cuando $L\to\infty$: con carga que llega al infinito, **no podemos poner el cero del potencial en el infinito**. Pero las diferencias de potencial, que son lo único que se mide y lo único que determina $\mathbf E$, convergen a $-2\lambda\ln(s/s_0)$.
#
# Para $s \gg L$ (se ve en el extremo derecho de la curva $L=10$), $\phi_L \approx 2\lambda L/s = Q/s$: de lejos, el hilo finito se ve como una carga puntual.
#
# ## Explorá
# 1. **Guía 2, P3.** Dibujá las equipotenciales del hilo finito de longitud $2L$ en el plano $xz$, sumando cargas puntuales o con la fórmula cerrada. Comprobá numéricamente que son **elipses** con focos en los extremos del hilo: la suma de las distancias a los extremos es constante sobre cada curva.
# 2. **Dos cargas iguales.** En el Experimento 1 poné $q_2=+1$. La equipotencial que pasa por el punto de campo nulo forma un cono. Medí el ángulo que forma con el eje $z$ y comparalo con $\arctan\sqrt2\approx 54.7^\circ$. (Pista: cerca de ese punto, $\phi \approx$ cte. $+\,a(2z^2 - s^2)$.)
# 3. **Anillo frente a disco.** Usá la fórmula del anillo para dibujar el potencial de un anillo. ¿Dónde se parece al de una carga puntual? ¿Qué pasa cerca del alambre?
# 4. **Guía 2, P1.** Calculá el potencial de una esfera uniformemente cargada integrando $-\int_\infty^r \mathbf E\cdot d\mathbf l$ numéricamente, y compará con la fórmula analítica dentro y fuera.
