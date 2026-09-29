# %% [markdown]
# # Clase 6 — Desarrollo multipolar
#
# **Objetivos**
# - Ver cuánto mejora la aproximación del potencial lejano al agregar cada término: monopolo, dipolo, cuadrupolo.
# - Comprobar cómo cambian los momentos al mover el origen, y cuándo no cambian.
# - Descubrir, promediando el campo de un dipolo dentro de una esfera, que la fórmula $1/r^3$ del dipolo ideal está **incompleta** en el origen.
# - Explorar la energía de interacción entre dos dipolos y sus configuraciones de equilibrio.
#
# **Material relacionado:** notas de la Clase 6. Guía 3: problemas 5 a 7.

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
def momentos(q, r):
    """Carga total, momento dipolar y tensor cuadrupolar (sin traza) de cargas q en posiciones r (N x 3)."""
    Q = np.sum(q)
    p = q @ r
    r2 = np.sum(r**2, axis=1)
    Qij = np.einsum("n,ni,nj->ij", q, 3 * r, r) - np.sum(q * r2) * np.eye(3)
    return Q, p, Qij

def potencial_exacto(q, r, puntos):
    """φ en puntos (M x 3) de cargas puntuales."""
    return np.sum(q[None, :] / np.linalg.norm(puntos[:, None, :] - r[None, :, :], axis=2), axis=1)

def direcciones(M=400):
    """M direcciones casi uniformes (espiral de Fibonacci)."""
    i = np.arange(M) + 0.5
    th = np.arccos(1 - 2 * i / M); ph = np.pi * (1 + 5**0.5) * i
    return np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], axis=1)

# %% [markdown]
# ## Experimento 1 ★ — ¿Cuánto mejora cada término?
#
# Una "molécula" de 6 cargas al azar dentro de una esfera de radio 1. Comparamos el potencial exacto a distancia $r$ con las aproximaciones
# $$\phi^{(0)}=\frac qr,\qquad \phi^{(1)}=\phi^{(0)}+\frac{\mathbf p\cdot\hat{\mathbf r}}{r^2},\qquad \phi^{(2)}=\phi^{(1)}+\frac{Q_{ij}\hat r_i\hat r_j}{2r^3},$$
# y medimos el error cuadrático medio sobre todas las direcciones.
#
# ### Predecí
# ¿Cómo decae el error de cada aproximación con $r$? ¿Con qué potencia de $r$?

# %%
rng = np.random.default_rng(3)
r_mol = rng.normal(size=(6, 3)); r_mol *= (rng.uniform(0.3, 1.0, 6) / np.linalg.norm(r_mol, axis=1))[:, None]
q_mol = rng.uniform(-1, 1, 6); q_mol[0] += 0.8           # carga neta distinta de cero

def errores(q, r, rs):
    Q, p, Qij = momentos(q, r)
    n = direcciones()
    out = []
    for R in rs:
        exacto = potencial_exacto(q, r, R * n)
        a0 = Q / R
        a1 = a0 + (n @ p) / R**2
        a2 = a1 + np.einsum("mi,ij,mj->m", n, Qij, n) / (2 * R**3)
        out.append([np.sqrt(np.mean((exacto - a)**2)) for a in (a0, a1, a2)])
    return np.array(out)

rs = np.geomspace(2, 200, 30)
err = errores(q_mol, r_mol, rs)
fig, ax = plt.subplots(figsize=(6.8, 4.8))
for k, (c, nombre) in enumerate(zip(RAMPA[1:], ["solo monopolo", "+ dipolo", "+ cuadrupolo"])):
    ax.loglog(rs, err[:, k], "o-", ms=3, color=c, label=nombre)
ax.set(xlabel="distancia r", ylabel="error cuadrático medio de φ", title="desarrollo multipolar truncado"); ax.legend()
guardar(fig, "nb06_truncado"); plt.show()

lejos = rs > 20
for k, esperada in enumerate([-2, -3, -4]):
    pendiente = np.polyfit(np.log(rs[lejos]), np.log(err[lejos, k]), 1)[0]
    verificar(f"pendiente del error con {k} término(s) de corrección = {esperada}", pendiente, esperada, tol=2e-2)

# %% [markdown]
# ### ¿Qué pasó?
# Truncar el desarrollo en el orden $\ell$ deja un error del orden del término siguiente, $\sim 1/r^{\ell+2}$. Cada término agregado hace que el error baje una potencia más de $r$. Lejos, el primer término no nulo domina, y el resto son correcciones cada vez más chicas.
#
# ## Experimento 2 — Mover el origen
#
# Calculamos los momentos respecto del origen y respecto de un punto $\mathbf a$. Las notas deducen
# $$\tilde{\mathbf p}=\mathbf p-q\,\mathbf a,\qquad \tilde Q_{ij}=Q_{ij}-3(a_ip_j+a_jp_i)+2(\mathbf a\cdot\mathbf p)\,\delta_{ij}+q\,(3a_ia_j-a^2\delta_{ij}).$$
#
# ### Predecí
# ¿Existe un origen en el que el momento dipolar de la molécula (que tiene carga neta) se anule?

# %%
a = np.array([0.4, -0.7, 1.1])
q0, p0, Q0 = momentos(q_mol, r_mol)
q1, p1, Q1 = momentos(q_mol, r_mol - a)
Q1_formula = Q0 - 3 * (np.outer(a, p0) + np.outer(p0, a)) + 2 * (a @ p0) * np.eye(3) + q0 * (3 * np.outer(a, a) - (a @ a) * np.eye(3))
verificar("p̃ = p − q a (componente x)", p1[0], (p0 - q0 * a)[0], tol=1e-12)
verificar("máx |Q̃ − fórmula|", np.abs(Q1 - Q1_formula).max(), 0.0, tol=1e-12)

centro = p0 / q0                                        # "centro de carga"
_, p_c, _ = momentos(q_mol, r_mol - centro)
print("centro de carga:", centro.round(4))
verificar("|p| respecto del centro de carga", np.linalg.norm(p_c), 0.0, tol=1e-12)

# sistema neutro: el dipolo no depende del origen
q_neutro = q_mol - q_mol.mean()
_, pa, _ = momentos(q_neutro, r_mol); _, pb, _ = momentos(q_neutro, r_mol - a)
verificar("sistema neutro: p no depende del origen", np.abs(pa - pb).max(), 0.0, tol=1e-12)

# %% [markdown]
# ### ¿Qué pasó?
# Si $q\neq0$, siempre podemos elegir el origen en el **centro de carga** $\mathbf a=\mathbf p/q$, donde el dipolo se anula: el término $1/r^2$ desaparece y la primera corrección al monopolo es cuadrupolar. Si $q=0$, el dipolo es el mismo desde cualquier origen: es una propiedad del sistema.
#
# ## Experimento 3 — El dipolo finito y el dipolo ideal
#
# Un dipolo $\pm q$ separado por $d$, con $p = qd = 1$ fijo. Al achicar $d$, se acerca al dipolo ideal, cuyas líneas de campo son $r = C\sin^2\theta$.
#
# ### Predecí
# Promediamos el campo del dipolo finito dentro de una esfera de radio $R=1$ que lo contiene. ¿Cuánto da el promedio? ¿Y el promedio de la fórmula ideal $\mathbf E=[3(\mathbf p\cdot\hat{\mathbf r})\hat{\mathbf r}-\mathbf p]/r^3$?

# %%
def dibujar_dipolo(d=0.4):
    x = np.linspace(-2, 2, 500); z = np.linspace(-2, 2, 500)
    X, Z = np.meshgrid(x, z)
    q = 1 / d
    Psi = q * ((Z - d / 2) / np.hypot(X, Z - d / 2) - (Z + d / 2) / np.hypot(X, Z + d / 2))
    Psi_ideal = -X**2 / (X**2 + Z**2)**1.5              # −p sin²θ / r
    Psi_ideal[X**2 + Z**2 < 0.25**2] = np.nan             # el ideal es singular en el origen
    fig, ax = plt.subplots(figsize=(6, 6))
    niveles = -np.array([0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.2, 3.0, 4.0])
    ax.contour(X, Z, Psi, levels=np.sort(niveles), colors="0.25", linewidths=1, linestyles="solid")
    ax.contour(X, Z, Psi_ideal, levels=np.sort(niveles), colors=COLORES[1], linewidths=1, linestyles="dashed")
    ax.plot(0, d / 2, "o", color=COLORES[7], ms=7); ax.plot(0, -d / 2, "o", color=COLORES[0], ms=7)
    ax.add_patch(plt.Circle((0, 0), 1, fill=False, color="0.5", lw=1))
    ax.set(aspect="equal", title=f"d = {d:.2f}: dipolo finito (gris) e ideal (naranja, r = C sin²θ)"); ax.grid(False)
    plt.show()

interactuar(dibujar_dipolo, d=deslizador("d", 0.4, 0.05, 1.5, 0.05))

def campo_medio_esfera(q, zq, R=1.0, n=400):
    """⟨E_z⟩ en una esfera de radio R: (1/V)∫∇(−φ) = −(1/V)∮ φ n_z da, integrado en cos θ."""
    c, w = np.polynomial.legendre.leggauss(n)
    phi = sum(qi / np.sqrt(R**2 + zi**2 - 2 * R * zi * c) for qi, zi in zip(q, zq))
    V = 4 * np.pi * R**3 / 3
    return -np.sum(phi * c * w) * 2 * np.pi * R**2 / V

for d, z0 in [(0.3, 0.0), (0.05, 0.0), (0.3, 0.4)]:
    q = 1 / d
    Ez = campo_medio_esfera([q, -q], [z0 + d / 2, z0 - d / 2])
    verificar(f"⟨E_z⟩ en la esfera (d = {d}, centro del dipolo en z = {z0}) = −p/R³", Ez, -1.0, tol=1e-8)

# promedio angular de la fórmula ideal: ⟨3 cos²θ − 1⟩ = 0 en cada cáscara
c, w = np.polynomial.legendre.leggauss(50)
verificar("promedio angular de la fórmula ideal (3cos²θ − 1)/2", np.sum((3 * c**2 - 1) * w) / 4, 0.0, tol=1e-12)

# %% [markdown]
# ### ¿Qué pasó?
# El campo promedio dentro de la esfera es exactamente $-\mathbf p/R^3$, cualquiera sea $d$ y dondequiera que esté el dipolo adentro. Pero la fórmula ideal $[3(\mathbf p\cdot\hat{\mathbf r})\hat{\mathbf r}-\mathbf p]/r^3$ tiene promedio angular nulo en cada cáscara, así que "promediada" da cero. Lo que falta es un término concentrado en el origen: $\mathbf E_{\text{dip}} = \frac{3(\mathbf p\cdot\hat{\mathbf r})\hat{\mathbf r}-\mathbf p}{r^3}-\frac{4\pi}{3}\mathbf p\,\delta^3(\mathbf r)$ (deducido en las notas). Dentro del dipolo, entre las cargas, el campo apunta de $+$ a $-$, es decir **opuesto** a $\mathbf p$. Ese término lo recuerda.
#
# ## Experimento 4 — Dos dipolos
#
# Dos dipolos iguales, de momento $p$, separados una distancia $r$ a lo largo del eje $x$, girando en el plano $xy$ con ángulos $\theta_1$ y $\theta_2$ respecto del eje $x$. La energía de interacción es
# $$U=\frac{\mathbf p_1\cdot\mathbf p_2-3(\mathbf p_1\cdot\hat{\mathbf n})(\mathbf p_2\cdot\hat{\mathbf n})}{r^3}=\frac{p^2}{r^3}\left[\cos(\theta_1-\theta_2)-3\cos\theta_1\cos\theta_2\right].$$
#
# ### Predecí
# ¿Cuál es la configuración de mínima energía? ¿Uno detrás del otro ($\to\,\to$) o uno al lado del otro ($\uparrow\,\downarrow$)?

# %%
th = np.linspace(-np.pi, np.pi, 361)
T1, T2 = np.meshgrid(th, th)
U = np.cos(T1 - T2) - 3 * np.cos(T1) * np.cos(T2)          # p = r = 1
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.pcolormesh(np.degrees(T1), np.degrees(T2), U, cmap="RdBu_r", vmin=-2, vmax=2, shading="auto")
ax.plot([0, 180, -180], [0, 180, -180], "k*", ms=12)
ax.set(xlabel="θ1 (grados)", ylabel="θ2 (grados)", title="U(θ1, θ2) en unidades de p²/r³  (★: mínimos)")
fig.colorbar(im, ax=ax); ax.grid(False); plt.show()
verificar("mínimo de U = −2p²/r³ (uno detrás del otro)", U.min(), -2.0, tol=1e-6)

# comparación con la energía exacta de cuatro cargas (dipolos finitos muy chicos)
def U_cuatro_cargas(t1, t2, r=1.0, d=1e-3):
    q = 1 / d
    u1, u2 = np.array([np.cos(t1), np.sin(t1)]), np.array([np.cos(t2), np.sin(t2)])
    c1, c2 = np.array([0.0, 0.0]), np.array([r, 0.0])
    cargas1 = [(q, c1 + d / 2 * u1), (-q, c1 - d / 2 * u1)]
    cargas2 = [(q, c2 + d / 2 * u2), (-q, c2 - d / 2 * u2)]
    return sum(qa * qb / np.linalg.norm(ra - rb) for qa, ra in cargas1 for qb, rb in cargas2)

t1, t2 = 0.3, 1.2
verificar("fórmula dipolo-dipolo frente a cuatro cargas (d = 10⁻³)", U_cuatro_cargas(t1, t2),
          np.cos(t1 - t2) - 3 * np.cos(t1) * np.cos(t2), tol=1e-5)

# %% [markdown]
# ### ¿Qué pasó?
# El mínimo absoluto, $U=-2p^2/r^3$, es **uno detrás del otro** ($\theta_1=\theta_2=0$ o $\pi$). Uno al lado del otro y antiparalelos ($\uparrow\downarrow$) es un mínimo local más alto, $-p^2/r^3$. La fórmula de las notas coincide con la energía exacta de cuatro cargas cuando los dipolos son chicos. Por eso las moléculas polares tienden a formar cadenas cabeza con cola.
#
# ## Explorá
# 1. **Guía 3, P6.** Construí una distribución de cargas con $q=0$, $\mathbf p=0$ y $Q_{ij}=0$, y verificá con `momentos` que el error del Experimento 1 decae como $1/r^5$ (octupolo) ya desde el primer término.
# 2. **Guía 3, P7.** Calculá con `momentos` los tensores $Q_{ij}$ de los dos cuadrados de cuatro cargas del problema, y compará con tus cuentas.
# 3. **El núcleo deformado.** Distribuí muchas cargas uniformemente dentro de un esferoide alargado de semiejes $a$ (en $z$) y $b$, y verificá que $Q_{zz}\approx\frac25\,Q\,(a^2-b^2)$ (deducido en las notas).
# 4. **Cuatro dipolos en un cuadrado.** Con la fórmula dipolo-dipolo, minimizá numéricamente (`scipy.optimize.minimize`) la energía de cuatro dipolos en los vértices de un cuadrado. ¿Qué configuración aparece?
