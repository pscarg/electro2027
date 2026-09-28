# %% [markdown]
# # Clase 5 — Conductores y capacidad
#
# **Objetivos**
# - Calcular numéricamente cómo se reparte la carga sobre un conductor de forma arbitraria (con simetría de revolución), y ver que **se acumula en bordes y puntas**.
# - Verificar la simetría, los signos y la positividad de la **matriz de capacidades** de dos esferas.
# - Comprobar que la fuerza calculada por trabajo virtual **a carga constante** y **a potencial constante** es la misma, y cuánta energía aporta la batería.
#
# **Material relacionado:** notas de la Clase 5. Guía 2: problema 7. Guía 3: problemas 1 a 4.
#
# **El método.** Dividimos la superficie del conductor en anillos (paneles) con densidad $\sigma_k$ uniforme en cada uno, y exigimos que el potencial en el centro de cada panel sea el del conductor. Queda un sistema lineal $A\,\sigma = \phi$, donde $A_{ik}$ es el potencial en el panel $i$ producido por el panel $k$ con densidad 1. Se llama **método de momentos**. Cada anillo produce el potencial $\frac{2q}{\pi}\frac{K(m)}{D}$ deducido en la Clase 3.

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
# herramienta: momentos_axial v1 (NB05)
import warnings
from scipy.special import ellipk
from scipy.integrate import quad
warnings.filterwarnings("ignore", message=".*roundoff error.*")

def G_anillo(s, z, sp, zp):
    """Potencial en (s, z) de un anillo de carga 1 y radio sp en la altura zp (Clase 3)."""
    D2 = (s + sp)**2 + (z - zp)**2
    m = np.clip(4 * s * sp / D2, 0, 1 - 1e-16)
    return (2 / np.pi) * ellipk(m) / np.sqrt(D2)

def paneles(curvas):
    """curvas: lista de (s(t), z(t), bordes en t, número de conductor). Cada tramo es un panel."""
    return [(fs, fz, a, b, c) for fs, fz, tb, c in curvas for a, b in zip(tb[:-1], tb[1:])]

def _dl(fs, fz, t, h):
    """Longitud de arco por unidad de t (derivada numérica del perfil)."""
    return np.hypot((fs(t + h) - fs(t - h)) / (2 * h), (fz(t + h) - fz(t - h)) / (2 * h))

def matriz_momentos(P, ng=8):
    """A[i, k] = potencial en el centro del panel i producido por el panel k con σ = 1."""
    x, w = np.polynomial.legendre.leggauss(ng)
    centros = np.array([(p[0](0.5 * (p[2] + p[3])), p[1](0.5 * (p[2] + p[3]))) for p in P])
    A = np.zeros((len(P), len(P)))
    for k, (fs, fz, a, b, _) in enumerate(P):             # paneles lejanos: Gauss-Legendre
        t = 0.5 * (b - a) * x + 0.5 * (b + a)
        area = 2 * np.pi * fs(t) * _dl(fs, fz, t, 1e-6 * (b - a)) * 0.5 * (b - a) * w
        A[:, k] = G_anillo(centros[:, :1], centros[:, 1:], fs(t)[None], fz(t)[None]) @ area
    for i, (si, zi) in enumerate(centros):                 # panel propio y vecinos: singularidad logarítmica
        for k in range(max(0, i - 1), min(len(P), i + 2)):
            fs, fz, a, b, _ = P[k]
            f = lambda t: 2 * np.pi * fs(t) * _dl(fs, fz, t, 1e-7 * (b - a)) * G_anillo(si, zi, fs(t), fz(t))
            A[i, k] = quad(f, a, b, points=[0.5 * (a + b)] if k == i else None, limit=200)[0]
    return A, centros

def areas_paneles(P):
    return np.array([quad(lambda t: 2 * np.pi * fs(t) * _dl(fs, fz, t, 1e-7 * (b - a)), a, b)[0]
                     for fs, fz, a, b, _ in P])

def resolver(curvas, potenciales):
    """Densidades σ de cada panel si el conductor c está al potencial potenciales[c]."""
    P = paneles(curvas)
    A, centros = matriz_momentos(P)
    cond = np.array([p[4] for p in P])
    sigma = np.linalg.solve(A, np.asarray(potenciales, float)[cond])
    return sigma, centros, areas_paneles(P), cond

def matriz_capacidades(curvas, n_cond):
    """C[a, b] = carga del conductor a cuando b está a potencial 1 y los demás a 0."""
    P = paneles(curvas)
    A, _ = matriz_momentos(P)
    cond = np.array([p[4] for p in P]); ar = areas_paneles(P)
    C = np.zeros((n_cond, n_cond))
    for b in range(n_cond):
        sigma = np.linalg.solve(A, (cond == b).astype(float))
        for a in range(n_cond):
            C[a, b] = np.sum((sigma * ar)[cond == a])
    return C
# fin herramienta

# %% [markdown]
# ## Experimento 1 ★ — ¿Dónde se acumula la carga?
#
# Ponemos a potencial $\phi = 1$ tres conductores aislados:
# - una **esfera** de radio 1;
# - un **esferoide alargado**, con semiejes $a$ (a lo largo de $z$) y $b$;
# - un **disco** de radio 1 y espesor nulo.
#
# ### Predecí
# 1. ¿En qué conductor la densidad $\sigma$ es uniforme?
# 2. En el esferoide, ¿dónde es mayor $\sigma$: en las puntas o en el ecuador? ¿Cuántas veces mayor, si $a/b = 5$?
# 3. En el disco, ¿qué pasa con $\sigma$ al acercarse al borde?

# %%
N = 120
esfera = [(np.sin, np.cos, np.linspace(0, np.pi, N + 1), 0)]
disco = [(lambda t: np.sin(np.pi * t / 2), lambda t: 0 * t, np.linspace(0, 1, N + 1), 0)]   # paneles más finos cerca del borde

def esferoide(a, b):
    return [(lambda t: b * np.sin(t), lambda t: a * np.cos(t), np.linspace(0, np.pi, N + 1), 0)]

def capacidad_prolato(a, b):
    e = np.sqrt(a**2 - b**2)
    return e / np.arccosh(a / b)

resultados = {}
for nombre, curvas in [("esfera", esfera), ("esferoide a/b = 5", esferoide(2.5, 0.5)), ("disco", disco)]:
    sigma, centros, ar, _ = resolver(curvas, [1.0])
    resultados[nombre] = (sigma, centros, ar)

fig, axs = plt.subplots(1, 2, figsize=(12, 5))
for (nombre, (sigma, centros, ar)), c in zip(resultados.items(), COLORES):
    media = np.sum(sigma * ar) / np.sum(ar)
    axs[0].scatter(centros[:, 0], centros[:, 1], c=sigma / media, cmap="Blues", vmin=0, vmax=4, s=12)
    axs[0].scatter(-centros[:, 0], centros[:, 1], c=sigma / media, cmap="Blues", vmin=0, vmax=4, s=12)
    arco = np.concatenate([[0], np.cumsum(np.hypot(np.diff(centros[:, 0]), np.diff(centros[:, 1])))])
    axs[1].plot(arco / arco[-1], sigma / media, color=c, label=nombre)
axs[0].set(aspect="equal", title="perfiles (color: σ / σ media)", xlabel="x", ylabel="z"); axs[0].grid(False)
axs[1].set(xlabel="posición a lo largo del perfil (0 → 1)", ylabel="σ / σ media", ylim=(0, 6),
           title="densidad de carga sobre la superficie"); axs[1].legend()
plt.tight_layout(); guardar(fig, "nb05_densidades"); plt.show()

Q = {k: np.sum(s * a) for k, (s, _, a) in resultados.items()}
verificar("capacidad de la esfera = R", Q["esfera"], 1.0, tol=1e-6)
verificar("capacidad del disco = 2R/π", Q["disco"], 2 / np.pi, tol=1e-6)
verificar("capacidad del esferoide (fórmula exacta)", Q["esferoide a/b = 5"], capacidad_prolato(2.5, 0.5), tol=1e-5)
s_disco, c_disco, _ = resultados["disco"]
verificar("σ del disco en r = 0.5 frente a Q/(2πR√(R²−r²))", np.interp(0.5, c_disco[:, 0], s_disco),
          Q["disco"] / (2 * np.pi * np.sqrt(1 - 0.25)), tol=1e-4)
s_esf, _, _ = resultados["esferoide a/b = 5"]
verificar("esferoide: σ(punta)/σ(ecuador) ≈ a/b", s_esf[0] / s_esf[N // 2], 5.0, tol=2e-2)

# %% [markdown]
# ### ¿Qué pasó?
# - En la **esfera** $\sigma$ es uniforme, y la capacidad es $C = R$ (en el sistema gaussiano la capacidad tiene unidades de longitud).
# - En el **esferoide** la carga se amontona en las puntas. Para un elipsoide conductor, $\sigma$ es proporcional a la distancia del centro al plano tangente, y en las puntas vale $a/b$ veces el valor del ecuador.
# - En el **disco**, $\sigma(r) = \dfrac{Q}{2\pi R\sqrt{R^2-r^2}}$ **diverge en el borde**, aunque la carga total es finita. El campo cerca de un filo es muy intenso: es el principio del pararrayos, y la razón por la que los electrodos de alta tensión tienen bordes redondeados.
#
# Probá otras proporciones del esferoide:

# %%
def punta(a_sobre_b=5.0):
    b = 0.5; a = a_sobre_b * b
    if abs(a - b) < 1e-9:
        a += 1e-6
    sigma, centros, ar, _ = resolver(esferoide(a, b), [1.0])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(centros[:, 1], sigma, color=COLORES[1])
    ax.set(xlabel="z", ylabel="σ", title=f"a/b = {a_sobre_b:.1f}:  σ(punta)/σ(ecuador) = {sigma[0] / sigma[N // 2]:.2f},   C = {np.sum(sigma * ar):.4f}")
    plt.show()

interactuar(punta, a_sobre_b=deslizador("a/b", 5.0, 1.0, 10.0, 0.5))

# %% [markdown]
# ## Experimento 2 — La matriz de capacidades de dos esferas
#
# Dos esferas de radios $R_1 = 1$ y $R_2 = 0.5$, con centros a distancia $d$. La matriz $C$ se arma columna por columna: se pone una esfera a potencial 1 y la otra a 0, y se miden las cargas.
#
# ### Predecí
# 1. ¿$C_{12}$ es positivo o negativo?
# 2. ¿$C_{12} = C_{21}$? El método numérico no impone esa simetría, así que la estamos poniendo a prueba.
# 3. Lejos ($d \gg R$), ¿cómo decae $C_{12}$ con $d$?

# %%
def dos_esferas(R1, R2, d, n=80):
    tb = np.linspace(0, np.pi, n + 1)
    return [(lambda t: R1 * np.sin(t), lambda t: d / 2 + R1 * np.cos(t), tb, 0),
            (lambda t: R2 * np.sin(t), lambda t: -d / 2 + R2 * np.cos(t), tb, 1)]

R1, R2 = 1.0, 0.5
C = matriz_capacidades(dos_esferas(R1, R2, 2.5), 2)
print("C (d = 2.5) =\n", C.round(5))
print("autovalores:", np.linalg.eigvalsh((C + C.T) / 2).round(5))
verificar("simetría C12 = C21", C[1, 0], C[0, 1], tol=1e-5)

ds = np.array([2.0, 3.0, 5.0, 10.0])
C12 = np.array([matriz_capacidades(dos_esferas(R1, R2, d), 2)[0, 1] for d in ds])
aprox = -(1 / ds) / (1 / (R1 * R2) - 1 / ds**2)
fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.plot(ds, -C12, "o", label="numérico  −C12")
dd = np.linspace(1.6, 10.5, 200)
ax.plot(dd, (1 / dd) / (1 / (R1 * R2) - 1 / dd**2), "--", lw=1.2, label="aproximación lejana (ver notas)")
ax.plot(dd, R1 * R2 / dd, ":", lw=1.2, label="R1 R2 / d")
ax.set(xlabel="distancia entre centros d", ylabel="−C12", title="coeficiente de inducción de dos esferas"); ax.legend()
guardar(fig, "nb05_C12"); plt.show()
verificar("C12 lejos (d = 10) frente a la aproximación de las notas", C12[-1], aprox[-1], tol=1e-3)

# %% [markdown]
# ### ¿Qué pasó?
# - $C_{11}, C_{22} > 0$ y $C_{12} = C_{21} < 0$. Si subimos una esfera a potencial 1 con la otra a tierra, la de tierra adquiere carga **negativa** inducida.
# - La simetría $C_{12}=C_{21}$ (teorema de reciprocidad de Green, en las notas) se cumple numéricamente a $10^{-7}$ aunque el método no la impone.
# - $C$ tiene autovalores positivos: la energía $\tfrac12\phi^TC\phi = \frac{1}{8\pi}\int E^2$ nunca es negativa.
# - Lejos, $C_{12}\approx -R_1R_2/d$.
#
# ## Experimento 3 — Trabajo virtual: carga fija o potencial fijo
#
# Las dos esferas tienen cargas $Q_1 = 1$ y $Q_2 = -0.5$ a distancia $d_0 = 3$. La fuerza entre ellas se puede calcular de dos maneras:
# - **aisladas** (carga fija): $F = -\partial U/\partial d$ con $U = \tfrac12 Q^TC^{-1}Q$;
# - **conectadas a baterías** (potenciales fijos): $F = +\partial U/\partial d$ con $U = \tfrac12\phi^TC\phi$.
#
# ### Predecí
# ¿Dan lo mismo? Si la batería mantiene los potenciales, ¿cuánta energía entrega, comparada con el cambio de $U$?

# %%
Qs = np.array([1.0, -0.5]); d0, h = 3.0, 0.02
Cm, C0, Cp = (matriz_capacidades(dos_esferas(R1, R2, d), 2) for d in (d0 - h, d0, d0 + h))
Cm, C0, Cp = ((M + M.T) / 2 for M in (Cm, C0, Cp))                  # simetrizamos (diferencia ~1e-7)
phis = np.linalg.solve(C0, Qs)                                       # potenciales en el estado d0

U_Q = lambda M: 0.5 * Qs @ np.linalg.solve(M, Qs)
U_phi = lambda M: 0.5 * phis @ M @ phis
F_Q = -(U_Q(Cp) - U_Q(Cm)) / (2 * h)
F_phi = +(U_phi(Cp) - U_phi(Cm)) / (2 * h)
print(f"potenciales: φ1 = {phis[0]:.4f}, φ2 = {phis[1]:.4f}")
print(f"F a carga fija     = {F_Q:+.6f}")
print(f"F a potencial fijo = {F_phi:+.6f}   (negativa: atracción)")
verificar("F(Q fija) = F(φ fijo)", F_phi, F_Q, tol=1e-4)

dC = (Cp - Cm) / 2                                                   # cambio de C al alejar las esferas h
W_bateria = phis @ (dC @ phis)                                        # Σ φ_α dQ_α
dU_phi = 0.5 * phis @ dC @ phis
verificar("trabajo de la batería = 2 × cambio de U (a φ fijo)", W_bateria, 2 * dU_phi, tol=1e-10)

# %% [markdown]
# ### ¿Qué pasó?
# Las dos maneras dan **la misma fuerza**, con signos opuestos en la derivada. La fuerza depende del estado de las cargas en ese instante, no de si hay una batería conectada. A potencial fijo, la batería entrega el doble del cambio de energía del campo: la mitad se almacena en el campo y la otra mitad es el trabajo mecánico.
#
# ## Explorá
# 1. **Dos esferas conectadas.** Poné las dos esferas al mismo potencial (vector de potenciales `[1, 1]` en `resolver`) y a distancia grande. Compará las densidades medias: ¿se cumple $\sigma_1/\sigma_2 \approx R_2/R_1$?
# 2. **Una esfera cargada atrae a una neutra.** Con $Q_1 = 1$ y $Q_2 = 0$, calculá la fuerza a carga fija en función de $d$. ¿Es atractiva? ¿Cómo decae? (Se entiende con el dipolo inducido, en las Clases 6 y 11.)
# 3. **Guía 3, P1.** Una cáscara con carga $Q$ y una carga puntual externa. ¿Vale la superposición si la cáscara es aislante? ¿Y si es conductora? Usá dos esferas (una muy chica, que hace de carga puntual) para explorarlo.
# 4. **Un conductor con una punta.** Armá un perfil propio, por ejemplo una esfera con una aguja, con `paneles` y `resolver`. ¿Dónde es máximo $\sigma$?
