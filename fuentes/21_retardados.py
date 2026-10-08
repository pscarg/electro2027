# %% [markdown]
# # Clase 21 — Paquetes de ondas, la función de Green de ondas y los potenciales retardados
#
# **Objetivos**
# - Comparar la respuesta a un destello de un punto, de una recta y de un plano, y ver de dónde sale la estela en dos y en una dimensión (principio de Huygens).
# - Descomponer un pulso en frecuencias: comprobar el teorema de Parseval y la relación entre la duración y el ancho de banda.
# - Ver cómo la suma de modos $e^{ik\mathcal R}/\mathcal R$ arma un pulso retardado (y uno adelantado con $e^{-ik\mathcal R}/\mathcal R$).
# - Medir cuándo vale la aproximación cuasiestática.
#
# **Material relacionado:** notas de la Clase 21. Guía 8: problemas 6 a 8.
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
from scipy.special import erf
from scipy.integrate import trapezoid
from matplotlib import animation

# %% [markdown]
# ## Experimento 1 ★ — Un destello en tres, dos y una dimensión
#
# Las notas (sección 4) dan la respuesta a un destello $\delta(t)$, con $c=1$:
# - en un punto (tres dimensiones), $\psi_3=\delta(t-r)/r$: una cáscara esférica que se aleja;
# - a lo largo de una recta (dos dimensiones), $\psi_2=2\,\Theta(t-s)/\sqrt{t^2-s^2}$;
# - sobre un plano (una dimensión), $\psi_1=2\pi\,\Theta(t-|x|)$.
#
# Acá el destello es una gaussiana angosta de área 1, $\mathcal S(t)=e^{-t^2/2\tau^2}/(\sqrt{2\pi}\,\tau)$, y la respuesta del punto es $\mathcal S(t-r)/r$. Las de la recta y del plano **no** usan las fórmulas: suman las respuestas de muchos destellos puntuales, uno por cada elemento de la recta o del plano, como la integral de las notas. Después las comparamos con las fórmulas.
#
# ### Predecí
# A distancia 2 de la fuente, ¿cómo se ve el destello en cada caso? ¿Cuándo empieza la señal y cuándo se apaga?

# %%
def destello(t, tau):
    """Gaussiana de área 1 y ancho tau: un destello δ(t) suavizado."""
    return np.exp(-t**2 / (2 * tau**2)) / (np.sqrt(2 * np.pi) * tau)

def respuesta_recta(rho, ts, tau, L=12.0, dz=1e-3):
    """Suma de destellos puntuales 𝒮(t − 𝓡)/𝓡 dz' a lo largo del eje z (|z'| < L)."""
    z = np.arange(-L, L + dz / 2, dz); R = np.sqrt(rho**2 + z**2)
    return np.array([np.sum(destello(t - R, tau) / R) * dz for t in np.atleast_1d(ts)])

def respuesta_plano(x, ts, tau, L=12.0, ds=5e-4):
    """Suma de destellos puntuales sobre el plano x = 0, por anillos de radio s' y área 2πs' ds' (s' < L)."""
    s = np.arange(ds / 2, L, ds); R = np.sqrt(x**2 + s**2)
    return np.array([np.sum(2 * np.pi * s * destello(t - R, tau) / R) * ds for t in np.atleast_1d(ts)])

d, tau = 2.0, 0.03
ts = np.linspace(0, 8, 1601)
p3 = destello(ts - d, tau) / d
p2 = respuesta_recta(d, ts, tau)
p1 = respuesta_plano(d, ts, tau)

fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
axs[0].plot(ts, p3, color=COLORES[0])
axs[0].set(title=r"punto (3D): $\psi=\mathcal{S}(t-r)/r$", xlabel="t", ylabel="ψ a distancia 2")
tt = ts[ts > d + 0.02]
axs[1].plot(ts, p2, color=COLORES[1], label="suma de destellos")
axs[1].plot(tt, 2 / np.sqrt(tt**2 - d**2), "k--", lw=1, label="2/√(t² − s²)")
axs[1].set(title="recta (2D)", xlabel="t", ylim=(0, 2.5)); axs[1].legend(fontsize=9)
axs[2].plot(ts, p1, color=COLORES[2], label="suma de destellos")
axs[2].plot(ts, 2 * np.pi * (ts > d), "k--", lw=1, label="2π Θ(t − |x|)")
axs[2].set(title="plano (1D)", xlabel="t", ylim=(0, 8)); axs[2].legend(fontsize=9, loc="lower right")
for ax in axs:
    ax.axvline(d, color="0.6", lw=0.8)
plt.tight_layout(); plt.show()
guardar(fig, "nb21_huygens")

i4 = np.argmin(np.abs(ts - 4.0))
verificar("recta: la suma de destellos da 2/√(t² − s²) (t = 4)", p2[i4], 2 / np.sqrt(16 - d**2), tol=1e-3)
verificar("plano: la suma de destellos da 2π (t = 4)", p1[i4], 2 * np.pi, tol=1e-4)
verificar("punto: el área del pulso es 1/r", trapezoid(p3, ts), 1 / d, tol=1e-6)
verificar("punto: después del frente no queda nada (t = 4)", p3[i4], 0.0, tol=1e-12)

# %% [markdown]
# **¿De dónde sale la estela?** Reemplazamos la recta por fuentes puntuales separadas una distancia $h$ (cada una con peso $h$). Cada una manda un destello nítido, que llega a distancia $s$ en el instante $\sqrt{s^2+z_n'^2}$ con área $h/\sqrt{s^2+z_n'^2}$. La curva gruesa es el promedio de los picos en la ventana $[t-h,\,t]$. Mové $h$.

# %%
def estela(h=0.5):
    zn = np.arange(-int(12 / h), int(12 / h) + 1) * h; Rn = np.sqrt(d**2 + zn**2)
    pn = sum(h * destello(ts - R, 0.02) / R for R in Rn)
    m = max(1, int(round(h / (ts[1] - ts[0]))))
    promedio = np.convolve(pn, np.ones(m) / m)[:len(pn)]          # promedio en [t − h, t]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(ts, pn, color=COLORES[1], lw=1, alpha=0.4, label=f"fuentes separadas h = {h:g}")
    ax.plot(ts, promedio, color=COLORES[1], lw=2.2, label="promedio en [t − h, t]")
    ax.plot(tt, 2 / np.sqrt(tt**2 - d**2), "k--", lw=1, label="recta continua")
    ax.set(xlabel="t", ylabel="ψ a distancia 2", ylim=(0, 4), title="cada pico es la señal de un punto de la recta")
    ax.legend(fontsize=9); plt.show()

interactuar(estela, h=deslizador("h", 0.5, 0.05, 1.5, 0.05))

# %% [markdown]
# Ahora los mapas en el plano $xy$, para un destello en el origen (tres dimensiones) y a lo largo del eje $z$ (dos dimensiones).

# %%
tau_a = 0.08
xg = np.linspace(-4, 4, 161); X, Y = np.meshgrid(xg, xg); Rg = np.hypot(X, Y)
rhos = np.linspace(0.02, 5.7, 300)
zs = np.arange(-8, 8 + 0.0025, 0.005); Rz = np.sqrt(rhos[:, None]**2 + zs[None, :]**2)

def mapas(t):
    m3 = destello(t - Rg, tau_a) / np.maximum(Rg, 0.05)
    perfil = np.sum(destello(t - Rz, tau_a) / Rz, axis=1) * 0.005   # suma de destellos sobre la recta
    return m3, np.interp(Rg, rhos, perfil)

tiempos = np.linspace(0.4, 3.8, 3 if PRUEBA else 35)
fig, axs = plt.subplots(1, 2, figsize=(9.5, 4.6))
m3, m2 = mapas(tiempos[0])
im3 = axs[0].imshow(m3, extent=(-4, 4, -4, 4), origin="lower", cmap="Blues", vmin=0, vmax=2.0)
im2 = axs[1].imshow(m2, extent=(-4, 4, -4, 4), origin="lower", cmap="Blues", vmin=0, vmax=2.0)
axs[0].set(title="punto (3D)", xlabel="x", ylabel="y"); axs[1].set(title="recta según z (2D)", xlabel="x")
for ax in axs:
    ax.grid(False)
titulo = fig.suptitle("")

def cuadro(n):
    m3, m2 = mapas(tiempos[n])
    im3.set_data(m3); im2.set_data(m2); titulo.set_text(f"t = {tiempos[n]:.2f}")
    return im3, im2

anim = animation.FuncAnimation(fig, cuadro, frames=len(tiempos), interval=150)
plt.close(fig)
display(HTML(anim.to_jshtml()))

# %% [markdown]
# ### ¿Qué pasó?
# Los tres frentes llegan en $t=2$: nada viaja más rápido que $c$. Pero después del frente:
# - el destello del **punto** llega entero y se va; detrás del frente no queda nada (en el mapa, el interior del anillo está vacío);
# - el de la **recta** deja una estela que decae como $2/t$: las señales de los puntos lejanos de la recta siguen llegando, cada vez más tarde y más débiles (como $1/R$), sin terminar nunca; justo después del frente se amontonan, porque los puntos cercanos al pie de la perpendicular están casi a la misma distancia (la figura con $h$ lo muestra pico por pico);
# - el del **plano** deja un escalón permanente: los anillos lejanos tienen más área en la misma proporción en que están más lejos.
#
# Solo en tres dimensiones un destello se ve como un destello (principio de Huygens). En electromagnetismo, $\psi$ es un potencial y los campos son sus derivadas: el escalón del plano da un campo $E=-\frac1c\partial_tA$ nítido (la lámina de la Clase 18), pero la estela de la recta deja una cola también en los campos.
#
# ## Experimento 2 — Un pulso y su espectro
#
# Un pulso gaussiano $f(t)=e^{-t^2/2\tau^2}\cos\omega_0t$, como el campo $E_y$ que pasa por un punto. La función `espectro` calcula $\tilde f(\omega)=\int f(t)\,e^{i\omega t}dt$ con la transformada rápida de Fourier. Las notas dan
# $$\tilde f(\omega)=\frac{\sqrt{2\pi}\,\tau}{2}\left[e^{-(\omega-\omega_0)^2\tau^2/2}+e^{-(\omega+\omega_0)^2\tau^2/2}\right],$$
# el teorema de Parseval, $\int f^2dt=\int\frac{d\omega}{2\pi}|\tilde f|^2$, y el producto de los anchos cuadráticos medios de la envolvente $|e^{-t^2/2\tau^2}|^2$ y del espectro $|\tilde f|^2$ (con $\omega>0$): $\Delta t\,\Delta\omega=\frac12$.
#
# ### Predecí
# Si acortás el pulso a la mitad, ¿qué le pasa a su espectro? ¿Cambia la frecuencia central?

# %%
def espectro(f, dt):
    """f̃(ω) = ∫ f(t) e^{iωt} dt para muestras f(t_n), con t_n = (n − N/2) dt. Devuelve (ω, f̃) con ω creciente."""
    N = len(f)
    F = np.fft.ifft(np.fft.ifftshift(f)) * N * dt          # Σ f_n e^{+iω t_n} dt
    w = 2 * np.pi * np.fft.fftfreq(N, dt)
    return np.fft.fftshift(w), np.fft.fftshift(F)

def ancho_rms(x, peso):
    m = np.sum(x * peso) / np.sum(peso)
    return np.sqrt(np.sum((x - m)**2 * peso) / np.sum(peso))

Nt, Tt = 2**14, 400.0
tp = (np.arange(Nt) - Nt // 2) * (Tt / Nt); dtp = Tt / Nt

def pulso(tau_p, w0):
    f = np.exp(-tp**2 / (2 * tau_p**2)) * np.cos(w0 * tp)
    w, F = espectro(f, dtp)
    Dt = ancho_rms(tp, np.exp(-tp**2 / tau_p**2))
    pos = w > 0
    Dw = ancho_rms(w[pos], np.abs(F[pos])**2)
    return f, w, F, Dt, Dw

def mostrar_pulso(tau_p=3.0, w0=3.0):
    f, w, F, Dt, Dw = pulso(tau_p, w0)
    Fa = np.sqrt(2 * np.pi) * tau_p / 2 * (np.exp(-(w - w0)**2 * tau_p**2 / 2) + np.exp(-(w + w0)**2 * tau_p**2 / 2))
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.8))
    axs[0].plot(tp, f, color=COLORES[0], lw=1.2); axs[0].plot(tp, np.exp(-tp**2 / (2 * tau_p**2)), "k--", lw=1)
    axs[0].set(xlim=(-15, 15), xlabel="t", ylabel="f(t)", title=f"pulso: Δt = {Dt:.3f}")
    axs[1].plot(w, np.abs(F)**2, color=COLORES[1], label="FFT")
    axs[1].plot(w, Fa**2, "k--", lw=1, label="analítico")
    axs[1].set(xlim=(0, 10), xlabel="ω", ylabel="|f̃(ω)|²", title=f"espectro: Δω = {Dw:.3f},  Δt Δω = {Dt * Dw:.4f}")
    axs[1].legend(fontsize=9); plt.show()

interactuar(mostrar_pulso, tau_p=deslizador("τ", 3.0, 0.4, 6.0, 0.1), w0=deslizador("ω₀", 3.0, 1.0, 6.0, 0.25))

f, w, F, Dt, Dw = pulso(3.0, 3.0)
Fa = np.sqrt(2 * np.pi) * 3.0 / 2 * (np.exp(-(w - 3.0)**2 * 9 / 2) + np.exp(-(w + 3.0)**2 * 9 / 2))
verificar("espectro numérico = fórmula de las notas", np.max(np.abs(F - Fa)), 0.0, tol=1e-10)
verificar("Parseval: ∫f² dt = ∫dω/2π |f̃|²", np.sum(np.abs(F)**2) * (w[1] - w[0]) / (2 * np.pi), np.sum(f**2) * dtp, tol=1e-10)
verificar("Δt Δω = 1/2 para el pulso gaussiano", Dt * Dw, 0.5, tol=1e-6)
Dw_mitad = pulso(1.5, 3.0)[4]
verificar("con la mitad de duración, el doble de ancho de banda", Dw_mitad / Dw, 2.0, tol=1e-6)

# %% [markdown]
# **Dos pulsos que se cruzan.** En una dimensión, $E_y=f(x-ct)+g(x+ct)$ y $B_z=f(x-ct)-g(x+ct)$ (Clase 18). Tomamos dos pulsos gaussianos iguales, uno hacia cada lado, con el mismo signo de $E_y$.
#
# ### Predecí
# Cuando los pulsos se superponen, $E_y$ se duplica. ¿Se cuadruplica la energía en esa región? ¿Dónde está la energía en ese instante?

# %%
xc = np.linspace(-30, 30, 6001); dxc = xc[1] - xc[0]
fp = lambda u: np.exp(-(u + 10)**2)          # va hacia +x; en t = 0 está en x = −10
gp = lambda v: np.exp(-(v - 10)**2)          # va hacia −x; en t = 0 está en x = +10
tc = np.linspace(0, 20, 201)
UE, UB = [], []
for t in tc:
    E = fp(xc - t) + gp(xc + t); B = fp(xc - t) - gp(xc + t)
    UE.append(np.sum(E**2) * dxc / (8 * np.pi)); UB.append(np.sum(B**2) * dxc / (8 * np.pi))
UE, UB = np.array(UE), np.array(UB)

fig, axs = plt.subplots(1, 2, figsize=(12, 3.8))
for t, col in zip((6.0, 9.0, 10.0), RAMPA[1:]):
    axs[0].plot(xc, fp(xc - t) + gp(xc + t), color=col, label=f"E_y, t = {t:g}")
    axs[0].plot(xc, fp(xc - t) - gp(xc + t), "--", color=col, lw=1.2, label=f"B_z, t = {t:g}")
axs[0].set(xlim=(-8, 8), xlabel="x", title="E_y (llena) y B_z (a trazos)"); axs[0].legend(fontsize=8, ncol=2)
axs[1].plot(tc, UE, color=COLORES[0], label="∫E²/8π"); axs[1].plot(tc, UB, color=COLORES[1], label="∫B²/8π")
axs[1].plot(tc, UE + UB, "k", label="total")
axs[1].set(xlabel="t", ylabel="energía por unidad de área", title="la energía total no cambia"); axs[1].legend(fontsize=9)
plt.show()

U_notas = (np.sum(fp(xc)**2) + np.sum(gp(xc)**2)) * dxc / (4 * np.pi)
verificar("energía total constante (máxima desviación relativa)", np.max(np.abs(UE + UB - U_notas)) / U_notas, 0.0, tol=1e-12)
verificar("en la superposición toda la energía es eléctrica: ∫B² = 0 en t = 10", UB[100], 0.0, tol=1e-12)

# %% [markdown]
# ### ¿Qué pasó?
# Al acortar el pulso, el espectro se ensancha en la misma proporción y la frecuencia central no se mueve: $\Delta t\,\Delta\omega$ no depende de $\tau$. Un pulso de pocos ciclos no tiene un color definido. Parseval dice que la energía del pulso es la suma de las energías de sus frecuencias: por eso $|\tilde f(\omega)|^2$ es la distribución de la energía en frecuencia, lo que separa un espectrómetro.
#
# Con los dos pulsos, la energía total no cambia, aunque $E_y$ se duplica: en ese instante $B_z=0$, y la energía, que era mitad eléctrica y mitad magnética, está toda en el campo eléctrico. El término cruzado $2fg$ suma en $E^2$ y resta en $B^2$: las dos partes no interfieren en la energía (sección 1.3 de las notas).
#
# ## Experimento 3 — De las frecuencias al tiempo: $e^{ik\mathcal R}/\mathcal R$
#
# Cada frecuencia de una fuente produce, a distancia $\mathcal R$, la amplitud $\tilde{\mathcal S}(\omega)\,e^{ik\mathcal R}/\mathcal R$ con $k=\omega/c$ (la función de Green de Helmholtz de las notas). Sumamos las frecuencias de un destello gaussiano de área 1, $\tilde{\mathcal S}(\omega)=e^{-\omega^2\tau^2/2}$, con $\int\frac{d\omega}{2\pi}\,e^{-i\omega t}$. Las notas predicen $\mathcal S(t-\mathcal \mathcal R/c)/\mathcal \mathcal R$.
#
# ### Predecí
# Si cada frecuencia se multiplica por $e^{ik\mathcal R}/\mathcal R$, ¿qué le pasa al pulso? ¿Y con $e^{-ik\mathcal R}/\mathcal R$, que también resuelve la ecuación de Helmholtz?

# %%
R0, tau_h = 3.0, 0.3
om = np.linspace(-40, 40, 4001)
s_om = np.exp(-om**2 * tau_h**2 / 2)
th = np.linspace(-6, 8, 701)

def al_tiempo(G):
    """∫ dω/2π s̃(ω) G(ω) e^{−iωt}."""
    return np.real(trapezoid(s_om * G * np.exp(-1j * np.outer(th, om)), om, axis=1)) / (2 * np.pi)

ret = al_tiempo(np.exp(1j * om * R0) / R0)
adv = al_tiempo(np.exp(-1j * om * R0) / R0)

fig, axs = plt.subplots(1, 2, figsize=(12, 3.8))
for wn, col in zip((1.0, 2.5, 4.0), RAMPA[1:]):
    axs[0].plot(th, np.exp(-wn**2 * tau_h**2 / 2) * np.cos(wn * (th - R0)) / R0, color=col, lw=1.2, label=f"ω = {wn:g}")
axs[0].axvline(R0, color="0.5", lw=0.8)
axs[0].set(xlabel="t", title=r"algunos modos: todos en fase en $t=\mathcal{R}/c$"); axs[0].legend(fontsize=9)
axs[1].plot(th, ret, color=COLORES[0], label=r"con $e^{ik\mathcal{R}}/\mathcal{R}$: retardado")
axs[1].plot(th, adv, color=COLORES[1], label=r"con $e^{-ik\mathcal{R}}/\mathcal{R}$: adelantado")
axs[1].plot(th, destello(th, tau_h), color="0.6", lw=1, label=r"la fuente, $\mathcal{S}(t)$")
axs[1].set(xlabel="t", title=rf"suma de todos los modos, $\mathcal{{R}}={R0:g}$"); axs[1].legend(fontsize=9)
plt.show()

verificar("suma de modos con e^{ik𝓡}/𝓡 = 𝒮(t − 𝓡/c)/𝓡", np.max(np.abs(ret - destello(th - R0, tau_h) / R0)), 0.0, tol=1e-10)
verificar("suma de modos con e^{−ik𝓡}/𝓡 = 𝒮(t + 𝓡/c)/𝓡", np.max(np.abs(adv - destello(th + R0, tau_h) / R0)), 0.0, tol=1e-10)

# Helmholtz: laplaciano cartesiano (diferencias finitas) de G = e^{ik𝓡}/𝓡, y flujo de ∇G por una esfera chica
k, h = 2.0, 1e-3
G = lambda x, y, z: np.exp(1j * k * np.sqrt(x**2 + y**2 + z**2)) / np.sqrt(x**2 + y**2 + z**2)
p = np.array([0.6, -0.3, 0.5])
lap = sum(G(*(p + h * e)) + G(*(p - h * e)) - 2 * G(*p) for e in np.eye(3)) / h**2
verificar("(∇² + k²) e^{ik𝓡}/𝓡 = 0 fuera del origen", abs(lap + k**2 * G(*p)) / abs(k**2 * G(*p)), 0.0, tol=1e-5)
r, hr = 1e-3, 1e-6
flujo = 4 * np.pi * r**2 * (G(r + hr, 0, 0) - G(r - hr, 0, 0)) / (2 * hr)
verificar("flujo de ∇G por una esfera de radio 10⁻³ = −4π (la delta)", flujo.real, -4 * np.pi, tol=1e-4)

# %% [markdown]
# ### ¿Qué pasó?
# Cada modo, a distancia $\mathcal R$, está corrido en fase en $k\mathcal R=\omega \mathcal R/c$: eso es un retardo $\mathcal R/c$ **igual para todas las frecuencias**, así que todo el pulso llega $\mathcal R/c$ más tarde, con la misma forma y dividido por $\mathcal R$. Es la cuenta de la sección 3.1 de las notas: $e^{ik\mathcal R}$ es un retardo. Con $e^{-ik\mathcal R}$ el pulso aparece $\mathcal R/c$ **antes** de que la fuente lo emita: es matemáticamente correcto, pero describe una onda que viene desde el infinito y se concentra en la fuente; la descartamos porque en ella el efecto precede a la causa. Al final, el laplaciano numérico confirma que $e^{ik\mathcal R}/\mathcal R$ cumple la ecuación de Helmholtz fuera del origen, y el flujo de su gradiente, $-4\pi$, es la delta.
#
# ## Experimento 4 — ¿Cuándo vale lo cuasiestático?
#
# Dos cargas $\pm q(t)$, con $q=\cos\omega t$, en $z=\pm d/2$ con $d=1$ (un hilo entre ellas lleva la corriente $\dot q$). Comparamos el potencial retardado,
# $$\phi=\frac{q(t-R_+/c)}{R_+}-\frac{q(t-R_-/c)}{R_-},$$
# con el de Coulomb instantáneo, $q(t)\left(\frac1{R_+}-\frac1{R_-}\right)$. Las notas (sección 6) dicen que la corrección de orden $1/c$ se anula porque la carga total se conserva, así que el error relativo crece como $\omega^2$.
#
# ### Predecí
# Si duplicás la frecuencia, ¿el error de la aproximación cuasiestática se duplica o se cuadruplica?

# %%
def potenciales(om_, robs, d=1.0, n=2001):
    """φ retardado e instantáneo de ±cos ωt en z = ±d/2, en el punto robs, durante un período."""
    tq = np.linspace(0, 2 * np.pi / om_, n)
    Rm = np.linalg.norm(robs - np.array([0, 0, d / 2])); Rn = np.linalg.norm(robs + np.array([0, 0, d / 2]))
    ret = np.cos(om_ * (tq - Rm)) / Rm - np.cos(om_ * (tq - Rn)) / Rn
    ins = np.cos(om_ * tq) * (1 / Rm - 1 / Rn)
    return tq, ret, ins, Rm, Rn

def error_cuasiestatico(om_, robs):
    tq, ret, ins, _, _ = potenciales(om_, robs)
    return np.max(np.abs(ret - ins)) / np.max(np.abs(ins))

oms = np.logspace(-3, 0.5, 30)

def mostrar_cuasi(log_omega=-1.0, r=2.0):
    om_ = 10**log_omega; robs = r * np.array([np.sin(1.0), 0, np.cos(1.0)])
    tq, ret, ins, _, _ = potenciales(om_, robs)
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.8))
    axs[0].plot(om_ * tq, ret, color=COLORES[0], label="retardado"); axs[0].plot(om_ * tq, ins, "--", color=COLORES[1], label="Coulomb instantáneo")
    axs[0].set(xlabel="ωt", ylabel="φ", title=f"ω d/c = {om_:.3g},  r = {r:g}"); axs[0].legend(fontsize=9)
    err = [error_cuasiestatico(o, robs) for o in oms]
    axs[1].loglog(oms, err, color=COLORES[0], label="error relativo")
    axs[1].loglog(oms, err[0] * (oms / oms[0])**2, "k--", lw=1, label="∝ ω²")
    axs[1].plot(om_, error_cuasiestatico(om_, robs), "o", color=COLORES[1])
    axs[1].set(xlabel="ω d/c", ylim=(1e-7, 10)); axs[1].legend(fontsize=9)
    plt.show()

interactuar(mostrar_cuasi, log_omega=deslizador("log₁₀(ωd/c)", -1.0, -3.0, 0.5, 0.25), r=deslizador("r/d", 2.0, 1.0, 30.0, 0.5))

robs = 2.0 * np.array([np.sin(1.0), 0, np.cos(1.0)])
pend = np.polyfit(np.log(oms[:12]), np.log([error_cuasiestatico(o, robs) for o in oms[:12]]), 1)[0]
verificar("el error crece como ω² (pendiente en escala logarítmica)", pend, 2.0, tol=2e-3)
om_ = 0.01
tq, ret, ins, Rm, Rn = potenciales(om_, robs)
q2, q3 = -om_**2 * np.cos(om_ * tq), om_**3 * np.sin(om_ * tq)        # q̈ y q⃛
desarrollo = q2 / 2 * (Rm - Rn) - q3 / 6 * (Rm**2 - Rn**2)            # órdenes 1/c² y 1/c³ (el de 1/c es cero)
verificar("φ_ret − φ_inst = (q̈/2c²)(R₊ − R₋) − (q⃛/6c³)(R₊² − R₋²) + …", np.max(np.abs(ret - ins - desarrollo)) / np.max(np.abs(desarrollo)), 0.0, tol=1e-3)

# %% [markdown]
# ### ¿Qué pasó?
# El error relativo crece como $\omega^2$, no como $\omega$: el término de orden $1/c$, $-\frac1c\frac{dQ}{dt}$, es cero porque la carga total ($q-q=0$) no cambia. Con $\omega d/c=10^{-2}$ el error es de una parte en $10^4$. Si alejás el observador (deslizador $r$) hasta $r\gtrsim c/\omega$, el error deja de ser chico aunque el sistema sea diminuto: el retardo entre el sistema y el observador ya no se puede despreciar. Es la **zona de radiación**, el tema de la Clase 22.
#
# ## Explorá
#
# 1. **Guía 8, P7.** `potencial_hilo(I, s, t)`, en la celda de abajo, calcula $A_z$ de un hilo infinito sobre el eje $z$ con una corriente $I(t)$ que se enciende en $t=0$, sumando las contribuciones retardadas $\frac1c I(t-\mathcal R/c)/\mathcal R\,dz'$ de los puntos del hilo cuya señal ya llegó. Usala con $I(t)=I_0\,\Theta(t)$ y comparala con tu $A_z$; derivá numéricamente (`np.gradient`) para obtener $B_\varphi=-\partial_sA_z$ y $E_z=-\frac1c\partial_tA_z$, graficalos para varios $t$ (inciso b) y mirá a qué tienden para $t\to\infty$ (inciso c).
# 2. **Guía 8, P8.** `potencial_lamina(K, x, t, x0)` da el $A_y$ de una lámina en $x=x_0$ con corriente superficial $K(t)\hat{\mathbf y}$ (sección 4 de las notas). Superponé dos láminas con corrientes opuestas (en el problema son los planos $z=\pm d$; acá usamos $x$), calculá los campos entre ellas y afuera, y con ellos el flujo de energía.
# 3. **Guía 8, P6.** `paquete_1d(E0, k, x, t)` arma en una dimensión $E_y=\operatorname{Re}\int\frac{dk}{2\pi}E_0(k)\,e^{i(kx-c|k|t)}$ y $B_z$ (cada modo con $\mathbf B=\hat{\mathbf k}\times\mathbf E$). Elegí un $E_0(k)$ con partes en $k>0$ y en $k<0$, mirá cómo cambian por separado $\int E_y^2dx$ e $\int B_z^2dx$, y comparalos con la energía total y con $\frac{1}{8\pi}\int\frac{dk}{2\pi}|E_0(k)|^2$, la versión unidimensional del resultado del problema.
# 4. **Duración y ancho de banda.** Con `espectro` y `ancho_rms`, repetí el Experimento 2 con otra envolvente, por ejemplo $1/\cosh(t/\tau)$: ¿el producto $\Delta t\,\Delta\omega$ depende de $\tau$? ¿Es mayor o menor que $\frac12$?

# %%
def potencial_hilo(I, s, t, dz=1e-3):
    """A_z(s, t) de un hilo infinito sobre el eje z con corriente I(t), nula para t < 0 (c = 1).

    Suma (1/c) I(t − R/c)/R dz' sobre los puntos cuya señal ya llegó (R = √(s² + z'²) < t).
    """
    A = []
    for tn in np.atleast_1d(t):
        if tn <= s:
            A.append(0.0); continue
        z = np.linspace(-np.sqrt(tn**2 - s**2), np.sqrt(tn**2 - s**2), int(2 * np.sqrt(tn**2 - s**2) / dz) + 2)
        R = np.sqrt(s**2 + z**2)
        A.append(trapezoid(I(tn - R) / R, z))
    return np.array(A)

def potencial_lamina(K, x, t, x0=0.0, dt=1e-3):
    """A_y(x, t) de una lámina en x = x0 con corriente superficial K(t) ŷ, nula para t < 0 (c = 1): 2π ∫ K hasta t − |x − x0|."""
    A = []
    for tn in np.atleast_1d(t):
        tr = tn - abs(x - x0)
        if tr <= 0:
            A.append(0.0); continue
        tt_ = np.linspace(0, tr, int(tr / dt) + 2)
        A.append(2 * np.pi * trapezoid(K(tt_), tt_))
    return np.array(A)

def paquete_1d(E0, k, x, t):
    """E_y y B_z de Re ∫ dk/2π E0(k) e^{i(kx − c|k|t)} (suma sobre la grilla k), con B_z = sgn(k) E_y en cada modo."""
    dk = k[1] - k[0]
    fase = np.exp(1j * (np.outer(x, k) - np.abs(k) * t))
    Ey = np.real(fase @ E0) * dk / (2 * np.pi)
    Bz = np.real(fase @ (np.sign(k) * E0)) * dk / (2 * np.pi)
    return Ey, Bz

# Las herramientas, comprobadas con el destello del Experimento 1 (corrido para que empiece después de t = 0)
I_destello = lambda t: destello(t - 1.0, tau)
tchk = np.array([1.0 + d + 0.5, 1.0 + d + 2.0])
verificar("potencial_hilo con un destello = la recta del Experimento 1", potencial_hilo(I_destello, d, tchk)[1],
          respuesta_recta(d, tchk - 1.0, tau)[1], tol=1e-4)
verificar("potencial_lamina con un destello = el plano del Experimento 1", potencial_lamina(I_destello, d, tchk)[1],
          respuesta_plano(d, tchk - 1.0, tau)[1], tol=1e-4)
kk = np.linspace(-6, 6, 1201); E0k = np.where(kk > 0, np.exp(-(kk - 3)**2), 0) * np.exp(10j * kk)
xx = np.linspace(-40, 40, 4001)
Ey, Bz = paquete_1d(E0k, kk, xx, 0.0)
verificar("paquete_1d: un paquete que va hacia +x tiene B_z = E_y", np.max(np.abs(Bz - Ey)), 0.0, tol=1e-6)
