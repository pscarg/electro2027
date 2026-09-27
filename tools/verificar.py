#!/usr/bin/env python3
"""Verifica los notebooks: reglas de estilo, ejecución, chequeos analíticos y enlaces.

Uso (con el entorno de prueba):
    ~/.venvs/electro2027/bin/python tools/verificar.py [01 02 ...] [--imagenes CARPETA]

Para cada notebook:
  * sin salidas guardadas, con badge de Colab, sin "pip install", con secciones
    "Predecí" y "Explorá", nombre ASCII;
  * se ejecuta completo (kernel "electro2027", variable E2027_PRUEBA=1) y se mide el tiempo;
  * ninguna línea de verificación marcada con ✘;
  * los bloques "# herramienta: NOMBRE vN" coinciden entre notebooks;
  * cada \\notebook{...}, explora{...} y demo{...} de clases/*.tex apunta a un notebook existente.
"""
import os
import re
import sys
import time
import base64
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

RAIZ = Path(__file__).resolve().parents[1]
NOTEBOOKS = RAIZ / "notebooks"
CLASES = RAIZ.parent / "clases"
LIMITE_SEGUNDOS = 30


def herramientas(nb):
    """Devuelve {"nombre vN": código} para los bloques marcados."""
    bloques = {}
    for c in nb.cells:
        if c.cell_type != "code":
            continue
        for m in re.finditer(r"# herramienta: (.+?)\n(.*?)# fin herramienta", c.source + "\n", re.S):
            clave = re.sub(r"\s*\(.*\)$", "", m.group(1).strip())
            bloques[clave] = m.group(2).strip()
    return bloques


def estatico(ruta, nb):
    problemas = []
    if not ruta.name.isascii():
        problemas.append("nombre no ASCII")
    if "colab-badge.svg" not in nb.cells[0].source:
        problemas.append("falta el badge de Colab en la primera celda")
    texto = "\n".join(c.source for c in nb.cells)
    if re.search(r"pip install", texto):
        problemas.append("contiene pip install")
    for seccion in ("Predecí", "Explorá"):
        if seccion not in texto:
            problemas.append(f"falta la sección {seccion}")
    for c in nb.cells:
        if c.cell_type == "code" and (c.get("outputs") or c.get("execution_count")):
            problemas.append("hay salidas guardadas")
            break
    return problemas


def ejecutar(ruta, nb, carpeta_imagenes=None):
    os.environ["E2027_PRUEBA"] = "1"
    ep = ExecutePreprocessor(timeout=300, kernel_name="electro2027")
    t0 = time.time()
    ep.preprocess(nb, {"metadata": {"path": str(ruta.parent)}})
    dt = time.time() - t0
    salidas, n_img = [], 0
    for i, c in enumerate(nb.cells):
        for o in c.get("outputs", []):
            if o.get("output_type") == "stream":
                salidas.append(o.get("text", ""))
            if o.get("output_type") == "error":
                salidas.append("ERROR " + o.get("ename", ""))
            datos = o.get("data", {})
            if "text/plain" in datos:
                salidas.append(datos["text/plain"])
            if carpeta_imagenes and "image/png" in datos:
                n_img += 1
                destino = Path(carpeta_imagenes) / f"{ruta.stem}_c{i:02d}_{n_img}.png"
                destino.write_bytes(base64.b64decode(datos["image/png"]))
    texto = "\n".join(salidas)
    return dt, texto


def enlaces_tex(existentes):
    faltan = []
    patron = re.compile(r"\\(?:notebook|begin\{explora\}|begin\{demo\})\{([^}]+)\}")
    for tex in sorted(CLASES.glob("*.tex")):
        for nombre in patron.findall(tex.read_text(encoding="utf-8")):
            if nombre not in existentes:
                faltan.append(f"{tex.name}: {nombre}")
    return faltan


def main():
    args = sys.argv[1:]
    carpeta = None
    if "--imagenes" in args:
        k = args.index("--imagenes")
        carpeta = args[k + 1]
        Path(carpeta).mkdir(parents=True, exist_ok=True)
        del args[k:k + 2]
    rutas = sorted(NOTEBOOKS.glob("[0-9][0-9]_*.ipynb"))
    if args:
        rutas = [r for r in rutas if any(r.name.startswith(p) for p in args)]

    todo_bien = True
    registro = {}
    for ruta in rutas:
        nb = nbformat.read(ruta, as_version=4)
        problemas = estatico(ruta, nb)
        for clave, codigo in herramientas(nb).items():
            if clave in registro and registro[clave][1] != codigo:
                problemas.append(f"herramienta '{clave}' difiere de {registro[clave][0]}")
            registro.setdefault(clave, (ruta.name, codigo))
        try:
            dt, texto = ejecutar(ruta, nb, carpeta)
            ok = texto.count("✔")
            if "✘" in texto:
                problemas.append("hay verificaciones fallidas (✘):\n" +
                                 "\n".join(l for l in texto.splitlines() if "✘" in l))
            if ok == 0:
                problemas.append("no hay ninguna verificación analítica (✔)")
            if dt > LIMITE_SEGUNDOS:
                problemas.append(f"tarda {dt:.0f} s (> {LIMITE_SEGUNDOS} s)")
            resumen = f"{dt:5.1f} s, {ok} verificaciones ✔"
        except Exception as e:  # error de ejecución
            problemas.append(f"falla al ejecutar: {type(e).__name__}: {str(e)[-600:]}")
            resumen = "no ejecutó"
        estado = "OK " if not problemas else "MAL"
        todo_bien &= not problemas
        print(f"[{estado}] {ruta.name}: {resumen}")
        for p in problemas:
            print("      -", p)

    faltan = enlaces_tex({r.stem for r in NOTEBOOKS.glob("*.ipynb")})
    if faltan:
        todo_bien = False
        print("[MAL] enlaces en .tex a notebooks inexistentes:")
        for f in faltan:
            print("      -", f)
    else:
        print("[OK ] enlaces de los .tex a notebooks")
    sys.exit(0 if todo_bien else 1)


if __name__ == "__main__":
    main()
