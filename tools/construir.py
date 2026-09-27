#!/usr/bin/env python3
"""Construye los notebooks de Colab a partir de las fuentes en formato "percent".

Uso:
    python tools/construir.py            # todas las fuentes
    python tools/construir.py 01 02      # solo las que empiezan con 01 o 02

Formato de las fuentes (colab/fuentes/NN_tema.py):
    # %% [markdown]
    # # Título
    # Texto en markdown (cada línea comentada con "# ").
    # %%
    codigo_python()

El badge "Abrir en Colab" se agrega automáticamente como primera celda, con la
URL base tomada de \\colabbase en clases/electro2027.sty (única fuente de verdad).
También se regenera la tabla de README.md.
"""
import re
import sys
from pathlib import Path

import nbformat

RAIZ = Path(__file__).resolve().parents[1]          # .../Electro 2027/colab
FUENTES = RAIZ / "fuentes"
NOTEBOOKS = RAIZ / "notebooks"
STY = RAIZ.parent / "clases" / "electro2027.sty"
BADGE = "https://colab.research.google.com/assets/colab-badge.svg"


def colab_base():
    m = re.search(r"\\newcommand\{\\colabbase\}\{([^}]*)\}", STY.read_text(encoding="utf-8"))
    if not m:
        sys.exit("No encontré \\colabbase en electro2027.sty")
    return m.group(1).strip()


def celdas(texto):
    """Divide la fuente en celdas (tipo, contenido)."""
    bloques, tipo, actual = [], None, []
    for linea in texto.splitlines():
        if linea.startswith("# %%"):
            if tipo is not None:
                bloques.append((tipo, actual))
            tipo = "markdown" if "[markdown]" in linea else "code"
            actual = []
        elif tipo is not None:
            actual.append(linea)
    if tipo is not None:
        bloques.append((tipo, actual))

    resultado = []
    for tipo, lineas in bloques:
        if tipo == "markdown":
            lineas = [l[2:] if l.startswith("# ") else l[1:] if l.startswith("#") else l for l in lineas]
        contenido = "\n".join(lineas).strip("\n")
        if contenido.strip():
            resultado.append((tipo, contenido))
    return resultado


def construir(fuente, base):
    nombre = fuente.stem
    url = f"{base}/{nombre}.ipynb"
    nb = nbformat.v4.new_notebook()
    nb.metadata = {
        "colab": {"provenance": [], "toc_visible": True},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    nb.cells.append(nbformat.v4.new_markdown_cell(f"[![Abrir en Colab]({BADGE})]({url})"))
    for tipo, contenido in celdas(fuente.read_text(encoding="utf-8")):
        if tipo == "markdown":
            nb.cells.append(nbformat.v4.new_markdown_cell(contenido))
        else:
            nb.cells.append(nbformat.v4.new_code_cell(contenido))
    for i, c in enumerate(nb.cells):          # ids estables: diffs limpios en git
        c["id"] = f"{nombre[:2]}c{i:03d}"
    NOTEBOOKS.mkdir(exist_ok=True)
    destino = NOTEBOOKS / f"{nombre}.ipynb"
    nbformat.write(nb, destino)
    return destino


def titulo(fuente):
    for tipo, contenido in celdas(fuente.read_text(encoding="utf-8")):
        if tipo == "markdown":
            for l in contenido.splitlines():
                if l.startswith("# "):
                    return l[2:].strip()
    return fuente.stem


def readme(base):
    filas = []
    for f in sorted(FUENTES.glob("[0-9][0-9]_*.py")):
        url = f"{base}/{f.stem}.ipynb"
        filas.append(f"| {f.stem[:2]} | {titulo(f)} | [![Abrir en Colab]({BADGE})]({url}) |")
    texto = (RAIZ / "README.md").read_text(encoding="utf-8")
    ini, fin = "<!-- TABLA-INICIO -->", "<!-- TABLA-FIN -->"
    tabla = "| Clase | Notebook | Abrir |\n|---|---|---|\n" + "\n".join(filas)
    texto = re.sub(f"{ini}.*{fin}", f"{ini}\n{tabla}\n{fin}", texto, flags=re.S)
    (RAIZ / "README.md").write_text(texto, encoding="utf-8")


def main():
    base = colab_base()
    filtros = sys.argv[1:]
    fuentes = sorted(FUENTES.glob("[0-9][0-9]_*.py"))
    if filtros:
        fuentes = [f for f in fuentes if any(f.name.startswith(p) for p in filtros)]
    for f in fuentes:
        print("construido:", construir(f, base).relative_to(RAIZ))
    readme(base)


if __name__ == "__main__":
    main()
