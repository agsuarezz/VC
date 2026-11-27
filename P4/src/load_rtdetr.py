import importlib
import importlib.util
import os
import shutil
import sys
import urllib.request

import torch

P4_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.join(P4_ROOT, "RT-DETR", "rtdetrv2_pytorch")
if not os.path.isfile(os.path.join(REPO, "src", "core", "__init__.py")):
    raise ModuleNotFoundError(
        "Falta el código de RT-DETR. Desde P4 ejecuta: "
        "git clone --depth 1 https://github.com/lyuwenyu/RT-DETR.git RT-DETR"
    )


def _import_rtdetr_core():
    """Importa RT-DETR bajo un namespace aislado para no colisionar con P4/src."""
    core_mod = sys.modules.get("_rtdetr_src.core")
    if core_mod is not None:
        return core_mod.YAMLConfig, core_mod.load_config

    src_pkg_dir = os.path.join(REPO, "src")
    init_py = os.path.join(src_pkg_dir, "__init__.py")
    spec = importlib.util.spec_from_file_location(
        "_rtdetr_src",
        init_py,
        submodule_search_locations=[src_pkg_dir],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"No se pudo inicializar el paquete RT-DETR desde {src_pkg_dir}")

    pkg = importlib.util.module_from_spec(spec)
    sys.modules["_rtdetr_src"] = pkg
    try:
        spec.loader.exec_module(pkg)
        core_mod = importlib.import_module("_rtdetr_src.core")
    except Exception:
        for key in list(sys.modules):
            if key == "_rtdetr_src" or key.startswith("_rtdetr_src."):
                sys.modules.pop(key, None)
        raise
    return core_mod.YAMLConfig, core_mod.load_config


YAMLConfig, load_config = _import_rtdetr_core()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Para usar el modelo, necesitamos descargar los pesos oficiales desde el repositorio.
# Existen 10 configuraciones entrenadas (5 por familia: v1 y v2), con alias cortos s, m, l, x
# y selección directa por nombre de backbone (r18, r34, r50m, r50, r101).
# (config, tag de la release, fichero de pesos COCO)
MODELS = {
    "v1": {
        "r18":  ("rtdetr/rtdetr_r18vd_6x_coco.yml",    "v0.1", "rtdetr_r18vd_dec3_6x_coco_from_paddle.pth"),
        "r34":  ("rtdetr/rtdetr_r34vd_6x_coco.yml",    "v0.1", "rtdetr_r34vd_dec4_6x_coco_from_paddle.pth"),
        "r50m": ("rtdetr/rtdetr_r50vd_m_6x_coco.yml",  "v0.1", "rtdetr_r50vd_m_6x_coco_from_paddle.pth"),
        "r50":  ("rtdetr/rtdetr_r50vd_6x_coco.yml",    "v0.1", "rtdetr_r50vd_6x_coco_from_paddle.pth"),
        "r101": ("rtdetr/rtdetr_r101vd_6x_coco.yml",   "v0.1", "rtdetr_r101vd_6x_coco_from_paddle.pth"),
    },
    "v2": {
        "r18":  ("rtdetrv2/rtdetrv2_r18vd_120e_coco.yml",  "v0.2", "rtdetrv2_r18vd_120e_coco_rerun_48.1.pth"),
        "r34":  ("rtdetrv2/rtdetrv2_r34vd_120e_coco.yml",  "v0.1", "rtdetrv2_r34vd_120e_coco_ema.pth"),
        "r50m": ("rtdetrv2/rtdetrv2_r50vd_m_7x_coco.yml",  "v0.1", "rtdetrv2_r50vd_m_7x_coco_ema.pth"),
        "r50":  ("rtdetrv2/rtdetrv2_r50vd_6x_coco.yml",    "v0.1", "rtdetrv2_r50vd_6x_coco_ema.pth"),
        "r101": ("rtdetrv2/rtdetrv2_r101vd_6x_coco.yml",   "v0.1", "rtdetrv2_r101vd_6x_coco_from_paddle.pth"),
    },
}
SIZE_ALIASES = {"s": "r18", "m": "r50m", "l": "r50", "x": "r101"}   # nombres cortos


def _download_weights(url: str, dest_path: str) -> None:
    """Descarga pesos de forma atómica evitando ficheros truncados tras cortes de red."""
    weights_dir = os.path.dirname(dest_path)
    os.makedirs(weights_dir, exist_ok=True)
    tmp_path = f"{dest_path}.part"
    try:
        with urllib.request.urlopen(url, timeout=60) as resp, open(tmp_path, "wb") as out_f:
            shutil.copyfileobj(resp, out_f)
        if os.path.getsize(tmp_path) == 0:
            raise IOError(f"El archivo descargado desde {url} está vacío.")
        os.replace(tmp_path, dest_path)
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise


def load_rtdetr(family="v2", size="s", device=DEVICE):
    if not isinstance(family, str) or not isinstance(size, str):
        raise ValueError("family y size deben ser cadenas de texto (ej. family='v2', size='s').")

    family = family.lower()
    size = size.lower()
    print(f"Cargando RT-DETR familia {family!r}, tamaño {size!r} en {device}")

    size = SIZE_ALIASES.get(size, size)

    if family not in MODELS or size not in MODELS[family]:
        raise ValueError(
            f"Modelo desconocido: {family}/{size}. Inserte familia v2 o v1 y tamaño s,m,l,x o r18,r34,r50m,r50,r101"
        )

    cfg_name, tag, wfile = MODELS[family][size]
    weights_path = os.path.join(P4_ROOT, "weights", wfile)
    if not os.path.isfile(weights_path) or os.path.getsize(weights_path) == 0:
        print(f"No se detectaron los pesos del modelo {family}/{size}. Descargando en {weights_path}...")
        url = f"https://github.com/lyuwenyu/storage/releases/download/{tag}/{wfile}"
        _download_weights(url, weights_path)

    # RT-DETR usa un diccionario mutable por defecto en load_config; limpiarlo evita arrastrar claves previas.
    if load_config.__defaults__ and isinstance(load_config.__defaults__[0], dict):
        load_config.__defaults__[0].clear()

    cfg_path = os.path.join(REPO, "configs", cfg_name)
    cfg = YAMLConfig(cfg_path, PResNet={"pretrained": False})

    ckpt = torch.load(weights_path, map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict) and isinstance(ckpt.get("ema"), dict) and "module" in ckpt["ema"]:
        state_dict = ckpt["ema"]["module"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        raise ValueError(f"Formato de checkpoint no reconocido en {weights_path}")

    cfg.model.load_state_dict(state_dict)
    model = cfg.model.deploy().to(device).eval()
    postprocessor = cfg.postprocessor.deploy().to(device)
    return model, postprocessor
