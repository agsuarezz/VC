import os
import sys
import urllib.request

P4_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.join(P4_ROOT, "RT-DETR", "rtdetrv2_pytorch")
if not os.path.isfile(os.path.join(REPO, "src", "core", "__init__.py")):
    raise ModuleNotFoundError(
        "Falta el código de RT-DETR. Desde P4 ejecuta: "
        "git clone --depth 1 https://github.com/lyuwenyu/RT-DETR.git RT-DETR"
    )
repo_abs = os.path.abspath(REPO)
if repo_abs not in sys.path:
    sys.path.insert(0, repo_abs)
if "src" in sys.modules:
    mod_file = getattr(sys.modules["src"], "__file__", None) or ""
    if repo_abs.replace("\\", "/").lower() not in mod_file.replace("\\", "/").lower():
        sys.modules.pop("src", None)

import cv2
import numpy as np
import torch
from PIL import Image
from src.core import YAMLConfig
from src.core.yaml_utils import load_config      # next to your YAMLConfig import

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Para usar el modelo, necesitamos descargar los pesos oficiales desde el Repositorio.
# Actualmente, existen 8 modelos entrenados, divididos en dos familias y cuatro tamaños:
# La familia RT-DETRv1 y la V2. 
# Cada familia tiene cuatro tamaños, de menor a mayor, S,M,L y X.

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


def load_rtdetr(family="v2", size="s", device=DEVICE):

    family = family.lower()
    size = size.lower()
    print(f"Cargando RT-DETR familia {family!r}, tamaño {size!r} en {device}")
    
    size = SIZE_ALIASES[size] if size in SIZE_ALIASES else size
    
    if family not in MODELS or size not in MODELS[family]:
        raise ValueError(f"Modelo desconocido: {family}/{size}. Inserte familia v2 o v1 y tamaño s,m,l,x o r18,r34,r50m,r50,r101")
    
    # Sacamos el diccionario el archivo de configuración, y lo necesario para descargar los pesos.
    
    cfg_name, tag, wfile = MODELS[family][size]

    WEIGHTS = os.path.join(P4_ROOT, "weights", wfile)
    if not os.path.exists(WEIGHTS):
        print(f'No se detectaron los pesos del modelo {family}/{size}. Se estan descargando y guardando en {WEIGHTS}')
        os.makedirs(os.path.join(P4_ROOT, "weights"), exist_ok=True)
        urllib.request.urlretrieve(
            f"https://github.com/lyuwenyu/storage/releases/download/{tag}/{wfile}", WEIGHTS)

    
    # Teniendo el archivo de configuración y los pesos descargados, cargamos el modelo y el postprocesador.
    load_config.__defaults__[0].clear()
    # El checkpoint completo ya incluye el backbone.
    cfg = YAMLConfig(f"{REPO}/configs/{cfg_name}", PResNet={"pretrained": False})
   
    ckpt = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    cfg.model.load_state_dict(ckpt["ema"]["module"] if "ema" in ckpt else ckpt["model"])
    model = cfg.model.deploy().to(device).eval()
    postprocessor = cfg.postprocessor.deploy().to(device)
    return model, postprocessor
