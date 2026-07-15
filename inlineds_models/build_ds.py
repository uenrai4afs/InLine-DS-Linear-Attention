"""
build_ds.py — unified model builder for InLine^D-S.

Imports inline_deit_ds directly so timm's registry always finds
VisionTransformerDS (our v2 rewrite) under 'inline_deit_ds_tiny' etc.
No aliases to original InLine models.
"""
import torch
from timm.models import create_model

# Force-import our v2 model file so its @register_model decorators fire.
# This must happen before create_model() is called.
from . import inline_deit_ds   # noqa: F401  registers inline_deit_ds_tiny/small/base
from . import inline_swin_ds   # noqa: F401
from . import inline_pvt_ds    # noqa: F401
from . import inline_cswin_ds  # noqa: F401


def build_model(config):
    """
    Build a model from config.MODEL.TYPE using timm's registry.
    The correct model class is guaranteed because the imports above
    registered our v2 implementations before this function runs.
    """
    model_type = config.MODEL.TYPE

    model = create_model(
        model_type,
        pretrained=False,
        num_classes=config.MODEL.NUM_CLASSES,
        drop_rate=config.MODEL.DROP_RATE,
        drop_path_rate=config.MODEL.DROP_PATH_RATE,
        img_size=config.DATA.IMG_SIZE,
    )

    return model
