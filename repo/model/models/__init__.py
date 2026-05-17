from .gan import ConditionalGAN
from .vae import ConditionalVAE
from .transformer import ConditionalTransformer
from .diffusion import ConditionalDiffusion

MODEL_REGISTRY = {
    "gan":         ConditionalGAN,
    "vae":         ConditionalVAE,
    "transformer": ConditionalTransformer,
    "diffusion":   ConditionalDiffusion,
}


def get_model(name: str, config: dict):
    """Instantiate a model by name with given config dict."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](config)
