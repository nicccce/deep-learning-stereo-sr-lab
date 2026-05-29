from .mono_sr import MonoSRNet
from .stereo_sr import StereoSRNet


def create_model(config: dict):
    cfg = config.get("model", config)
    name = cfg.get("name", "stereo_sr").lower()
    if name == "stereo_sr":
        return StereoSRNet(
            scale=cfg.get("scale", config.get("data", {}).get("scale", 2)),
            channels=cfg.get("channels", 48),
            num_feature_blocks=cfg.get("num_feature_blocks", 6),
            num_reconstruct_blocks=cfg.get("num_reconstruct_blocks", 4),
            max_disp=cfg.get("max_disp", 0),
        )
    if name == "mono_sr":
        return MonoSRNet(
            scale=cfg.get("scale", config.get("data", {}).get("scale", 2)),
            channels=cfg.get("channels", 48),
            num_blocks=cfg.get("num_blocks", 8),
        )
    raise ValueError(f"Unsupported model: {name}")


def count_parameters(model) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)

