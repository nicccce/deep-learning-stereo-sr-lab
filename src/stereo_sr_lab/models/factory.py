from .swin_mono_sr import SwinMonoSRNet
from .stereo_sr import StereoSRNet
from .swin_stereo_sr import SwinStereoSRNet


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
    if name == "swin_stereo_sr":
        return SwinStereoSRNet(
            scale=cfg.get("scale", config.get("data", {}).get("scale", 2)),
            embed_dim=cfg.get("embed_dim", 60),
            depths=cfg.get("depths", [6, 6, 6, 6, 6, 6]),
            num_heads=cfg.get("num_heads", [6, 6, 6, 6, 6, 6]),
            window_size=cfg.get("window_size", 8),
            mlp_ratio=cfg.get("mlp_ratio", 2.0),
            max_disp=cfg.get("max_disp", 0),
            drop_path_rate=cfg.get("drop_path_rate", 0.1),
            resi_connection=cfg.get("resi_connection", "1conv"),
            use_checkpoint=cfg.get("use_checkpoint", False),
            img_size=cfg.get("img_size", 48),
        )
    if name == "swin_mono_sr":
        return SwinMonoSRNet(
            scale=cfg.get("scale", config.get("data", {}).get("scale", 2)),
            embed_dim=cfg.get("embed_dim", 60),
            depths=cfg.get("depths", [6, 6, 6, 6, 6, 6]),
            num_heads=cfg.get("num_heads", [6, 6, 6, 6, 6, 6]),
            window_size=cfg.get("window_size", 8),
            mlp_ratio=cfg.get("mlp_ratio", 2.0),
            max_disp=cfg.get("max_disp", 0),
            drop_path_rate=cfg.get("drop_path_rate", 0.1),
            resi_connection=cfg.get("resi_connection", "1conv"),
            use_checkpoint=cfg.get("use_checkpoint", False),
            img_size=cfg.get("img_size", 48),
            fusion_context=cfg.get("fusion_context", "zero"),
        )
    raise ValueError(f"Unsupported model: {name}")


def count_parameters(model) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)

