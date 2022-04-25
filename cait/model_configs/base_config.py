import ml_collections


def get_config(
    model_name: str = "cait_xxs24_224",
    image_size: int = 224,
    patch_size: int = 16,
    projection_dim: int = 192,
    sa_ffn_layers: int = 24,
    ca_ffn_layers: int = 2,
    num_heads: int = 4,
    mlp_ratio=4,
    layer_norm_eps=1e-6,
    init_values: float = 1e-5,
    dropout_rate: float = 0.0,
    drop_path_rate: float = 0.0,
    pre_logits: bool = False,
) -> ml_collections.ConfigDict:
    """Default configuration for CaiT models (cait_xxs24_224).

    Reference:
        https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/cait.py
    """
    config = ml_collections.ConfigDict()
    config.model_name = model_name

    config.image_size = image_size
    config.patch_size = patch_size
    config.num_patches = (config.image_size // config.patch_size) ** 2
    config.num_classes = 1000

    config.initializer_range = 0.02
    config.layer_norm_eps = layer_norm_eps
    config.projection_dim = projection_dim
    config.num_heads = num_heads
    config.sa_ffn_layers = sa_ffn_layers
    config.ca_ffn_layers = ca_ffn_layers
    config.mlp_units = [
        config.projection_dim * mlp_ratio,
        config.projection_dim,
    ]
    config.dropout_rate = dropout_rate
    config.init_values = init_values
    config.drop_path_rate = drop_path_rate
    config.global_pool = "token"
    config.pre_logits = pre_logits

    return config.lock()
