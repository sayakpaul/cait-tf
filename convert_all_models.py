import os

# These configs are from:
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/cait.py#L354-#L421
MODEL_CONFIGS = {
    "cait_xxs24_224": {
        "dim": 192,
        "sa_ffn_layers": 24,
        "num_heads": 4,
        "image_size": 224,
        "init_values": 1e-5,
    },
    "cait_xxs24_384": {
        "dim": 192,
        "sa_ffn_layers": 24,
        "num_heads": 4,
        "image_size": 384,
        "init_values": 1e-5,
    },
    "cait_xxs36_224": {
        "dim": 192,
        "sa_ffn_layers": 36,
        "num_heads": 4,
        "image_size": 224,
        "init_values": 1e-5,
    },
    "cait_xxs36_384": {
        "dim": 192,
        "sa_ffn_layers": 36,
        "num_heads": 4,
        "image_size": 384,
        "init_values": 1e-5,
    },
    "cait_xs24_384": {
        "dim": 288,
        "sa_ffn_layers": 24,
        "num_heads": 6,
        "image_size": 384,
        "init_values": 1e-5,
    },
    "cait_s24_224": {
        "dim": 384,
        "sa_ffn_layers": 24,
        "num_heads": 8,
        "image_size": 224,
        "init_values": 1e-5,
    },
    "cait_s24_384": {
        "dim": 384,
        "sa_ffn_layers": 24,
        "num_heads": 8,
        "image_size": 384,
        "init_values": 1e-5,
    },
    "cait_s36_384": {
        "dim": 384,
        "sa_ffn_layers": 36,
        "num_heads": 8,
        "image_size": 384,
        "init_values": 1e-6,
    },
    "cait_m36_384": {
        "dim": 768,
        "sa_ffn_layers": 36,
        "num_heads": 16,
        "image_size": 384,
        "init_values": 1e-6,
    },
    "cait_m48_448": {
        "dim": 768,
        "sa_ffn_layers": 48,
        "num_heads": 16,
        "image_size": 384,
        "init_values": 1e-6,
    },
}


def main():
    for model_name in MODEL_CONFIGS.keys():
        model_config = MODEL_CONFIGS.get(model_name)

        image_sz = model_config.get("image_size")
        proj_dim = model_config.get("dim")
        num_layers = model_config.get("sa_ffn_layers")
        num_heads = model_config.get("num_heads")
        init_values = model_config.get("init_values")

        for i in range(2):
            command = f"python convert.py -m {model_name} -is {image_sz} -pd {proj_dim} -sa {num_layers} -nh {num_heads} -iv {init_values}"
            if i == 1:
                command += " -pl"
            os.system(command)


if __name__ == "__main__":
    main()
