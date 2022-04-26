import argparse

import tensorflow as tf

from cait.layers import ClassAttn, LayerScale, TalkingHeadAttn
from cait.model_configs import base_config
from cait.models import CaiT
from utils import helpers


def parse_args():
    parser = argparse.ArgumentParser(
        description="Conversion of the PyTorch pre-trained CaiT weights to TensorFlow."
    )
    parser.add_argument(
        "-m",
        "--model-name",
        default="cait_xxs24_224",
        type=str,
        choices=[
            "cait_xxs24_224",
            "cait_xxs24_384",
            "cait_xxs36_224",
            "cait_xxs36_384",
            "cait_xs24_384",
            "cait_s24_224",
            "cait_s24_384",
            "cait_s36_384",
            "cait_m36_384",
            "cait_m48_448",
        ],
        help="Name of the DeiT model variant.",
    )
    parser.add_argument(
        "-is",
        "--image-size",
        default=224,
        type=int,
        choices=[224, 384, 448],
        help="Image resolution.",
    )
    parser.add_argument(
        "-p",
        "--patch-size",
        default=16,
        type=int,
        help="Patch size.",
    )
    parser.add_argument(
        "-pd",
        "--projection-dim",
        default=192,
        type=int,
        help="Patch projection dimension.",
    )
    parser.add_argument(
        "-sa",
        "--sa-ffn-layers",
        default=24,
        type=int,
        help="Number of SA-FFN layers.",
    )
    parser.add_argument(
        "-nh",
        "--num-heads",
        default=4,
        type=int,
        help="Number of attention heads.",
    )
    parser.add_argument(
        "-iv",
        "--init-values",
        default=1e-5,
        type=float,
        help="Initial value for LayerScale.",
    )
    parser.add_argument(
        "-pl",
        "--pre-logits",
        action="store_true",
        help="If we don't need the classification outputs.",
    )
    return parser.parse_args()


def main(args):
    tf_model_config = base_config.get_config(**vars(args))
    tf_model = CaiT(tf_model_config)

    dummy_inputs = tf.ones((2, args.image_size, args.image_size, 3))
    _ = tf_model(dummy_inputs)
    all_layers = [layer.name for layer in tf_model.layers]

    all_sa_ffn_blocks = list(filter(lambda x: "sa_ffn_block" in x, all_layers))
    all_ca_ffn_blocks = list(filter(lambda x: "ca_ffn_block" in x, all_layers))

    print(len(all_sa_ffn_blocks), len(all_ca_ffn_blocks))


if __name__ == "__main__":
    args = parse_args()
    main(args)
