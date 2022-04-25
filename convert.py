import argparse
import os
import sys

import tensorflow as tf
import timm
from tensorflow import keras

sys.path.append("..")

from cait.layers import ClassAttn, LayerScale, TalkingHeadAttn
from cait.model_configs import base_config
from cait.models import CaiT
from utils import helpers

TF_MODEL_ROOT = "gs://cait-tf"


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
    if args.pre_logits:
        print(f"Converting {args.model_name} for feature extraction...")
    else:
        print(f"Converting {args.model_name}...")

    print("Instantiating PyTorch model...")
    pt_model = timm.create_model(
        model_name=args.model_name, num_classes=1000, pretrained=True
    )
    pt_model.eval()

    print("Instantiating TF model...")
    tf_model_config = base_config.get_config(**vars(args))
    tf_model = CaiT(tf_model_config)

    dummy_inputs = tf.ones((2, args.image_size, args.image_size, 3))
    _ = tf_model(dummy_inputs)

    if not args.pre_logits:
        assert tf_model.count_params() == sum(
            p.numel() for p in pt_model.parameters()
        )

    # Load the PT params.
    pt_model_dict = pt_model.state_dict()
    pt_model_dict = {k: pt_model_dict[k].numpy() for k in pt_model_dict}

    print("Beginning parameter porting process...")

    # Projection layers.
    tf_model.layers[0].layers[0] = helpers.modify_tf_block(
        tf_model.layers[0].layers[0],
        pt_model_dict["patch_embed.proj.weight"],
        pt_model_dict["patch_embed.proj.bias"],
    )

    # Positional embedding.
    tf_model.pos_embed.assign(tf.Variable(pt_model_dict["pos_embed"]))

    # CLS token.
    tf_model.cls_token.assign(tf.Variable(pt_model_dict["cls_token"]))

    # Layer norm layers.
    ln_idx = -2
    tf_model.layers[ln_idx] = helpers.modify_tf_block(
        tf_model.layers[ln_idx],
        pt_model_dict["norm.weight"],
        pt_model_dict["norm.bias"],
    )

    # Head layers.
    if not args.pre_logits:
        head_layer = tf_model.get_layer("classification_head")
        head_layer_idx = -1
        tf_model.layers[head_layer_idx] = helpers.modify_tf_block(
            head_layer,
            pt_model_dict["head.weight"],
            pt_model_dict["head.bias"],
        )

    # SA-FFN and CA-FFN blocks.
    idx = 0
    start_ca = 0

    for outer_layer in tf_model.layers:
        if (
            isinstance(outer_layer, tf.keras.Model)
            and outer_layer.name != "projection"
        ):
            tf_block = tf_model.get_layer(outer_layer.name)
            pt_block_name = (
                f"blocks.{idx}" if idx < 24 else f"blocks_token_only.{start_ca}"
            )

            # LayerNorm layers.
            layer_norm_idx = 1
            for layer in tf_block.layers:
                if isinstance(layer, keras.layers.LayerNormalization):
                    # print(f"LayerNorm visited: {layer.name}.")
                    layer_norm_pt_prefix = (
                        f"{pt_block_name}.norm{layer_norm_idx}"
                    )
                    layer.gamma.assign(
                        tf.Variable(
                            pt_model_dict[f"{layer_norm_pt_prefix}.weight"]
                        )
                    )
                    layer.beta.assign(
                        tf.Variable(
                            pt_model_dict[f"{layer_norm_pt_prefix}.bias"]
                        )
                    )
                    layer_norm_idx += 1

            # LayerScale gammas.
            layer_scale_idx = 1
            for layer in tf_block.layers:
                if isinstance(layer, LayerScale):
                    # print(f"LayerScale visited: {layer.name}.")
                    layer_scale_pt_prefix = (
                        f"{pt_block_name}.gamma_{layer_scale_idx}"
                    )
                    layer.gamma.assign(
                        tf.Variable(pt_model_dict[f"{layer_scale_pt_prefix}"])
                    )
                    layer_scale_idx += 1

            # FFN layers.
            ffn_layer_idx = 1
            for layer in tf_block.layers:
                if isinstance(layer, keras.layers.Dense):
                    # print(f"FFN visited: {layer.name}.")
                    dense_layer_pt_prefix = (
                        f"{pt_block_name}.mlp.fc{ffn_layer_idx}"
                    )
                    layer = helpers.modify_tf_block(
                        layer,
                        pt_model_dict[f"{dense_layer_pt_prefix}.weight"],
                        pt_model_dict[f"{dense_layer_pt_prefix}.bias"],
                    )
                    ffn_layer_idx += 1

            # Self-attention (SA).
            if "blocks_token_only" not in pt_block_name:
                for layer in tf_block.layers:
                    if isinstance(layer, TalkingHeadAttn):
                        # print(f"SA visited: {layer.name}.")
                        attn_layer_pt_prefix = f"{pt_block_name}.attn"

                        # QKV.
                        layer.qkv = helpers.modify_tf_block(
                            layer.qkv,
                            pt_model_dict[f"{attn_layer_pt_prefix}.qkv.weight"],
                            pt_model_dict[f"{attn_layer_pt_prefix}.qkv.bias"],
                        )
                        # Projection l.
                        layer.proj_l = helpers.modify_tf_block(
                            layer.proj_l,
                            pt_model_dict[
                                f"{attn_layer_pt_prefix}.proj_l.weight"
                            ],
                            pt_model_dict[
                                f"{attn_layer_pt_prefix}.proj_l.bias"
                            ],
                        )
                        # Projection w.
                        layer.proj_w = helpers.modify_tf_block(
                            layer.proj_w,
                            pt_model_dict[
                                f"{attn_layer_pt_prefix}.proj_w.weight"
                            ],
                            pt_model_dict[
                                f"{attn_layer_pt_prefix}.proj_w.bias"
                            ],
                        )

                        # Final dense projection.
                        layer.proj = helpers.modify_tf_block(
                            layer.proj,
                            pt_model_dict[
                                f"{attn_layer_pt_prefix}.proj.weight"
                            ],
                            pt_model_dict[f"{attn_layer_pt_prefix}.proj.bias"],
                        )
            else:
                # Class-attention (CA).
                for layer in tf_block.layers:
                    if isinstance(layer, ClassAttn):
                        attn_layer_pt_prefix = f"{pt_block_name}.attn"

                        # QKV.
                        layer.q = helpers.modify_tf_block(
                            layer.q,
                            pt_model_dict[f"{attn_layer_pt_prefix}.q.weight"],
                            pt_model_dict[f"{attn_layer_pt_prefix}.q.bias"],
                        )
                        layer.k = helpers.modify_tf_block(
                            layer.k,
                            pt_model_dict[f"{attn_layer_pt_prefix}.k.weight"],
                            pt_model_dict[f"{attn_layer_pt_prefix}.k.bias"],
                        )
                        layer.v = helpers.modify_tf_block(
                            layer.v,
                            pt_model_dict[f"{attn_layer_pt_prefix}.v.weight"],
                            pt_model_dict[f"{attn_layer_pt_prefix}.v.bias"],
                        )

                        # Final dense projection.
                        layer.proj = helpers.modify_tf_block(
                            layer.proj,
                            pt_model_dict[
                                f"{attn_layer_pt_prefix}.proj.weight"
                            ],
                            pt_model_dict[f"{attn_layer_pt_prefix}.proj.bias"],
                        )

            idx += 1
            # Since the minimum depth (number of SA-FFN layers) of CaiT models
            # is 24 this is a valid assumption.
            if idx > 24:
                start_ca += 1

    print("Porting successful, serializing TensorFlow model...")

    save_path = os.path.join(TF_MODEL_ROOT, args.model_name)
    save_path = f"{save_path}_fe" if args.pre_logits else save_path
    tf_model.save(save_path)
    print(f"TensorFlow model serialized to: {save_path}...")


if __name__ == "__main__":
    args = parse_args()
    main(args)
