import os
from tensorflow.keras.applications import MobileNetV2

DEFAULT_WEIGHTS_NAME = "mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224.h5"
DEFAULT_WEIGHTS_PATH = os.path.expanduser(
    os.path.join("~", ".keras", "models", DEFAULT_WEIGHTS_NAME)
)
ENV_WEIGHTS_PATH = "MOBILENETV2_WEIGHTS_PATH"


def get_mobilenetv2_weights_path():
    """Return the local weights path from env or the default Keras cache location."""
    env_path = os.environ.get(ENV_WEIGHTS_PATH)
    if env_path:
        return env_path
    return DEFAULT_WEIGHTS_PATH


def load_mobilenetv2(include_top=False, pooling="avg", input_shape=(224, 224, 3), allow_download=True):
    weights_path = get_mobilenetv2_weights_path()
    if os.path.exists(weights_path):
        print(f"Loading MobileNetV2 weights from local file: {weights_path}")
        return MobileNetV2(
            weights=weights_path,
            include_top=include_top,
            pooling=pooling,
            input_shape=input_shape,
        )

    print(f"Local MobileNetV2 weights not found at: {weights_path}")
    if allow_download:
        print("Falling back to Keras ImageNet weights; this may download weights if not already cached.")
        return MobileNetV2(
            weights="imagenet",
            include_top=include_top,
            pooling=pooling,
            input_shape=input_shape,
        )

    raise FileNotFoundError(
        f"MobileNetV2 weights not found locally at {weights_path}.\n"
        f"Set the environment variable {ENV_WEIGHTS_PATH} or place the weights file in ~/.keras/models/."
    )
