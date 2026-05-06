import tensorflow as tf
import numpy as np


# -------------------------------------------------
# 1. GRAD-CAM CORE (FIXED)
# -------------------------------------------------
def get_gradcam(model, img_array, last_conv_layer_name):
    """
    Compute Grad-CAM heatmap (Streamlit-safe version)
    """

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        predictions = tf.convert_to_tensor(predictions)

        loss = predictions[:, 0] if len(predictions.shape) == 2 else predictions

    grads = tape.gradient(loss, conv_outputs)

    if grads is None:
        raise ValueError("Gradients are None. Check model/layer name.")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)

    max_val = tf.reduce_max(heatmap)
    if max_val == 0:
        return np.zeros((7, 7), dtype=np.float32)

    heatmap = heatmap / max_val

    return heatmap.numpy()


# -------------------------------------------------
# 2. WRAPPER
# -------------------------------------------------
def generate_gradcam(model, img_array, layer_name):
    return get_gradcam(model, img_array, layer_name)


# -------------------------------------------------
# 3. SAFE OVERLAY (NO CV2)
# -------------------------------------------------
from PIL import Image

def overlay_gradcam(original_img, heatmap, alpha=0.4):
    """
    Overlay Grad-CAM WITHOUT OpenCV (Streamlit-safe)
    """

    # Ensure numpy
    original_img = np.array(original_img)

    # Resize heatmap manually (no cv2)
    heatmap = tf.image.resize(
        heatmap[..., np.newaxis],
        (original_img.shape[0], original_img.shape[1])
    ).numpy().squeeze()

    # Normalize heatmap
    vmin, vmax = np.percentile(heatmap, (2, 98))
    heatmap = np.clip(heatmap, vmin, vmax)
    heatmap = (heatmap - vmin) / (vmax - vmin + 1e-8)
    heatmap = (heatmap * 255).astype(np.uint8)

    # Convert to RGB heatmap using matplotlib colormap
    import matplotlib.cm as cm

    colormap = cm.get_cmap("jet")
    heatmap_color = colormap(heatmap / 255.0)[..., :3]
    heatmap_color = (heatmap_color * 255).astype(np.uint8)

    # Ensure RGB image
    if original_img.max() <= 1:
        original_img = (original_img * 255).astype(np.uint8)

    # Blend images
    overlay = (1 - alpha) * original_img + alpha * heatmap_color
    overlay = overlay.astype(np.uint8)

    return overlay


# -------------------------------------------------
# 4. LAYER SELECTOR
# -------------------------------------------------
def get_last_conv_layer(model_name):
    model_name = model_name.lower()

    if model_name == "mobilenet":
        return "Conv_1"

    elif model_name == "resnet":
        return "conv5_block3_out"

    else:
        raise ValueError("Use 'mobilenet' or 'resnet'")