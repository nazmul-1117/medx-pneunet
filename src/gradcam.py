import tensorflow as tf
import numpy as np
import cv2


# -------------------------------------------------
# 1. CORE GRAD-CAM FUNCTION (FIXED & STABLE)
# -------------------------------------------------
def get_gradcam(model, img_array, last_conv_layer_name):
    """
    Computes Grad-CAM heatmap for a single image.
    """

    # Build model that maps input -> (conv layer, prediction)
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)

        # Ensure tensor (fixes your crash)
        predictions = tf.convert_to_tensor(predictions)

        # Handle binary classification safely
        if len(predictions.shape) == 2:
            loss = predictions[:, 0]
        else:
            loss = predictions

    # Compute gradients
    grads = tape.gradient(loss, conv_outputs)

    if grads is None:
        raise ValueError("Gradients returned None. Check model architecture.")

    # Global average pooling
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight feature maps
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU (keep positive influence only)
    heatmap = tf.maximum(heatmap, 0)

    # Avoid divide-by-zero
    max_val = tf.reduce_max(heatmap)

    if max_val == 0:
        return np.zeros((7, 7), dtype=np.float32)

    heatmap = heatmap / max_val

    return heatmap.numpy()


# -------------------------------------------------
# 2. WRAPPER FUNCTION
# -------------------------------------------------
def generate_gradcam(model, img_array, layer_name):
    """
    Wrapper for Grad-CAM generation.
    """
    return get_gradcam(model, img_array, layer_name)


# -------------------------------------------------
# 3. HEATMAP OVERLAY FUNCTION (STREAMLIT SAFE)
# -------------------------------------------------
def overlay_gradcam(original_img, heatmap, alpha=0.4):
    """
    Overlay Grad-CAM heatmap on original image.
    """

    # Resize heatmap to image size
    heatmap = cv2.resize(
        heatmap,
        (original_img.shape[1], original_img.shape[0])
    )

    vmin, vmax = np.percentile(heatmap, (2, 98))
    heatmap = np.clip(heatmap, vmin, vmax)
    heatmap = (heatmap - vmin) / (vmax - vmin + 1e-8)
    heatmap = np.uint8(255 * heatmap)

    # Normalize to 0–255
    # heatmap = np.uint8(255 * heatmap)

    # Apply color map
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Ensure image format is uint8
    if original_img.max() <= 1:
        original_img = (original_img * 255).astype(np.uint8)

    # Convert grayscale to RGB if needed
    if len(original_img.shape) == 2:
        original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)

    # Overlay
    overlay = cv2.addWeighted(original_img, 1 - alpha, heatmap_color, alpha, 0)

    return overlay


# -------------------------------------------------
# 4. MODEL-SPECIFIC LAYER SELECTOR (RESEARCH SAFE)
# -------------------------------------------------
def get_last_conv_layer(model_name):
    """
    Returns last convolutional layer for Grad-CAM.
    """

    model_name = model_name.lower()

    if model_name == "mobilenet":
        return "Conv_1"

    elif model_name == "resnet":
        return "conv5_block3_out"

    else:
        raise ValueError(
            "Unknown model. Use 'mobilenet' or 'resnet'"
        )