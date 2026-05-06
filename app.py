import streamlit as st
from src.inference import load_models, predict
from src.utils import preprocess_image
from src.gradcam import generate_gradcam, overlay_gradcam, get_last_conv_layer

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="MedX-PneuNet",
    page_icon="🩺",
    layout="wide"
)

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.title("🩺 MedX-PneuNet")
st.subheader("AI-powered Pneumonia Detection using MobileNetV2 & ResNet50")

st.markdown("""
### 🔬 About the System
This application uses deep learning models to detect **Pneumonia from Chest X-ray images**.

- 🧠 Models: MobileNetV2 & ResNet50  
- 🔍 Explainability: Grad-CAM heatmaps  
- ⚡ Output: Prediction + Confidence + Visual explanation  

> Upload a chest X-ray image and compare both models side-by-side.
""")

st.divider()

# -------------------------------------------------
# LOAD MODELS (cached)
# -------------------------------------------------
@st.cache_resource
def get_models():
    return load_models()

mobilenet, resnet = get_models()

# -------------------------------------------------
# UPLOAD IMAGE
# -------------------------------------------------
uploaded_file = st.file_uploader(
    "📤 Upload Chest X-ray Image",
    type=["jpg", "jpeg", "png"]
)

# -------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------
if uploaded_file:

    # ---------------- PREPROCESS ----------------
    img, img_array = preprocess_image(uploaded_file)

    # ---------------- PREDICTIONS ----------------
    mob_label, mob_conf, _ = predict(mobilenet, img_array)
    res_label, res_conf, _ = predict(resnet, img_array)

    # ---------------- GRAD-CAM ----------------
    mob_layer = get_last_conv_layer("mobilenet")
    res_layer = get_last_conv_layer("resnet")

    mob_heatmap = generate_gradcam(mobilenet, img_array, mob_layer)
    res_heatmap = generate_gradcam(resnet, img_array, res_layer)

    mob_overlay = overlay_gradcam(img, mob_heatmap)
    res_overlay = overlay_gradcam(img, res_heatmap)

    # -------------------------------------------------
    # LAYOUT
    # -------------------------------------------------
    col1, col2 = st.columns(2)

    # ================= MOBILE NET =================
    with col1:
        st.markdown("## 📱 MobileNetV2")

        st.image(img, caption="Input Image", use_container_width=True)

        st.metric("Prediction", mob_label)
        st.metric("Confidence", f"{mob_conf:.2%}")

        st.markdown("### 🔍 Grad-CAM Heatmap")
        st.image(mob_heatmap, use_container_width=True, clamp=True)

        st.markdown("### 🎯 Overlay")
        st.image(mob_overlay, use_container_width=True)

    # ================= RESNET =================
    with col2:
        st.markdown("## 🧠 ResNet50")

        st.image(img, caption="Input Image", use_container_width=True)

        st.metric("Prediction", res_label)
        st.metric("Confidence", f"{res_conf:.2%}")

        st.markdown("### 🔍 Grad-CAM Heatmap")
        st.image(res_heatmap, use_container_width=True, clamp=True)

        st.markdown("### 🎯 Overlay")
        st.image(res_overlay, use_container_width=True)

    st.divider()

    # -------------------------------------------------
    # COMPARISON SUMMARY
    # -------------------------------------------------
    st.markdown("## 📊 Model Comparison")

    if mob_label == res_label:
        st.success("Both models agree on the prediction.")
    else:
        st.warning("Models disagree — check Grad-CAM explanations.")

    st.write(f"""
- **MobileNetV2:** {mob_label} ({mob_conf:.2%})  
- **ResNet50:** {res_label} ({res_conf:.2%})  
""")
    

# -------------------------------------------------
# OPEN SOURCE + GITHUB SECTION
# -------------------------------------------------

st.markdown("---")

st.markdown("""
## 🌐 Open Source Project

This project is fully **open-source** and welcomes contributions from the community.

### 🔓 License
![MIT License](https://img.shields.io/badge/License-MIT-green.svg)

This project is licensed under the **MIT License**, which means:
- ✔ You can use it freely
- ✔ You can modify it
- ✔ You can distribute it
- ✔ You can contribute improvements

👉 License file: https://github.com/nazmul-1117/medx-pneunet/blob/main/LICENSE

---

### 📦 GitHub Repository

👉 Source Code: https://github.com/nazmul-1117/medx-pneunet

![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-black?logo=github)

---

### 🤝 Contribute to this Project

We welcome contributions from developers, researchers, and students!

You can contribute by:
- 🐛 Reporting bugs
- 🚀 Improving model accuracy
- 🎨 Enhancing UI/UX
- 🧠 Adding new AI models (e.g., EfficientNet, DenseNet)
- 📊 Improving Grad-CAM visualizations
- 📄 Improving documentation

👉 Feel free to fork and submit a pull request!

---

### 💡 Open Source Philosophy

> “AI should be transparent, explainable, and accessible to everyone.”

This project follows that principle by providing:
- Explainable AI (Grad-CAM)
- Open training/inference pipeline
- Reproducible ML workflow
""")