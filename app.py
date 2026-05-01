import streamlit as st
from inference import load_models, predict
from utils import preprocess_image
from gradcam import generate_gradcam, overlay_gradcam, get_last_conv_layer

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
# FOOTER
# -------------------------------------------------
st.divider()
st.markdown("""
---
## 👨‍💻 Authors

**MedX AI Project Team**

- 🧑‍💼 **Md. Nazmul Hossain** *(Team Lead)*  
  🔗 GitHub: https://github.com/nazmul-1117  

- 👨‍💻 **Md. Reahoon Zannah**  
  🔗 GitHub: https://github.com/pro382r  

- 👨‍💻 **Nasif Hasan Toha**

---

## ⚠️ Disclaimer
This tool is intended for **research and educational purposes only** and must not be used for clinical diagnosis.
""")

st.markdown("""
---
<div style="text-align: center; color: #888; font-size: 13px; padding: 10px;">
© 2026 <b>MedX AI Project Team</b> • All Rights Reserved • Built with Streamlit
</div>
""", unsafe_allow_html=True)