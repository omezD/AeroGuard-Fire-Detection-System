import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Drone Fire Detection System",
    page_icon="🔥",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>

.stApp {
    background: linear-gradient(to bottom right, #0f172a, #111827);
    color: white;
}

h1, h2, h3, h4 {
    color: white;
}

.result-fire {
    background-color: #7f1d1d;
    color: white;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-size: 28px;
    font-weight: bold;
}

.result-safe {
    background-color: #14532d;
    color: white;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-size: 28px;
    font-weight: bold;
}

.metric-card {
    background-color: #1e293b;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
}

.footer {
    text-align: center;
    color: gray;
    margin-top: 40px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# TITLE
# =========================================================
st.title("🔥 Drone-Based Forest Fire Detection")
st.markdown("### Deep Learning Fire Segmentation using U-Net")

st.markdown("""
Upload drone images and the AI model will predict:

- 🔥 Fire Detected
- ✅ No Fire
- 📍 Fire Segmentation Mask
""")

# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_fire_model():
    model = load_model("fire_unet_final.h5", compile=False)
    return model

model = load_fire_model()

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("⚙ Detection Settings")

threshold = st.sidebar.slider(
    "Fire Threshold",
    0.1,
    1.0,
    0.5,
    0.05
)

mask_opacity = st.sidebar.slider(
    "Mask Opacity",
    0.1,
    1.0,
    0.6,
    0.1
)

# =========================================================
# IMAGE PREPROCESS
# =========================================================
IMG_SIZE = 256

def preprocess_image(image):

    image = image.resize((IMG_SIZE, IMG_SIZE))

    image_np = np.array(image)

    image_np = image_np / 255.0

    return image_np

# =========================================================
# FILE UPLOADER
# =========================================================
uploaded_files = st.file_uploader(
    "📤 Upload Drone Images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# =========================================================
# PROCESS IMAGES
# =========================================================
if uploaded_files:

    fire_percentages = []

    for file in uploaded_files:

        st.markdown("---")

        st.subheader(f"📷 {file.name}")

        image = Image.open(file).convert("RGB")

        original_image = np.array(image)

        processed_image = preprocess_image(image)

        input_image = np.expand_dims(processed_image, axis=0)

        # =================================================
        # PREDICTION
        # =================================================
        prediction = model.predict(input_image, verbose=0)[0]

        mask = prediction.squeeze()

        binary_mask = (mask > threshold).astype(np.uint8)

        # =================================================
        # FIRE AREA
        # =================================================
        fire_pixels = np.sum(binary_mask)

        total_pixels = binary_mask.shape[0] * binary_mask.shape[1]

        fire_percentage = (fire_pixels / total_pixels) * 100

        fire_percentages.append(fire_percentage)

        # =================================================
        # DETECTION STATUS
        # =================================================
        fire_detected = fire_percentage > 1.0

        # =================================================
        # CREATE HEATMAP
        # =================================================
        heatmap = cv2.applyColorMap(
            (binary_mask * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )

        heatmap = cv2.cvtColor(
            heatmap,
            cv2.COLOR_BGR2RGB
        )

        resized_original = cv2.resize(
            original_image,
            (256, 256)
        )

        overlay = cv2.addWeighted(
            resized_original,
            1 - mask_opacity,
            heatmap,
            mask_opacity,
            0
        )

        # =================================================
        # DISPLAY IMAGES
        # =================================================
        col1, col2, col3 = st.columns(3)

        with col1:
            st.image(
                original_image,
                caption="Original Drone Image",
                use_container_width=True
            )

        with col2:
            st.image(
                binary_mask * 255,
                caption="Predicted Fire Mask",
                use_container_width=True
            )

        with col3:
            st.image(
                overlay,
                caption="Fire Heatmap Overlay",
                use_container_width=True
            )

        # =================================================
        # RESULT BOX
        # =================================================
        if fire_detected:

            st.markdown(f"""
            <div class="result-fire">
            🔥 FIRE DETECTED <br><br>
            Fire Area: {fire_percentage:.2f}%
            </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown(f"""
            <div class="result-safe">
            ✅ NO FIRE DETECTED <br><br>
            Fire Area: {fire_percentage:.2f}%
            </div>
            """, unsafe_allow_html=True)

        # =================================================
        # METRICS
        # =================================================
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Fire Pixels",
                f"{fire_pixels}"
            )

        with col2:
            st.metric(
                "Fire Coverage",
                f"{fire_percentage:.2f}%"
            )

        with col3:
            st.metric(
                "Threshold",
                f"{threshold}"
            )

    # =====================================================
    # ANALYTICS SECTION
    # =====================================================
    st.markdown("---")
    st.header("📊 Detection Analytics")

    # =====================================================
    # FIRE DISTRIBUTION GRAPH
    # =====================================================
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        range(1, len(fire_percentages)+1),
        fire_percentages,
        marker='o',
        linewidth=3
    )

    ax.set_title("Fire Percentage Across Images")
    ax.set_xlabel("Image Number")
    ax.set_ylabel("Fire Percentage")

    st.pyplot(fig)

    # =====================================================
    # HISTOGRAM
    # =====================================================
    fig2, ax2 = plt.subplots(figsize=(10, 5))

    sns.histplot(
        fire_percentages,
        bins=10,
        kde=True,
        ax=ax2
    )

    ax2.set_title("Fire Distribution Histogram")
    ax2.set_xlabel("Fire Percentage")

    st.pyplot(fig2)

    # =====================================================
    # DATAFRAME
    # =====================================================
    df = pd.DataFrame({
        "Image Number": list(range(1, len(fire_percentages)+1)),
        "Fire Percentage": fire_percentages
    })

    st.subheader("📋 Prediction Summary")

    st.dataframe(
        df,
        use_container_width=True
    )

# =========================================================
# EMPTY STATE
# =========================================================
else:

    st.info("👆 Upload one or more drone images to start detection.")

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")

st.markdown("""
<div class='footer'>
Developed using TensorFlow • U-Net • Streamlit • Computer Vision
</div>
""", unsafe_allow_html=True)