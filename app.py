# ====================================================
# Streamlit App: Plant Type + Disease Detection (Sidebar Uploads + Dark Mode)
# ====================================================

import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
import joblib
from PIL import Image, ImageOps




logo = Image.open("Logo.png")

base_width = 150
w_percent = (base_width / float(logo.width))
h_size = int((float(logo.height) * float(w_percent)))
logo_resized = logo.resize((base_width, h_size), Image.LANCZOS)

padding = 10
logo_with_padding = ImageOps.expand(logo_resized, border=padding, fill=(0,0,0,0))

col_logo, _ = st.columns([2,1])
col_logo.image(logo_with_padding, use_container_width=False)


# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="🌿 Aivine",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Dark Mode CSS
# -----------------------------
st.markdown("""
<style>
body {background-color: #0E1117; color: #FFFFFF;}
h1, h2, h3, h4, h5, h6, p, span {color: #E0E0E0;}
.stButton>button {background-color:#1f2c34; color:#FFFFFF;}
.stFileUploader>div>div {background-color:#1f2c34; color:#FFFFFF;}
.stTextInput>div>input {background-color:#1f2c34; color:#FFFFFF;}
.css-1d391kg, .stMarkdown {color: #E0E0E0;}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Models
# -----------------------------
plant_model = joblib.load("best_model_svm.pkl")
plant_classes = ["Chrysanthemum", "Hibiscus", "Money Plant", "Rose", "Turmeric"]
feature_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128,128,3), pooling='avg')

disease_model = load_model("output/models/mobilenetv2_plant_disease_final.keras")
disease_classes = [
    "Chrysanthemum_Healthy","Chrysanthemum_Bacterial_Leaf_Spot","Chrysanthemum_Septoria_Leaf_Spot",
    "Hibiscus_Healthy","Hibiscus_Blight","Hibiscus_Necrosis","Hibiscus_Scorch",
    "Money_Plant_Healthy","Money_Plant_Bacterial_Wilt","Money_Plant_Chlorosis","Money_Plant_Manganese_Toxicity",
    "Rose_Healthy","Rose_Black_Spot","Rose_Downy_Mildew","Rose_Mosaic_Virus","Rose_Powdery_Mildew",
    "Rose_Rust","Rose_Yellow_Mosaic_Virus",
    "Turmeric_Healthy","Turmeric_Aphid_Infestation","Turmeric_Blotch",
    "Turmeric_Leaf_Necrosis","Turmeric_Leaf_Spot"
]

disease_info = {
    "Chrysanthemum_Healthy": "Plant is healthy. Water moderately and give indirect sunlight.",
    "Chrysanthemum_Bacterial_Leaf_Spot": "Remove affected leaves, avoid overhead watering, ensure airflow.",
    "Chrysanthemum_Septoria_Leaf_Spot": "Prune infected areas, use fungicide spray, avoid wetting foliage.",

    "Hibiscus_Healthy": "Plant is healthy. Keep soil moist and provide bright light.",
    "Hibiscus_Blight": "Remove damaged leaves, reduce humidity, apply fungicide.",
    "Hibiscus_Necrosis": "Cut off necrotic parts, check soil drainage, reduce fertilizer.",
    "Hibiscus_Scorch": "Move plant to shaded area and water deeply.",

    "Money_Plant_Healthy": "Plant is healthy. Keep soil slightly moist and indirect sunlight.",
    "Money_Plant_Bacterial_Wilt": "Remove infected stems, improve drainage, avoid overwatering.",
    "Money_Plant_Chlorosis": "Add iron-rich fertilizer, ensure proper sunlight.",
    "Money_Plant_Manganese_Toxicity": "Flush soil with water, avoid manganese fertilizers.",

    "Rose_Healthy": "Plant is healthy. Water regularly and provide full sun.",
    "Rose_Black_Spot": "Prune infected leaves, apply fungicide, avoid wetting foliage.",
    "Rose_Downy_Mildew": "Increase airflow, remove affected leaves, reduce humidity.",
    "Rose_Mosaic_Virus": "No cure. Remove infected plants to prevent spread.",
    "Rose_Powdery_Mildew": "Spray fungicide, avoid crowded planting, water at base.",
    "Rose_Rust": "Remove affected leaves, use fungicide, maintain good airflow.",
    "Rose_Yellow_Mosaic_Virus": "No cure. Remove infected plants and control aphids.",

    "Turmeric_Healthy": "Plant is healthy. Keep soil moist and provide partial sun.",
    "Turmeric_Aphid_Infestation": "Spray insecticidal soap, remove infested leaves.",
    "Turmeric_Blotch": "Remove infected leaves, apply appropriate fungicide.",
    "Turmeric_Leaf_Necrosis": "Ensure proper watering, remove necrotic leaves.",
    "Turmeric_Leaf_Spot": "Treat with fungicide, improve air circulation, avoid wet foliage."
}

IMG_SIZE_PLANT = (128,128)
IMG_SIZE_DISEASE = (224,224)

# ====================================================
# Sidebar Uploads
# ====================================================
st.sidebar.title("🌿 About Aivine")
st.sidebar.markdown(
    """
    <div style="background-color:#1e1e1e; padding:15px; border-radius:10px; margin-bottom:10px;">
        <p style="color:#E0E0E0; font-size:14px;">
        🌿 Upload images to detect plant type or disease.<br>
        🌿 Plant Type uses SVM + MobileNetV2 features.<br>
        🌿 Disease Detection uses MobileNetV2 CNN.<br>
        🌿 For best disease results, upload a clear leaf image.
        </p>
    </div>
    """, unsafe_allow_html=True
)

plant_file = st.sidebar.file_uploader(
    "Upload image for Plant Type prediction", type=["jpg","jpeg","png"], key="plant"
)
disease_file = st.sidebar.file_uploader(
    "Upload image for Disease prediction (leaf only)", type=["jpg","jpeg","png"], key="disease"
)



# ====================================================
# Main Page
# ====================================================
st.title("🌿 Aivine - Plant Type & Disease Detection")
st.write("Upload images in the sidebar to detect plant type or disease.")

col1, col2 = st.columns(2)

# -----------------------------
# Plant Type Prediction
# -----------------------------
with col1:
    st.header("Predict Plant Type")
    if plant_file is not None:
        st.image(plant_file, caption="Uploaded Image for Plant Type", use_container_width=True)
        img_plant = load_img(plant_file, target_size=IMG_SIZE_PLANT)
        img_array_plant = img_to_array(img_plant)/255.0
        img_array_plant_exp = np.expand_dims(img_array_plant, axis=0)
        features = feature_model.predict(preprocess_input(img_array_plant_exp*255.0))
        plant_pred = plant_model.predict(features)
        plant_name = plant_classes[plant_pred[0]]
        st.subheader(f"Predicted Plant Type: {plant_name}")

# -----------------------------
# Disease Prediction
# -----------------------------
with col2:
    st.header("Predict Plant Disease")
    if disease_file is not None:
        st.image(disease_file, caption="Uploaded Image for Disease Detection", use_container_width=True)
        img_disease = load_img(disease_file, target_size=IMG_SIZE_DISEASE)
        img_array_disease = img_to_array(img_disease)/255.0
        img_array_disease_exp = np.expand_dims(img_array_disease, axis=0)
        disease_pred = disease_model.predict(img_array_disease_exp)
        disease_index = np.argmax(disease_pred, axis=1)[0]
        disease_name = disease_classes[disease_index]
        st.subheader(f"Detected Disease: {disease_name}")
        st.info(disease_info.get(disease_name, "No care information available for this disease."))
