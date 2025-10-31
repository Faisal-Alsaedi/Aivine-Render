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
    "Chrysanthemum_Healthy": (
        "✅ The chrysanthemum plant is in good health. Maintain moderate watering, "
        "avoid soggy soil, and ensure it receives bright but indirect sunlight. "
        "Regularly inspect for pests and keep the leaves dry to prevent fungal issues."
    ),
    "Chrysanthemum_Bacterial_Leaf_Spot": (
        "⚠️ Caused by bacterial infection leading to small brown or black spots on leaves. "
        "Remove all affected leaves immediately, avoid overhead watering, and increase air circulation. "
        "Apply a copper-based bactericide if the infection spreads."
    ),
    "Chrysanthemum_Septoria_Leaf_Spot": (
        "⚠️ A fungal disease that creates dark circular spots with light centers on leaves. "
        "Prune and discard infected foliage, apply a suitable fungicide spray, "
        "and water the plant early in the day to allow the leaves to dry quickly."
    ),

    "Hibiscus_Healthy": (
        "✅ The hibiscus plant is thriving. Keep the soil consistently moist but not waterlogged, "
        "and provide plenty of bright, indirect light. Feed regularly with a balanced fertilizer."
    ),
    "Hibiscus_Blight": (
        "⚠️ A fungal disease that causes sudden wilting and browning of leaves or flowers. "
        "Remove and discard all damaged parts, reduce humidity levels, and apply a broad-spectrum fungicide. "
        "Avoid splashing water on the leaves when watering."
    ),
    "Hibiscus_Necrosis": (
        "⚠️ Characterized by dead, brown patches on leaves and stems due to nutrient imbalance or poor drainage. "
        "Trim necrotic sections, check soil drainage, and reduce excess fertilizer. "
        "Ensure proper watering and healthy soil aeration."
    ),
    "Hibiscus_Scorch": (
        "⚠️ Caused by excessive sunlight or heat stress, resulting in leaf browning or curling. "
        "Move the plant to a partially shaded area and water deeply. "
        "Avoid watering during the hottest part of the day."
    ),

    "Money_Plant_Healthy": (
        "✅ The money plant looks healthy. Maintain slightly moist soil and indirect sunlight. "
        "Fertilize monthly with a mild liquid fertilizer and wipe the leaves occasionally to keep them dust-free."
    ),
    "Money_Plant_Bacterial_Wilt": (
        "⚠️ A bacterial disease that causes sudden wilting even when the soil is moist. "
        "Remove infected stems immediately, improve drainage, and avoid waterlogging. "
        "Disinfect pruning tools after use to prevent the spread."
    ),
    "Money_Plant_Chlorosis": (
        "⚠️ Characterized by yellowing leaves due to iron or nutrient deficiency. "
        "Add iron-rich fertilizer or compost, and ensure the plant gets enough sunlight. "
        "Avoid overwatering, which can worsen nutrient absorption."
    ),
    "Money_Plant_Manganese_Toxicity": (
        "⚠️ Occurs when manganese levels in the soil are too high, leading to dark leaf spots and poor growth. "
        "Flush the soil thoroughly with clean water and avoid fertilizers containing manganese. "
        "Repot if the condition persists."
    ),

    "Rose_Healthy": (
        "✅ The rose plant is healthy. Water regularly, provide at least six hours of sunlight daily, "
        "and prune old branches to encourage new growth. Apply organic fertilizer during blooming season."
    ),
    "Rose_Black_Spot": (
        "⚠️ A common fungal disease that causes black spots on leaves, leading to yellowing and drop. "
        "Prune and discard infected leaves, apply fungicide regularly, and avoid overhead watering. "
        "Ensure good airflow between plants."
    ),
    "Rose_Downy_Mildew": (
        "⚠️ Fungal infection producing purplish spots and fuzzy growth on the undersides of leaves. "
        "Increase ventilation, remove infected leaves, and reduce humidity. "
        "Apply a systemic fungicide if needed."
    ),
    "Rose_Mosaic_Virus": (
        "🚫 A viral disease causing yellow mosaic patterns or distorted leaves. "
        "Unfortunately, there is no cure — remove and destroy infected plants to prevent spreading. "
        "Control aphids and insects that can transmit the virus."
    ),
    "Rose_Powdery_Mildew": (
        "⚠️ Appears as white powdery coating on leaves and stems. "
        "Apply fungicide, improve air circulation, and avoid crowding plants. "
        "Water only at the base to keep foliage dry."
    ),
    "Rose_Rust": (
        "⚠️ A fungal disease forming orange or rust-colored spots under leaves. "
        "Remove affected foliage, apply fungicide, and ensure proper airflow. "
        "Clean up fallen leaves to reduce reinfection."
    ),
    "Rose_Yellow_Mosaic_Virus": (
        "🚫 Viral infection causing bright yellow patches on leaves and stunted growth. "
        "No cure exists; remove infected plants and control pests like aphids that spread the virus."
    ),

    "Turmeric_Healthy": (
        "✅ The turmeric plant is healthy. Keep the soil evenly moist, provide partial sunlight, "
        "and ensure rich, well-draining soil. Add organic compost regularly."
    ),
    "Turmeric_Aphid_Infestation": (
        "⚠️ Caused by sap-sucking insects that weaken the plant and curl leaves. "
        "Spray insecticidal soap or neem oil, and manually remove heavily infested leaves."
    ),
    "Turmeric_Blotch": (
        "⚠️ A fungal disease that creates brown or gray blotches on leaves. "
        "Cut off infected leaves, use a recommended fungicide, and ensure proper spacing for airflow."
    ),
    "Turmeric_Leaf_Necrosis": (
        "⚠️ Leads to brown, dead patches on leaves due to water stress or nutrient issues. "
        "Ensure consistent watering, remove necrotic leaves, and apply balanced fertilizer."
    ),
    "Turmeric_Leaf_Spot": (
        "⚠️ Fungal infection causing small brown circular spots. "
        "Treat with fungicide, avoid overhead watering, and maintain good ventilation."
    )
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



def resize_image(uploaded_file, size=(300, 200)):
    img = Image.open(uploaded_file)
    img = img.resize(size)
    return img

# -----------------------------
# Plant Type Prediction
# -----------------------------
with col1:
    st.header("Predict Plant Type")
    if plant_file is not None:
        st.image(resize_image(plant_file), caption="Uploaded Image for Plant Type")
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
        st.image(resize_image(disease_file), caption="Uploaded Image for Disease Detection")
        img_disease = load_img(disease_file, target_size=IMG_SIZE_DISEASE)
        img_array_disease = img_to_array(img_disease)/255.0
        img_array_disease_exp = np.expand_dims(img_array_disease, axis=0)
        disease_pred = disease_model.predict(img_array_disease_exp)
        disease_index = np.argmax(disease_pred, axis=1)[0]
        disease_name = disease_classes[disease_index]
        st.subheader(f"Detected Disease: {disease_name}")
        st.info(disease_info.get(disease_name, "No care information available for this disease."))
