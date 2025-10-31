# ====================================================
# Streamlit App: Plant Type + Disease Detection (Sidebar Uploads + Dark Mode)
# ====================================================

import streamlit as st
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
import joblib
from PIL import Image, ImageOps

# ====================================================
# Page Configuration (must be first)
# ====================================================
st.set_page_config(
    page_title="🌿 Aivine",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================================================
# Logo Display (Safe + Cross-platform)
# ====================================================
import os

try:
    # --- Build absolute path for Streamlit Cloud compatibility ---
    base_path = os.path.dirname(os.path.abspath(__file__))
    logo_path = os.path.join(base_path, "Logo.png")

    # --- Load logo if exists ---
    if os.path.exists(logo_path):
        logo = Image.open(logo_path)
        base_width = 150
        w_percent = (base_width / float(logo.width))
        h_size = int((float(logo.height) * float(w_percent)))
        logo_resized = logo.resize((base_width, h_size), Image.LANCZOS)

        padding = 10
        logo_with_padding = ImageOps.expand(logo_resized, border=padding, fill=(255, 255, 255, 0))

        # --- Display in left column ---
        col_logo, _ = st.columns([2, 1])
        with col_logo:
            st.image(logo_with_padding, use_container_width=False)
    else:
        st.warning("⚠️ Logo not found. Please ensure 'Logo.png' exists in the project root.")
except Exception as e:
    st.error(f"🚫 Error loading logo: {e}")


# ====================================================
# Dark Mode Style
# ====================================================
st.markdown("""
<style>
html, body, [class*="css"]  {
    background-color: #0E1117 !important;
    color: #E0E0E0 !important;
}
.stButton>button {
    background-color:#1f2c34;
    color:#FFFFFF;
    border-radius:8px;
}
.stFileUploader>div>div {
    background-color:#1f2c34;
    color:#FFFFFF;
}
.stTextInput>div>input {
    background-color:#1f2c34;
    color:#FFFFFF;
}
</style>
""", unsafe_allow_html=True)

# ====================================================
# Load Models (cached for performance)
# ====================================================
@st.cache_resource
def load_models():
    base_path = os.path.dirname(os.path.abspath(__file__))

    # --- Model paths ---
    plant_model_path = os.path.join(base_path, "output", "models", "plant_model", "best_model_svm.pkl")
    disease_model_path = os.path.join(base_path, "output", "models", "disease_model", "mobilenetv2_plant_disease_final.keras")

    # --- Load Plant Type Model ---
    if os.path.exists(plant_model_path):
        plant_model = joblib.load(plant_model_path)
    else:
        st.warning("⚠️ Plant type model not found at 'output/models/plant_model/best_model_svm.pkl'.")
        plant_model = None

    # --- Load Feature Extractor ---
    feature_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128,128,3), pooling='avg')

    # --- Load Disease Model ---
    if os.path.exists(disease_model_path):
        disease_model = load_model(disease_model_path)
    else:
        st.warning("⚠️ Disease model not found at 'output/models/disease_model/mobilenetv2_plant_disease_final.keras'.")
        disease_model = None

    return plant_model, feature_model, disease_model


# Load models once (cached)
plant_model, feature_model, disease_model = load_models()


# ====================================================
# Class Lists
# ====================================================
plant_classes = ["Chrysanthemum", "Hibiscus", "Money Plant", "Rose", "Turmeric"]
disease_classes = [
    "Chrysanthemum_Healthy","Chrysanthemum_Bacterial_Leaf_Spot","Chrysanthemum_Septoria_Leaf_Spot",
    "Hibiscus_Healthy","Hibiscus_Blight","Hibiscus_Necrosis","Hibiscus_Scorch",
    "Money_Plant_Healthy","Money_Plant_Bacterial_Wilt","Money_Plant_Chlorosis","Money_Plant_Manganese_Toxicity",
    "Rose_Healthy","Rose_Black_Spot","Rose_Downy_Mildew","Rose_Mosaic_Virus","Rose_Powdery_Mildew",
    "Rose_Rust","Rose_Yellow_Mosaic_Virus",
    "Turmeric_Healthy","Turmeric_Aphid_Infestation","Turmeric_Blotch",
    "Turmeric_Leaf_Necrosis","Turmeric_Leaf_Spot"
]

# ====================================================
# Disease Info Dictionary
# ====================================================
disease_info = {
    "Chrysanthemum_Healthy": (
        "✅ The chrysanthemum plant appears to be in excellent health. Maintain moderate and consistent watering, "
        "keeping the soil moist but not soggy. Place the plant in a bright location with indirect sunlight, "
        "and ensure good airflow to prevent fungal issues. Regularly inspect the leaves for early signs of disease "
        "and remove any damaged or yellowing foliage promptly."
    ),
    "Chrysanthemum_Bacterial_Leaf_Spot": (
        "⚠️ This bacterial disease causes small dark brown or black spots on leaves that may merge over time. "
        "Immediately remove affected leaves and dispose of them away from healthy plants. Avoid overhead watering "
        "and ensure the plant has adequate air circulation. To control the spread, apply a copper-based bactericide "
        "once a week until the infection subsides."
    ),
    "Chrysanthemum_Septoria_Leaf_Spot": (
        "⚠️ A common fungal infection that produces dark circular spots with light centers on the foliage. "
        "Prune and discard infected leaves to prevent reinfection, and apply a fungicide suitable for ornamental plants. "
        "Always water early in the morning so the leaves have time to dry before nightfall, reducing humidity buildup."
    ),

    "Hibiscus_Healthy": (
        "✅ Your hibiscus is thriving! Keep the soil consistently moist but avoid waterlogging. "
        "Provide at least 6 hours of bright, indirect sunlight daily. Fertilize every 2–3 weeks during the growing season "
        "with a balanced liquid fertilizer to promote vibrant blooms. Remove faded flowers and prune regularly for fuller growth."
    ),
    "Hibiscus_Blight": (
        "⚠️ Hibiscus blight causes sudden wilting and browning of leaves and flowers. "
        "Remove and destroy all infected parts immediately to stop the spread. Reduce humidity around the plant "
        "and avoid splashing water on the foliage. Applying a general-purpose fungicide can help protect new growth."
    ),
    "Hibiscus_Necrosis": (
        "⚠️ This condition results in brown, dead patches on leaves or stems, often due to nutrient imbalance or poor drainage. "
        "Trim away necrotic tissue, ensure the soil drains well, and avoid overfertilizing. Add organic compost or slow-release fertilizer "
        "to restore nutrient balance and improve plant vigor."
    ),
    "Hibiscus_Scorch": (
        "⚠️ Caused by prolonged exposure to intense sunlight or heat stress. The leaves may curl, dry out, or develop brown edges. "
        "Move the plant to a partially shaded location and water deeply in the early morning. Mist the foliage occasionally "
        "to maintain humidity, but avoid wetting the flowers directly."
    ),

    "Money_Plant_Healthy": (
        "✅ The money plant is flourishing. Keep the soil slightly moist and ensure it receives bright but indirect sunlight. "
        "Feed with a diluted liquid fertilizer once a month and wipe the leaves regularly to keep them dust-free. "
        "Avoid overwatering, as the roots are sensitive to soggy conditions."
    ),
    "Money_Plant_Bacterial_Wilt": (
        "⚠️ This bacterial infection causes rapid wilting even when the soil is adequately moist. "
        "Remove and discard infected stems immediately. Improve soil drainage and avoid reusing contaminated soil or pots. "
        "Sterilize pruning tools after each use, and consider adding beneficial microbes to restore soil health."
    ),
    "Money_Plant_Chlorosis": (
        "⚠️ Chlorosis is characterized by yellowing leaves, often due to iron or nutrient deficiency. "
        "Add an iron-rich fertilizer or chelated micronutrient supplement, and ensure the plant receives sufficient sunlight. "
        "Avoid overwatering, as it may hinder nutrient uptake from the soil."
    ),
    "Money_Plant_Manganese_Toxicity": (
        "⚠️ Excess manganese in the soil can cause dark brown spots and slow growth. "
        "Flush the soil thoroughly with clean water to remove excess minerals, and avoid fertilizers containing manganese. "
        "If symptoms persist, repot the plant using fresh, balanced potting mix."
    ),

    "Rose_Healthy": (
        "✅ Your rose plant is in great condition! Water it regularly, ensuring the soil remains evenly moist but not waterlogged. "
        "Provide at least six hours of direct sunlight daily, prune old or dead branches to encourage new blooms, "
        "and apply organic fertilizer during the flowering season for optimal health."
    ),
    "Rose_Black_Spot": (
        "⚠️ A widespread fungal disease that causes round black spots on leaves, followed by yellowing and leaf drop. "
        "Remove all affected leaves and dispose of them away from the garden. Apply a fungicide every 7–10 days and ensure proper airflow. "
        "Avoid overhead watering and water only at the base of the plant."
    ),
    "Rose_Downy_Mildew": (
        "⚠️ Downy mildew appears as purple or grayish patches on the leaves, often accompanied by fuzzy growth underneath. "
        "Increase air circulation and reduce humidity levels. Remove affected leaves immediately and treat the plant with a systemic fungicide."
    ),
    "Rose_Mosaic_Virus": (
        "🚫 This viral infection causes irregular yellow mosaic patterns and leaf distortion. "
        "There is no effective cure — the best approach is to remove and destroy infected plants to prevent spread. "
        "Control insects such as aphids that can transmit the virus between plants."
    ),
    "Rose_Powdery_Mildew": (
        "⚠️ Characterized by white powdery coating on leaves, buds, and stems. "
        "Remove infected parts and spray a sulfur-based or neem oil fungicide. "
        "Ensure plants have enough spacing to improve air movement and avoid watering the foliage."
    ),
    "Rose_Rust": (
        "⚠️ A fungal disease producing orange or rust-colored pustules under the leaves. "
        "Prune infected leaves and dispose of fallen foliage. Apply a fungicide and ensure adequate sunlight and airflow "
        "to keep humidity low around the plant."
    ),
    "Rose_Yellow_Mosaic_Virus": (
        "🚫 A viral infection leading to bright yellow patches and stunted growth. "
        "Unfortunately, there’s no cure. Remove infected plants and manage pests such as aphids that can spread the virus."
    ),

    "Turmeric_Healthy": (
        "✅ The turmeric plant is thriving. Keep the soil evenly moist and well-drained. "
        "Provide partial sunlight and rich, organic soil enriched with compost. "
        "Regularly remove old leaves and maintain moderate humidity to support healthy rhizome development."
    ),
    "Turmeric_Aphid_Infestation": (
        "⚠️ Aphids are tiny sap-sucking insects that weaken the plant and cause curling leaves. "
        "Spray neem oil or insecticidal soap on affected areas, and manually remove heavily infested leaves. "
        "Encourage natural predators like ladybugs if growing outdoors."
    ),
    "Turmeric_Blotch": (
        "⚠️ This fungal disease forms brown or gray blotches on leaves, reducing photosynthesis. "
        "Prune infected leaves immediately and treat the plant with a copper-based fungicide. "
        "Avoid dense planting and water early in the morning to let the leaves dry quickly."
    ),
    "Turmeric_Leaf_Necrosis": (
        "⚠️ Characterized by brown dead patches caused by water stress or nutrient imbalance. "
        "Maintain consistent watering, avoid overfertilizing, and check for proper soil drainage. "
        "Trim necrotic leaves to prevent further stress on the plant."
    ),
    "Turmeric_Leaf_Spot": (
        "⚠️ A common fungal infection producing small brown circular spots on the leaves. "
        "Use a recommended fungicide, avoid overhead watering, and ensure the plant has proper ventilation. "
        "Regular pruning and balanced fertilization will help the plant recover faster."
    )
}


# ====================================================
# Sidebar Upload Section
# ====================================================
st.sidebar.title("🌿 About Aivine")
st.sidebar.markdown("""
<div style="background-color:#1e1e1e; padding:15px; border-radius:10px; margin-bottom:10px;">
<p style="color:#E0E0E0; font-size:14px;">
🌿 Upload images to detect plant type or disease.<br>
🌿 Plant Type uses SVM + MobileNetV2 features.<br>
🌿 Disease Detection uses MobileNetV2 CNN.<br>
🌿 For best results, upload a clear leaf image.
</p>
</div>
""", unsafe_allow_html=True)

plant_file = st.sidebar.file_uploader("Upload image for Plant Type prediction", type=["jpg","jpeg","png"], key="plant")
disease_file = st.sidebar.file_uploader("Upload image for Disease prediction (leaf only)", type=["jpg","jpeg","png"], key="disease")

# ====================================================
# Main Page Layout
# ====================================================
st.title("🌿 Aivine - Plant Type & Disease Detection")
st.write("Upload images in the sidebar to detect plant type or disease.")

col1, col2 = st.columns(2)

def resize_image(uploaded_file, size=(300, 200)):
    img = Image.open(uploaded_file)
    img = img.resize(size)
    return img

# ====================================================
# Plant Type Prediction
# ====================================================
with col1:
    st.header("Predict Plant Type")
    if plant_file is not None:
        st.image(resize_image(plant_file), caption="Uploaded Image for Plant Type")
        with st.spinner("🔍 Analyzing plant type..."):
            img_plant = load_img(plant_file, target_size=(128,128))
            img_array_plant = img_to_array(img_plant)/255.0
            img_array_plant_exp = np.expand_dims(img_array_plant, axis=0)
            features = feature_model.predict(preprocess_input(img_array_plant_exp*255.0))
            plant_pred = plant_model.predict(features)
            plant_name = plant_classes[int(plant_pred[0])]
        st.subheader(f"Predicted Plant Type: {plant_name}")

# ====================================================
# Disease Prediction
# ====================================================
with col2:
    st.header("Predict Plant Disease")
    if disease_file is not None:
        st.image(resize_image(disease_file), caption="Uploaded Image for Disease Detection")
        with st.spinner("🩺 Detecting disease..."):
            img_disease = load_img(disease_file, target_size=(224,224))
            img_array_disease = img_to_array(img_disease)/255.0
            img_array_disease_exp = np.expand_dims(img_array_disease, axis=0)
            disease_pred = disease_model.predict(img_array_disease_exp)
            disease_index = np.argmax(disease_pred, axis=1)[0]
            disease_name = disease_classes[disease_index]
        st.subheader(f"Detected Disease: {disease_name}")
        st.info(disease_info.get(disease_name, "No care information available for this disease."))

# ====================================================
# Footer
# ====================================================
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:gray;'>© 2025 Aivine Team 🌿 | Powered by Le Wagon & SDA</p>", unsafe_allow_html=True)
