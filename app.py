import streamlit as st
import joblib
import numpy as np
from PIL import Image, ImageStat
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from sklearn.preprocessing import MinMaxScaler

# Load your trained models
model1 = joblib.load("best_model.pkl")
model2 = joblib.load("best_fertilizer_model.pkl")
model3 = load_model("Plantdisease.h5")

# Custom CSS to inject our own styles
st.markdown("""
    <style>
    .title {
        color: #008000;  # A shade of green
        font-size: 40px !important;
        text-align: center;
    }
    .big-font {
        font-size:20px !important;
        color: #006400;  # Darker green
    }
    .sidebar .sidebar-content {
        background-color: #006400;  # Darker green for the sidebar
    }
    .reportview-container .main .block-container{
        background-color: #90EE90;  # Light green background for main content
        padding-top: 5rem;
        padding-bottom: 5rem;
    }
    button {
        color: white;
        background-color: #32CD32;  # Lime green buttons
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# Use HTML for titles with custom classes for styling
st.markdown('<h1 class="title">Data Driven Support System For Farmers</h1>', unsafe_allow_html=True)

# Similarly, you can apply custom classes to other markdown text
#st.markdown('<p class="big-font">Welcome to the agricultural support system!</p>', unsafe_allow_html=True)



 #Soil and Crop type mappings (you can keep these mappings)
soil_type_mapping = {
    1: 'Sandy',
    2: 'Loamy',
    3: 'Black',
    4: 'Red',
    5: 'Clayey'
}

crop_type_mapping = {
    1: 'Maize',
    2: 'Sugarcane',
    3: 'Cotton',
    4: 'Tobacco',
    5: 'Paddy',
    6: 'Barley',
    7: 'Wheat',
    8: 'Millets',
    9: 'Oil seeds',
    10: 'Pulses',
    11: 'Ground Nuts'
}

fertilizer_name_mapping = {
    1: '10-26-26',
    2: '14-35-14',
    3: '17-17-17',
    4: '20-20',
    5: '28-28',
    6: 'DAP',
    7: 'Urea'
    # Add more mappings as needed
}


# Function to check if the image is likely a leaf
def is_grayscale(image_pil):
    # Check if an image is grayscale
    if image_pil.mode == "L":
        return True
    elif image_pil.mode == "RGB":
        r, g, b = image_pil.split()
        if r == g == b:
            return True
    return False

def is_grayscale(image_pil):
    # Check if an image is grayscale
    if image_pil.mode == "L":
        return True
    elif image_pil.mode == "RGB":
        r, g, b = image_pil.split()
        if r == g == b:
            return True
    return False

def is_leaf_image(image_pil):
    # Check for dominant green color in an image
    if image_pil.mode != 'RGB':
        image_pil = image_pil.convert('RGB')
    stat = ImageStat.Stat(image_pil)
    r, g, b = stat.mean
    return g > r and g > b

def is_potentially_leaf(image_pil):
    # Check if the image is grayscale
    if is_grayscale(image_pil):
        return True  # Skip color check for grayscale images

    # Perform color check for non-grayscale images
    return is_leaf_image(image_pil)




# Sidebar navigation
selected_model = st.sidebar.selectbox("Select a Model", ["Crop Recommendation", "Fertilizer Recommendation", "Plant disease detection"])

if selected_model == "Crop Recommendation":
    st.subheader("Crop Recommender System: Enter Your inputs")
    N = st.number_input("Nitrogen (N)")
    P = st.number_input("Phosphorous (P)")
    K = st.number_input("Potassium (K)")
    temperature = st.number_input("Temperature")
    humidity = st.number_input("Humidity")
    ph = st.number_input("pH")
    rainfall = st.number_input("Rainfall")

    if st.button("Predict Crop Name"):
        # Make a prediction using Model 1
        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        prediction = model1.predict(input_data)

        # Display the prediction for Model 1
        st.subheader("Model 1 Prediction:")
        st.write(f"The predicted crop is: {prediction}")

elif selected_model == "Fertilizer Recommendation":
    st.subheader("Fertilizer Recommendation: Enter Your Input")
    temperature = st.number_input("Temperature")
    humidity = st.number_input("Humidity")
    moisture = st.number_input("Moisture")

    # Select Soil Type
    selected_soil_type_num = st.selectbox(
        "Select Soil Type",
        [(num, soil_type_mapping[num]) for num in soil_type_mapping.keys()]
    )
    selected_soil_type_num = selected_soil_type_num[0]  # Get the selected soil type as a numeric value

    # Select Crop Type
    selected_crop_type_num = st.selectbox(
        "Select Crop Type",
        [(num, crop_type_mapping[num]) for num in crop_type_mapping.keys()]
    )
    selected_crop_type_num = selected_crop_type_num[0]  # Get the selected crop type as a numeric value

    nitrogen = st.number_input("Nitrogen")
    potassium = st.number_input("Potassium")
    phosphorous = st.number_input("Phosphorous")

    if st.button("Predict fertilizer Name"):
        # Make a prediction using Model 2
        input_data = np.array([temperature, humidity, moisture, selected_soil_type_num, selected_crop_type_num, nitrogen, potassium, phosphorous], dtype=float)
        input_data = input_data.reshape(1, -1)
        prediction_num = model2.predict(input_data)[0]
        predicted_fertilizer = fertilizer_name_mapping.get(int(prediction_num), "Unknown")

        # Display the prediction for Model 2 with the actual fertilizer name
        st.subheader("Model 2 Prediction:")
        st.write(f"The predicted fertilizer for Model 2 is: {predicted_fertilizer}")

elif selected_model == "Plant disease detection":
    # Load VGG16 model for feature extraction
  
        #Load VGG16 model for feature extraction
    model_vgg = VGG16(weights='imagenet', include_top=False)
    scaler = MinMaxScaler()  # Initialize a Min-Max Scaler

    def preprocess_image(image_pil):
        image_pil = image_pil.resize((224, 224))
        image_np = np.asarray(image_pil)

        # Ensure image is in RGB format
        if image_np.shape[2] == 4:  # Check for alpha channel
            image_np = image_np[:, :, :3]  # Drop the alpha channel

        image_np = preprocess_input(image_np)  # VGG16 preprocessing
        image_np = image_np.reshape(1, 224, 224, 3)  # Reshape to add batch dimension
        return image_np

    def extract_vgg_features(img_data):
        vgg_features = model_vgg.predict(img_data)
        return vgg_features.flatten()

    def predict_disease(image_pil):
        try:
            preprocessed_img = preprocess_image(image_pil)
            features = extract_vgg_features(preprocessed_img)
            features = np.expand_dims(features, axis=0)  # Add batch dimension
            prediction = model3.predict(features)

            # Log the prediction for debugging
            #st.write("Prediction array:", prediction)

            # Ensure that prediction array is not empty
            if prediction is not None and len(prediction) > 0:
                class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy'
                               'Blueberry___healthy', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___healthy', 'Corn_(maize)___Northern_Leaf_Blight',
                               'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___healthy', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Orange___Haunglongbing_(Citrus_greening)',
                               'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___healthy', 'Potato___Late_blight',
                               'Raspberry___healthy', 'Soybean___healthy', 'Strawberry___healthy', 'Strawberry___Leaf_scorch', 'Tomato___Bacterial_spot', 'Tomato___Early_blight','Tomato___Septoria_leaf_spot',
                               'Tomato___Tomato_mosaic_virus']  # Update with your class names
                predicted_class_index = np.argmax(prediction, axis=-1)
                result = class_names[predicted_class_index[0]]
                return result
            else:
                return "No valid prediction could be made."
        except Exception as e:
            # Log the exception for debugging
            st.write("An error occurred during prediction:", e)
            return "Error in prediction"



    st.title("Plant Disease Detection")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image_pil = Image.open(uploaded_file)
        st.image(image_pil, caption='Uploaded Image', use_column_width=True)

        if is_potentially_leaf(image_pil):
            #user_confirmed = st.checkbox("Is this a leaf image?")
            if st.button('Predict'):
                result = predict_disease(image_pil)
                st.write(f"The disease the plant is suffering is: {result}")
        else:
            st.write("The uploaded image does not seem to be a leaf. Please upload a leaf image.")


# Footer
st.sidebar.text("By Ghea Sandrine Mawen")
