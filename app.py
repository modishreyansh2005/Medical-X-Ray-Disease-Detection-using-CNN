# import streamlit as st
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.utils import img_to_array # type: ignore
# from PIL import Image

# # -----------------------------
# # PAGE CONFIG
# # -----------------------------
# st.set_page_config(
#     page_title="AI X-Ray Diagnosis",
#     page_icon="🩻",
#     layout="wide"
# )

# # -----------------------------
# # LOAD MODEL
# # -----------------------------
# model = tf.keras.models.load_model("xray_cnn_model.h5")

# # -----------------------------
# # CUSTOM CSS (Professional Look)
# # -----------------------------
# st.markdown("""
#     <style>
#     .main-title {
#         font-size:40px;
#         font-weight:700;
#         color:#0E4D64;
#     }
#     .result-box {
#         padding:20px;
#         border-radius:10px;
#         background-color:#F5F9FF;
#         border:1px solid #D6E4FF;
#     }
#     </style>
# """, unsafe_allow_html=True)

# # -----------------------------
# # SIDEBAR
# # -----------------------------
# st.sidebar.title("🏥 About Project")
# st.sidebar.info("""
# AI-powered Chest X-ray Classification System.

# Model: CNN  
# Classes: Normal / Pneumonia  
# """)

# # -----------------------------
# # MAIN TITLE
# # -----------------------------
# st.markdown("<p class='main-title'>🩻 AI Chest X-Ray Diagnosis System</p>", unsafe_allow_html=True)
# st.write("Upload a chest X-ray image to receive AI-based prediction and summary.")

# # -----------------------------
# # FILE UPLOAD
# # -----------------------------
# uploaded_file = st.file_uploader("📤 Upload X-ray Image", type=["jpg", "jpeg", "png"])

# def predict_xray(img):
#     img = img.resize((224,224))
#     img_array = img_to_array(img) / 255.0
#     img_array = np.expand_dims(img_array, axis=0)

#     prediction = model.predict(img_array)[0][0]

#     if prediction > 0.5:
#         return "PNEUMONIA", prediction
#     else:
#         return "NORMAL", 1 - prediction

# # -----------------------------
# # PREDICTION SECTION
# # -----------------------------
# if uploaded_file is not None:
#     col1, col2 = st.columns(2)

#     img = Image.open(uploaded_file)

#     with col1:
#         st.image(img, caption="Uploaded X-ray", use_column_width=True)

#     with col2:
#         label, confidence = predict_xray(img)

#         st.markdown("<div class='result-box'>", unsafe_allow_html=True)
#         st.subheader("🔍 Prediction Result")

#         if label == "PNEUMONIA":
#             st.error("⚠ Pneumonia Detected")
#         else:
#             st.success("✔ Normal X-ray")

#         st.write(f"**Confidence:** {confidence*100:.2f}%")
#         st.progress(float(confidence))

#         st.subheader("📝 Medical Summary")

#         if label == "PNEUMONIA":
#             st.write("""
#             The X-ray shows possible lung opacity and infection signs 
#             consistent with pneumonia. Further clinical evaluation is advised.
#             """)
#         else:
#             st.write("""
#             The X-ray appears normal with no visible signs 
#             of infection or lung abnormalities.
#             """)

#         st.markdown("</div>", unsafe_allow_html=True)

# # -----------------------------
# # FOOTER
# # -----------------------------
# st.markdown("---")
# st.caption("Developed using CNN & Streamlit | AI Medical Project")


from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array

import os

app = Flask(__name__)

# Folder to save uploaded images
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load trained model
model = tf.keras.models.load_model("xray_cnn_model.h5")

def predict_xray(image_path):
    img = load_img(image_path, target_size=(224,224))
    img_array = img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        return "PNEUMONIA", float(prediction)
    else:
        return "NORMAL", float(1 - prediction)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]

        if file:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)

            label, confidence = predict_xray(filepath)

            if label == "PNEUMONIA":
                summary = "Signs of lung infection detected. Please consult a doctor."
            else:
                summary = "X-ray appears normal."

            return render_template("index.html",
                                   prediction=label,
                                   confidence=round(confidence*100,2),
                                   summary=summary,
                                   image_path=filepath)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
