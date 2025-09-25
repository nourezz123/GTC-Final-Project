import streamlit as st
import cv2
import numpy as np
import face_recognition
import pickle
from PIL import Image
import datetime

# -----------------------------
# CONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="🕵️‍♂️ Face Verification System",
    page_icon="🛡️",
    layout="wide",
)

EMBEDDING_FILE = "face_embeddings.pkl"

# -----------------------------
# LOAD EMBEDDINGS
# -----------------------------
@st.cache_resource
def load_embeddings():
    try:
        with open(EMBEDDING_FILE, 'rb') as f:
            data = pickle.load(f)
        return np.array(data['embeddings']), np.array(data['labels'])
    except FileNotFoundError:
        st.error("No embeddings file found. Please train the model first.")
        return np.array([]), np.array([])

# -----------------------------
# IMAGE PREPARATION
# -----------------------------
def prepare_image(image):
    if isinstance(image, Image.Image):
        image = np.array(image)
    if len(image.shape) == 2:  # grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:  # RGBA
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    return image

# -----------------------------
# FACE RECOGNITION
# -----------------------------
def recognize_face(image, known_embeddings, known_labels, tolerance=0.6, conf_threshold=75):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, boxes)

    if len(encodings) == 0:
        return "❌ Non Verified", None, 0

    for encoding in encodings:
        distances = face_recognition.face_distance(known_embeddings, encoding)
        best_idx = np.argmin(distances)
        best_distance = distances[best_idx]

        confidence = (1.0 - best_distance) * 100
        if best_distance < tolerance and confidence >= conf_threshold:
            name = known_labels[best_idx]
            return f"✅ Verified: {name}", name, confidence

    return "❌ Non Verified", None, 0

# -----------------------------
# APP START
# -----------------------------
embeddings, labels = load_embeddings()
if embeddings.size == 0:
    st.stop()

# -----------------------------
# SESSION STATE FOR HISTORY
# -----------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.image("logo.jpg", width=150)
st.sidebar.title("🔹 Face Verification")
st.sidebar.markdown("👋 Welcome to our Face Verification System")
st.sidebar.markdown("---")

mode = st.sidebar.radio("Select Mode:", ["Upload Image", "Use Camera"])
confidence_slider = st.sidebar.slider("Confidence Threshold", min_value=50, max_value=100, value=75, step=1)

st.sidebar.markdown("---")
st.sidebar.markdown("📅 Session Info:")
st.sidebar.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# -----------------------------
# MAIN PAGE
# -----------------------------
st.title("🛡️ Face Verification System")

if mode == "Upload Image":
    uploaded_file = st.file_uploader("📷 Upload a photo for verification", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        pil_image = Image.open(uploaded_file).convert("RGB")
        image_np = prepare_image(pil_image)

        status, name, confidence = recognize_face(image_np, embeddings, labels, conf_threshold=confidence_slider)

        st.image(image_np, caption="Uploaded Image", channels="RGB", use_column_width=True)
        if status.startswith("✅"):
            st.success(f"{status} | Confidence: {confidence:.2f}%")
            # Add to history
            st.session_state.history.append({
                "Name": name,
                "Confidence (%)": f"{confidence:.2f}",
                "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        else:
            st.error(status)

elif mode == "Use Camera":
    run = st.checkbox("Turn on Camera")
    FRAME_WINDOW = st.image([])

    cap = cv2.VideoCapture(0)
    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Camera not detected!")
            break

        status, name, confidence = recognize_face(frame, embeddings, labels, conf_threshold=confidence_slider)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        label_color = (0, 255, 0) if status.startswith("✅") else (255, 0, 0)
        cv2.putText(rgb, f"{status} ({confidence:.0f}%)", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, label_color, 2)

        FRAME_WINDOW.image(rgb)

        # Add to history if verified
        if status.startswith("✅") and name is not None:
            st.session_state.history.append({
                "Name": name,
                "Confidence (%)": f"{confidence:.2f}",
                "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

    cap.release()

# -----------------------------
# RECOGNITION HISTORY
# -----------------------------
st.markdown("---")
st.subheader("📜 Verification History")
if st.session_state.history:
    st.table(st.session_state.history)
else:
    st.info("No faces verified yet this session.")


