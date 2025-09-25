# Facial Recognition System 👤

###  Team Members
- Phase 1: Data Preparation – [اسم الشخص]
- Phase 2: EDA + Feature Building – [اسم الشخص]
- Phase 3: Model Training & Validation – [Mohamed & Mostafa]
- Phase 4: Deployment (Streamlit App) – [Nour]

---

##  Project Overview
Traditional security methods such as passwords, ID cards, or PINs are vulnerable to theft, forgery, and misuse.  
This project aims to build a **Facial Recognition System** that can reliably **identify and authenticate individuals** in real-time, even under variations in lighting, pose, or expression.  

The system is built using **LFW dataset** and deployed in a simple **Streamlit web app**.

---

##  Project Phases

###  Phase 1: Data Preparation
- Dataset used: **Labeled Faces in the Wild (LFW)**.
- Preprocessing applied:
  - Face detection and cropping.
  - Image resizing & normalization.


---

###  Phase 2: EDA + Feature Building
- Explored dataset distribution (number of images per identity).
- Applied **data augmentation** techniques (flipping, rotation, scaling).
- Extracted **face embeddings** using `face_recognition` library (HOG + CNN based encodings).

---

###  Phase 3: Model Training & Validation
- Stored embeddings & labels into a pickle file (`face_embeddings.pkl`).
- Used **distance-based matching** (face distance with tolerance threshold).
---

###  Phase 4: Deployment (Streamlit Web App)
- Built a simple **Streamlit interface** with two options:
  1. **Upload Image** → System verifies and shows identity (or "Non Verified").
  2. **Open Camera** → Real-time face recognition via webcam.
- Output: User name if verified, otherwise "Unknown / Non Verified".

---

##  Tech Stack
- **Python**
- **OpenCV**
- **face_recognition**
- **NumPy, Pickle**
- **Streamlit** (for deployment)

---

##  Demo Screenshots


---

##  Results & Conclusion
- Achieved reliable recognition performance on **LFW dataset**.  
- Real-time system works well for identity verification.  
- Future improvements:
  - Train with **VGGFace2** dataset for larger identity coverage.
  - Apply **FaceNet embeddings** for more robust matching.
