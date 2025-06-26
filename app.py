import streamlit as st
import torch
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# Configure page layout (Wide mode)
st.set_page_config(page_title="AI Fracture Detector", layout="wide")

# Load the trained YOLOv8 model
model_path = r"C:\AIfracture_detection\content\train4\weights\best.pt"
model = YOLO(model_path)

# Enhanced Fracture Details with Descriptions and Remedies
fracture_details = {
    0: {
        "name": "Hairline Fracture",
        "severity": "Mild",
        "description": """A hairline fracture, also called a stress fracture, is a tiny crack in the bone. 
        It usually occurs due to repetitive stress or overuse, rather than a sudden impact. These fractures are 
        common in weight-bearing bones such as the tibia, femur, or metatarsals. Since the fracture is not fully broken, 
        it often goes unnoticed for weeks until pain and swelling intensify. Though mild in nature, untreated stress 
        fractures can worsen over time.""",
        "remedy": """**Treatment & Recovery:**  
        - **Rest:** Avoid any weight-bearing activity that could worsen the fracture. Use crutches if necessary.  
        - **Ice Therapy:** Apply an ice pack wrapped in a cloth for 15-20 minutes every 3-4 hours to reduce swelling.  
        - **Compression & Elevation:** Wrapping the area with a soft bandage can help minimize swelling. Elevate the 
        affected limb while resting.  
        - **Pain Management:** Over-the-counter NSAIDs (e.g., ibuprofen) can help relieve pain. Avoid excessive use, 
        as some NSAIDs can slow down bone healing.  
        - **Diet & Supplements:** A calcium-rich diet, along with vitamin D, speeds up bone recovery.  
        - **Rehabilitation:** After healing, gradual strengthening exercises will help restore movement and prevent 
        re-injury.  
        - **Doctor Consultation:** If pain persists after 4-6 weeks, consult a specialist to check for non-healing fractures."""
    },
    1: {
        "name": "Partial Fracture",
        "severity": "Moderate",
        "description": """A partial fracture means the bone has cracked but has not completely broken into two separate 
        pieces. These fractures often occur due to moderate trauma, such as falling on an outstretched hand or twisting 
        an ankle. Unlike hairline fractures, partial fractures are more noticeable due to immediate swelling, tenderness, 
        and limited movement in the affected area.""",
        "remedy": """**Treatment & Recovery:**  
        - **Immobilization:** A cast or splint is necessary to prevent movement and allow proper bone healing.  
        - **Pain Control:** Prescription painkillers or NSAIDs can be used under medical supervision.  
        - **Physical Therapy:** After the cast is removed, light stretching and gradual movement exercises will restore 
        mobility.  
        - **Weight-bearing Precautions:** Avoid putting stress on the affected area until the bone has regained its strength.  
        - **Hydration & Nutrition:** Staying hydrated and maintaining a protein-rich diet helps in faster bone healing.  
        - **Follow-up X-rays:** Periodic X-rays may be required to ensure the fracture is healing properly.  
        - **Avoid Smoking & Alcohol:** Smoking slows down bone regeneration, and alcohol increases the risk of delayed healing."""
    },
    2: {
        "name": "Complete Fracture",
        "severity": "Severe",
        "description": """A complete fracture occurs when the bone is fully broken into two or more separate pieces. 
        This type of fracture is caused by a high-impact force, such as a car accident, sports injury, or a heavy fall. 
        Symptoms include severe pain, swelling, bruising, and an inability to move the affected limb. In some cases, the 
        broken bone may protrude through the skin, leading to an open fracture that requires emergency care.""",
        "remedy": """**Emergency Treatment & Recovery:**  
        - **Immediate Medical Attention:** Call for emergency help if the bone is visibly out of place or bleeding heavily.  
        - **Immobilization:** The affected limb should be kept stable with a splint to prevent further injury.  
        - **Surgery Requirement:** Severe fractures often require surgical intervention, such as inserting metal rods or 
        plates to align and stabilize the bone.  
        - **Hospitalization & Observation:** For complex fractures, a hospital stay may be needed to monitor healing.  
        - **Physical Therapy & Rehabilitation:** After healing, an extensive physical therapy program is needed to regain 
        strength and mobility.  
        - **Bone Stimulation Therapy:** In some cases, electrical stimulation is used to accelerate bone regeneration.  
        - **Long-Term Recovery:** Severe fractures may take several months to heal fully, and lifestyle adjustments may 
        be necessary to prevent re-injury."""
    }
}

# Function to detect fractures
def detect_fracture(image):
    results = model(image)
    for result in results:
        annotated_frame = result.plot()  # Overlay bounding boxes
    return annotated_frame, results

# Function to generate detailed summary & remedy
def generate_summary(results):
    summary = ""
    remedy = ""

    for result in results:
        for box in result.boxes:
            confidence = box.conf[0].item()
            class_id = int(box.cls[0].item())

            fracture_info = fracture_details.get(class_id, {"name": "Unknown", "severity": "Unknown", "description": "No data available.", "remedy": "Consult a doctor."})
            fracture_name = fracture_info["name"]
            severity = fracture_info["severity"]
            description = fracture_info["description"]
            remedy = fracture_info["remedy"]

            summary += f"### 🦴 {fracture_name}  \n"
            summary += f"**Severity:** {severity}  \n"
            summary += f"**Confidence Level:** {confidence:.2f}  \n\n"
            summary += f"📌 **Description:**  \n {description}  \n\n"

    if not summary:
        summary = "✅ No fractures detected."
        remedy = "🦴 Your bones appear healthy. Maintain good posture and a calcium-rich diet."

    return summary, remedy

# UI Layout
st.title("🦴 AI Fracture Detector")
st.write("Upload an X-ray image to detect fractures and receive medical recommendations.")

uploaded_file = st.file_uploader("📤 Upload an X-ray Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    image = np.array(image)

    annotated_image, results = detect_fracture(image)
    summary, remedy = generate_summary(results)

    # Layout - Three Equal Columns (Ensuring No Scrolling)
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.image(annotated_image, caption="📸 Detected Fracture", use_column_width=True)

    with col2:
        st.markdown("### 🏥 Diagnosis Summary")
        st.markdown(summary)

    with col3:
        st.markdown("### 🏥 Recommended Treatment")
        st.markdown(remedy)
