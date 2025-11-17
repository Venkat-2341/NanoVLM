import streamlit as st
from inference import predict

st.set_page_config(page_title="NanoVLM Inference", layout="centered")

st.title("NanoVLM — Multi-task Vision-Language Model")
st.write("Upload an image and ask a question.")


sample_questions = [
    "How many people are in this image?",
    "Rank the body parts from hottest to coldest.",
]

selected_question = st.selectbox(
    "Choose a sample question (optional):",
    sample_questions
)

question = st.text_input("Or type your own question:")

if selected_question != "-- Select --":
    question = selected_question


uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    st.image(uploaded, caption="Uploaded Image", use_column_width=True)

# Run inference
if uploaded is not None and question.strip() != "":
    if st.button("Run Inference"):
        # Save uploaded file to temp path
        temp_path = "temp_input.jpg"
        with open(temp_path, "wb") as f:
            f.write(uploaded.getbuffer())

        with st.spinner("Thinking..."):
            answer = predict(temp_path, question)

        st.subheader("🧠 Model Answer")
        st.success(answer)

elif uploaded is not None and question.strip() == "":
    st.warning("Please select or type a question.")
