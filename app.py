import streamlit as st

st.title("Traffic Light Detection Demo")
st.write("Upload your output video (MP4 recommended):")

uploaded_file = st.file_uploader("Choose a video...", type=["mp4"])

if uploaded_file is not None:
    st.video(uploaded_file)
    st.success("Video uploaded and viewable below!")
else:
    st.info("No video uploaded yet.")
