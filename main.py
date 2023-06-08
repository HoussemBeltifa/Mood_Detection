import streamlit as st
from PIL import Image
from ml import *
from streamlit_extras.let_it_rain import rain




st.image("images/mood.png",caption="",width=680)
st.title("Mood Detection")
st.subheader("Created by Mohamed Houssem Beltifa")
st.write("Note : the accuracy of the model is just 50% due to the limited data")

st.markdown("--------------------------------------------------")
st.header("Please insert an image of your face ")

uploaded_file = st.file_uploader("Choose a file", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("Classifying...")
    label, score = predict(image)
    st.write(f"Your Mood is most likely {label}, with a score of : {score*100:.2f} %")
    if label in ['happy', 'neutral','surprised']:
        st.balloons()
    elif label in ['angry', 'fearful', 'sad']:
        st.snow()
    elif label in ['disgusted']:
        rain(
            emoji="🤢",
            font_size=54,
            falling_speed=5,
            animation_length=10,
            )
