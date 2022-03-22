import cv2
import streamlit as st
import tensorflow as tf
from tensorflow import keras

def get_model():
    model = keras.applications.MobileNetV2(include_top=True, weights="imagenet")
    model.trainable= False
    return model

def get_decoder():
    decode_predictions = keras.applications.mobilenet_v2.decode_predictions
    return decode_predictions

def get_preprocessor():
    def func(image):
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, (224, 224))
        image = keras.applications.mobilenet_v2.preprocess_input(image)
        image = tf.expand_dims(image, axis=0)
        return image
    
    return func

class Classfier:
    def __init__(self, top_k=5):
        self.top_k = top_k
        self.model = get_model()
        self.decode_predictions = get_decoder()
        self.preprocessor = get_preprocessor

    def predict(self, image):
        image = self.preprocessor(image)
        probs = self.model.predict(image)
        result = self.decode_predictions(probs, top = self.top_k)
        return result

def main():
    st.markdown("# Image Classification app using Streamlit")
    st.markdown("model = MobileNetV2")

    device = user_input = st.text_input("input your video/camera device", "0")
    if device.isnumeric():
        device = int(device)
    cap = cv2.VideoCapture(device)
    classifier = Classifier(top_k=5)
    label_names_st = st.empty()
    scores_st = st.empty()
    image_loc = st.empty()

    while cap.isOpened():
        __, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = classifier.predict(frame)
        labels = []
        scores = []
        for (_, label, prob) in result[0]:
            labels.append(f"{label: <16}")
            s = f"{100*prob:.2f}[%]"
            scores.append(f"{s: <16}")
        label_names_st.text(",".join(scores))
        image_loc.image(frame)
        if cv2.waitKey & 0xFF == ord("q"):
            break
    
      cap.release()

    if __name__=="___main__":
        main()






