import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

# Page Config
st.set_page_config(
    page_title="Chest X-Ray Pneumonia Classifier",
    page_icon="🩺",
    layout="wide"
)

# UI COMPACT CSS
st.markdown("""
<style>
.block-container {
    padding-top: 0.8rem;
    padding-bottom: 0.5rem;
}
div[data-testid="stVerticalBlock"] {
    gap: 0.4rem;
}
img {
    margin-bottom: 0.3rem;
}
h1, h2, h3 {
    margin-top: 0.3rem;
    margin-bottom: 0.3rem;
}
.stAlert {
    padding: 0.4rem;
}
</style>
""", unsafe_allow_html=True)

# Keras Compatibility
try:
    import tf_keras as legacy_keras
    HAS_TF_KERAS = True
except Exception:
    legacy_keras = None
    HAS_TF_KERAS = False

try:
    import keras
    HAS_KERAS3 = True
except Exception:
    keras = None
    HAS_KERAS3 = False

# Legacy InputLayer Fix
def get_legacy_input_layer():
    base = keras.layers.InputLayer if HAS_KERAS3 else tf.keras.layers.InputLayer
    class LegacyInputLayer(base):
        def __init__(self, batch_shape=None, **kwargs):
            if batch_shape is not None:
                kwargs["batch_input_shape"] = batch_shape
            super().__init__(**kwargs)
    return LegacyInputLayer

# Robust Model Loader
@st.cache_resource(show_spinner=False)
def load_model_at(path):
    if not path or not os.path.exists(path):
        return None

    custom_objects = {"InputLayer": get_legacy_input_layer()}

    if HAS_KERAS3:
        try:
            return keras.models.load_model(path)
        except Exception:
            pass

    if HAS_TF_KERAS:
        try:
            return legacy_keras.models.load_model(path, custom_objects=custom_objects)
        except Exception:
            pass

    return tf.keras.models.load_model(path, compile=False, custom_objects=custom_objects)

# Model Input Shape
def input_spec(model):
    ish = model.input_shape
    h = ish[1] if ish[1] else 224
    w = ish[2] if ish[2] else 224
    c = ish[3] if len(ish) > 3 else 3
    return int(h), int(w), int(c)

# Image Preprocessing
def preprocess(img, spec):
    h, w, c = spec
    img = img.convert("L") if c == 1 else img.convert("RGB")
    img = img.resize((w, h))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if c == 1:
        arr = np.expand_dims(arr, -1)
    return np.expand_dims(arr, 0)

# Prediction Interpreter
def interpret(pred):
    pred = np.asarray(pred)

    if pred.ndim == 2 and pred.shape[1] == 1:
        p = float(pred[0, 0])
        return ("Pneumonia", p) if p >= 0.5 else ("Normal", 1 - p)

    if pred.ndim == 2 and pred.shape[1] == 2:
        idx = int(np.argmax(pred[0]))
        label = "Normal" if idx == 0 else "Pneumonia"
        return label, float(pred[0][idx])

    p = float(pred.squeeze())
    return ("Pneumonia", p) if p >= 0.5 else ("Normal", 1 - p)

#   GRAD-CAM
def get_gradcam(model, img_array):
    last_conv = None
    for layer in reversed(model.layers):
        if "conv" in layer.name.lower():
            last_conv = layer.name
            break

    if last_conv is None:
        return None

    grad_model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(last_conv).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)

 # handle list outputs
        if isinstance(preds, list):
            preds = preds[0]

        if preds.shape[-1] == 1:
            loss = preds[:, 0]
        else:
            class_idx = tf.argmax(preds[0])
            loss = preds[:, class_idx]

    grads = tape.gradient(loss, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_out[0]
    heatmap = tf.reduce_sum(conv_out * pooled, axis=-1)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8
    return heatmap

def overlay_gradcam(img, heatmap, alpha=0.4):
    if heatmap is None:
        return img

    heatmap = np.uint8(255 * heatmap)
    heatmap = Image.fromarray(heatmap).resize(img.size)
    heatmap = plt.cm.jet(np.array(heatmap) / 255.0)[:, :, :3]
    heatmap = Image.fromarray((heatmap * 255).astype(np.uint8))
    return Image.blend(img.convert("RGB"), heatmap, alpha)

# Session State
for k in ["cnn_model", "densenet_model", "vgg_model"]:
    st.session_state.setdefault(k, None)

# Sidebar
st.sidebar.title("Model & Settings")

cnn_path = st.sidebar.text_input("CNN model", "finalCNN.keras")
dn_path = st.sidebar.text_input("DenseNet model", "finaldensenet.keras")
vgg_path = st.sidebar.text_input("VGG16 model", "finalvgg16.keras")

def load_and_store(key, path):
    model = load_model_at(path)
    if model:
        st.session_state[key] = model
        st.sidebar.success(f"{key.upper()} loaded")
    else:
        st.sidebar.error("Load failed")

if st.sidebar.button("Load CNN"):
    load_and_store("cnn_model", cnn_path)

if st.sidebar.button("Load DenseNet"):
    load_and_store("densenet_model", dn_path)

if st.sidebar.button("Load VGG16"):
    load_and_store("vgg_model", vgg_path)

# Main UI
st.write("")
st.write("")

st.markdown("<h1 style='text-align:center;'>Chest X-Ray Pneumonia Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>CNN • DenseNet • VGG16 • Ensemble</p>", unsafe_allow_html=True)

#  Top image
st.image("images/image.gif", width='stretch')

# File uploader below 
uploaded = st.file_uploader(
    "Upload X-ray",
    type=["jpg", "png", "jpeg"]
)

# Prediction
if uploaded:
    img = Image.open(uploaded)
    left, right = st.columns([1, 1])

    with left:
        st.image(img, caption="Uploaded X-ray", width='stretch')

    ensemble = []

    def predict_block(model, name):
        spec = input_spec(model)
        arr = preprocess(img, spec)
        pred = model.predict(arr, verbose=0)
        label, conf = interpret(pred)

        st.markdown(f"**{name}**")
        if label == "Normal":
            st.success(f"NORMAL ({conf*100:.1f}%)")
            ensemble.append(1 - conf)
        else:
            st.error(f"PNEUMONIA ({conf*100:.1f}%)")
            ensemble.append(conf)

        with st.expander("Grad-CAM", expanded=False):
            cam = overlay_gradcam(img, get_gradcam(model, arr))
            st.image(cam, use_column_width=True)

    with right:
        if st.session_state.cnn_model:
            predict_block(st.session_state.cnn_model, "CNN")
        if st.session_state.densenet_model:
            predict_block(st.session_state.densenet_model, "DenseNet")
        if st.session_state.vgg_model:
            predict_block(st.session_state.vgg_model, "VGG16")

    if ensemble:
        avg = float(np.mean(ensemble))
        label = "Pneumonia" if avg >= 0.5 else "Normal"
        conf = avg if label == "Pneumonia" else 1 - avg
        st.markdown("### Ensemble Prediction")
        st.progress(int(conf * 100))
        st.success(f"{label.upper()} ({conf*100:.1f}%)")
