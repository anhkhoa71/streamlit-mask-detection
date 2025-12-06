import streamlit as st
from utils import *
from PIL import Image
import concurrent.futures

st.set_page_config(
    page_title="Mask Detection",
    layout="wide",
    initial_sidebar_state="collapsed"
)

@st.cache_resource
def load_css():
    with open("style.css") as f:
        return f"<style>{f.read()}</style>"

st.markdown(load_css(), unsafe_allow_html=True)

st.markdown('<div class="main-wow">Mask Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

class_names = ["background", "with_mask", "without_mask", "mask_weared_incorrect"]

@st.cache_resource(show_spinner=False)
def load_models():
    with st.spinner("🔄 Loading AI models..."):
        fasterrcnn = load_fasterrcnn()
        ssd300 = load_ssd300()
    return fasterrcnn, ssd300

fasterrcnn_model, ssd300_model = load_models()

st.markdown("<br>", unsafe_allow_html=True)
colA, colB = st.columns(2)

with colA:
    st.markdown("""
        <div class="slider-container">
            <div class="slider-title"> Fast RCNN Confidence </div>
        </div>
        """, unsafe_allow_html=True)
    
    conf_fast = st.slider(
    "Confidence Fast RCNN",
    0.1, 1.0, 0.7, 0.05,
    label_visibility="collapsed"
)

with colB:
    st.markdown("""
        <div class="slider-container">
            <div class="slider-title"> SSD300 Confidence </div>
        </div>
        """, unsafe_allow_html=True)
    
    conf_ssd = st.slider(
    "Confidence SSD", 
    0.1, 1.0, 0.5,
    label_visibility="collapsed"
)
st.markdown("<br>", unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Upload image",
    type=["jpg", "png"],
    label_visibility="collapsed"
)



if uploaded_file:
    st.markdown("<br>", unsafe_allow_html=True)
    
    detect = st.button("🔍 Start Detection", key="detect_btn", width="stretch")
    
    image = Image.open(uploaded_file)
    image = correct_orientation(image).convert("RGB")
    
    if not detect:    
        st.markdown(
            """
            <div class='preview-container'>
                <div class='preview-label'>Preview Image</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        col_center = st.columns([1, 3, 1])[1]
        with col_center:
            st.image(image)
    
    if detect:
        st.markdown("<br>", unsafe_allow_html=True)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.markdown('<div class="status-text">🚀 Running inference...</div>', unsafe_allow_html=True)
        progress_bar.progress(20)
        
        def run_fast():
            return run_inference(image, fasterrcnn_model, conf_fast)
        
        def run_ssd():
            return run_inference(image, ssd300_model, conf_ssd)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_fast = executor.submit(run_fast)
            future_ssd = executor.submit(run_ssd)
            
            progress_bar.progress(50)
            
            fast = future_fast.result()
            progress_bar.progress(75)
            
            ssd = future_ssd.result()
            progress_bar.progress(100)
        
        status_text.markdown('<div class="status-text">✅ Detection completed!</div>', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class='model-card'>
                <div class='model-header'>
                    <span class='model-name'>Faster R-CNN</span>
                    <span class='model-time'>⚡ {fast['infer_time']:.3f}s</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            draw_boxes(fast["img_tensor"], fast["outputs"], "Faster R-CNN", class_names)
        
        with col2:
            st.markdown(f"""
            <div class='model-card'>
                <div class='model-header'>
                    <span class='model-name'>SSD300</span>
                    <span class='model-time'>⚡ {ssd['infer_time']:.3f}s</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            draw_boxes(ssd["img_tensor"], ssd["outputs"], "SSD300", class_names)
        
        progress_bar.empty()
        status_text.empty()