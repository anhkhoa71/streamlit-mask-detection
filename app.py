import streamlit as st
from utils import *
from PIL import Image
import gc
import torch

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

# Load models riêng biệt để tránh tốn RAM
@st.cache_resource(show_spinner=False)
def get_fasterrcnn():
    with st.spinner("🔄 Loading Faster R-CNN..."):
        return load_fasterrcnn()

@st.cache_resource(show_spinner=False)
def get_ssd300():
    with st.spinner("🔄 Loading SSD300..."):
        return load_ssd300()

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
        0.1, 1.0, 0.7, 0.5,
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
        0.1, 1.0, 0.4, 0.5,
        label_visibility="collapsed"
    )

st.markdown("<br>", unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Upload image",
    type=["jpg", "png", "jpeg"],
    label_visibility="collapsed"
)

if uploaded_file:
    st.markdown("<br>", unsafe_allow_html=True)
    
    detect = st.button("🔍 Start Detection", key="detect_btn")
    
    # Đọc và xử lý ảnh
    image = Image.open(uploaded_file)
    image = correct_orientation(image).convert("RGB")
    
    # Resize ảnh lớn để tiết kiệm RAM
    max_size = 1024
    if max(image.size) > max_size:
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    
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
            st.image(image, use_column_width=True)
    
    if detect:
        st.markdown("<br>", unsafe_allow_html=True)
        
        status_text = st.empty()
        
        # Load models chỉ khi cần
        status_text.markdown('<div class="status-text">🚀 Loading models...</div>', unsafe_allow_html=True)
        fasterrcnn_model = get_fasterrcnn()
        ssd300_model = get_ssd300()
        
        status_text.markdown('<div class="status-text">🚀 Running Faster R-CNN...</div>', unsafe_allow_html=True)
        fast = run_inference(image, fasterrcnn_model, conf_fast)
        
        status_text.markdown('<div class="status-text">🚀 Running SSD300...</div>', unsafe_allow_html=True)
        ssd = run_inference(image, ssd300_model, conf_ssd)
        
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
        
        status_text.empty()
        
        # Dọn dẹp memory ngay sau khi vẽ xong
        del fast, ssd
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()