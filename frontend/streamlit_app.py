import streamlit as st
import requests
from PIL import Image
import io
import plotly.graph_objects as go
import pandas as pd
import time
import numpy as np

st.set_page_config(
    page_title="CV Inference Demo",
    page_icon="🎯",
    layout="wide"
)

# API endpoint
API_URL = "http://localhost:8000"

st.title("🎯 Real-Time Computer Vision Inference")
st.markdown("Upload an image to get real-time classification results")

# Sidebar
with st.sidebar:
    st.header("Settings")
    
    model_type = st.radio(
        "Select Model",
        ["quantized", "original"],
        help="Quantized model offers faster inference with minimal accuracy loss"
    )
    
    st.markdown("---")
    st.header("Performance Metrics")
    
    # Placeholder for metrics
    metrics_placeholder = st.empty()

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=['jpg', 'jpeg', 'png', 'webp']
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Predict button
        if st.button("Run Inference", type="primary"):
            with st.spinner("Processing..."):
                # Prepare image for API
                img_bytes = io.BytesIO()
                image.save(img_bytes, format='PNG')
                img_bytes = img_bytes.getvalue()
                
                # Make prediction request
                files = {"file": ("image.png", img_bytes, "image/png")}
                params = {"model_type": model_type}
                
                start_time = time.time()
                response = requests.post(
                    f"{API_URL}/predict", 
                    files=files, 
                    params=params
                )
                total_time = (time.time() - start_time) * 1000
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Store in session state
                    st.session_state['last_prediction'] = result
                    st.session_state['total_time'] = total_time
                else:
                    st.error(f"Error: {response.text}")

with col2:
    st.subheader("Results")
    
    if 'last_prediction' in st.session_state:
        result = st.session_state['last_prediction']
        
        # Display metrics
        col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
        
        with col_metrics1:
            st.metric(
                "Inference Time", 
                f"{result['inference_time_ms']:.1f} ms"
            )
        
        with col_metrics2:
            st.metric(
                "Preprocessing", 
                f"{result['preprocessing_time_ms']:.1f} ms"
            )
        
        with col_metrics3:
            st.metric(
                "Total Time", 
                f"{st.session_state['total_time']:.1f} ms"
            )
        
        st.markdown("---")
        
        # Display predictions
        st.subheader("Top Predictions")
        
        predictions = result['predictions']
        
        # Create bar chart
        fig = go.Figure(data=[
           go.Bar(
               x=[p['probability'] for p in predictions],
               y=[p['class'] for p in predictions],
               orientation='h',
               marker=dict(
                  color=['rgba(255,75,75,0.8)' if i == 0 else 'rgba(46,204,113,0.8)' for i in range(len(predictions))],
                  line=dict(color='rgba(0,0,0,0.3)', width=1)
        )
    )
])
        fig.update_layout(
            title="Prediction Probabilities",
            xaxis_title="Confidence",
            yaxis_title="Class",
            height=300,
            showlegend=False,
            xaxis=dict(range=[0, 1])
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display as table
        df = pd.DataFrame(predictions)
        df['probability'] = df['probability'].apply(lambda x: f"{x:.2%}")
        st.dataframe(df, use_container_width=True)
        
        # Update sidebar metrics
        with st.sidebar:
            metrics_placeholder.metric(
                "Last Inference Latency",
                f"{result['inference_time_ms']:.1f} ms",
                delta=f"{result['model_used']} model"
            )
    
    else:
        st.info("Upload an image and click 'Run Inference' to see results")

# Model comparison section
st.markdown("---")
st.subheader("Model Performance Comparison")

if st.button("Run Benchmark", key="benchmark"):
    with st.spinner("Running benchmark..."):
        # Test both models
        models = ['quantized', 'original']
        latencies = []
        
        # Use a sample image for testing
        test_image = Image.new('RGB', (224, 224), color='red')
        img_bytes = io.BytesIO()
        test_image.save(img_bytes, format='PNG')
        img_bytes = img_bytes.getvalue()
        files = {"file": ("test.png", img_bytes, "image/png")}
        
        for model in models:
            latencies_model = []
            for _ in range(10):
                response = requests.post(
                    f"{API_URL}/predict", 
                    files=files, 
                    params={"model_type": model}
                )
                if response.status_code == 200:
                    result = response.json()
                    latencies_model.append(result['inference_time_ms'])
            
            latencies.append(np.mean(latencies_model))
        
        # Display comparison
        fig_compare = go.Figure(data=[
            go.Bar(
                name='Quantized',
                x=['Quantized'],
                y=[latencies[0]],
                marker_color='#2ecc71'
            ),
            go.Bar(
                name='Original',
                x=['Original'],
                y=[latencies[1]],
                marker_color='#ff4b4b'
            )
        ])
        
        fig_compare.update_layout(
            title="Model Latency Comparison",
            yaxis_title="Latency (ms)",
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig_compare, use_container_width=True)
        
        reduction = ((latencies[1] - latencies[0]) / latencies[1]) * 100
        st.success(f"Quantized model is {reduction:.1f}% faster!")
