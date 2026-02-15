import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import requests
import time
import json
from collections import deque
import numpy as np

# Page config
st.set_page_config(
    page_title="CV Inference Monitor",
    page_icon="📊",
    layout="wide"
)

# Initialize session state for metrics
if 'latency_history' not in st.session_state:
    st.session_state.latency_history = deque(maxlen=100)
if 'throughput_history' not in st.session_state:
    st.session_state.throughput_history = deque(maxlen=100)
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = deque(maxlen=1000)

# Sidebar
st.sidebar.title("📊 Monitoring Dashboard")
st.sidebar.markdown("---")

# API Configuration
api_url = st.sidebar.text_input("API URL", value="http://localhost:8000")
refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 1, 10, 2)

# Model info section
st.sidebar.markdown("---")
st.sidebar.subheader("Model Information")

try:
    response = requests.get(f"{api_url}/model/info")
    if response.status_code == 200:
        model_info = response.json()
        st.sidebar.json(model_info)
except:
    st.sidebar.error("Could not fetch model info")

# Main content
st.title("Real-Time Computer Vision Inference Monitor")

# Metrics row
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Current Latency",
        value=f"{np.mean(st.session_state.latency_history) if st.session_state.latency_history else 0:.1f} ms",
        delta=None
    )

with col2:
    st.metric(
        label="Throughput",
        value=f"{len(st.session_state.prediction_history)} predictions",
        delta=None
    )

with col3:
    if st.session_state.latency_history:
        p95 = np.percentile(st.session_state.latency_history, 95)
        st.metric(label="P95 Latency", value=f"{p95:.1f} ms")
    else:
        st.metric(label="P95 Latency", value="0 ms")

with col4:
    if st.session_state.latency_history:
        p99 = np.percentile(st.session_state.latency_history, 99)
        st.metric(label="P99 Latency", value=f"{p99:.1f} ms")
    else:
        st.metric(label="P99 Latency", value="0 ms")

# Charts
col1, col2 = st.columns(2)

with col1:
    st.subheader("Latency Over Time")
    fig_latency = go.Figure()
    if st.session_state.latency_history:
        fig_latency.add_trace(go.Scatter(
            y=list(st.session_state.latency_history),
            mode='lines',
            name='Latency',
            line=dict(color='blue', width=2)
        ))
        fig_latency.update_layout(
            xaxis_title="Time",
            yaxis_title="Latency (ms)",
            height=400
        )
    st.plotly_chart(fig_latency, use_container_width=True)

with col2:
    st.subheader("Prediction Distribution")
    if st.session_state.prediction_history:
        pred_df = pd.DataFrame(
            list(st.session_state.prediction_history),
            columns=['class', 'confidence']
        )
        fig_dist = px.histogram(
            pred_df, 
            x='class',
            title="Prediction Class Distribution",
            height=400
        )
        st.plotly_chart(fig_dist, use_container_width=True)

# Confidence distribution
st.subheader("Confidence Distribution")
if st.session_state.prediction_history:
    confidences = [p[1] for p in st.session_state.prediction_history]
    fig_conf = go.Figure()
    fig_conf.add_trace(go.Histogram(
        x=confidences,
        nbinsx=20,
        name='Confidence',
        marker_color='green'
    ))
    fig_conf.update_layout(
        xaxis_title="Confidence Score",
        yaxis_title="Count",
        height=300
    )
    st.plotly_chart(fig_conf, use_container_width=True)

# Recent predictions table
st.subheader("Recent Predictions")
if st.session_state.prediction_history:
    recent = list(st.session_state.prediction_history)[-10:]
    df_recent = pd.DataFrame(recent, columns=['Class', 'Confidence'])
    df_recent.index = range(len(df_recent))
    st.dataframe(df_recent, use_container_width=True)

# Auto-refresh logic
placeholder = st.empty()
while True:
    try:
        # Simulate getting new predictions (in real implementation, you'd query your metrics DB)
        # For demo, we'll generate random metrics
        new_latency = np.random.normal(35, 5)  # Simulated latency
        new_confidence = np.random.random()
        new_class = np.random.choice(model_info['classes'])
        
        st.session_state.latency_history.append(new_latency)
        st.session_state.prediction_history.append((new_class, new_confidence))
        
        time.sleep(refresh_rate)
        st.rerun()
        
    except Exception as e:
        st.error(f"Error refreshing data: {str(e)}")
        time.sleep(refresh_rate)