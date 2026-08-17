# Purpose: Phase 5 Interactive Streamlit Web Application for Job Recommendation & Analytics

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(page_title="Job Market Recommendation & Analytics", layout="wide")
st.title("💼 Job Market Analysis & NLP Recommendation System")

# ---------------------------------------------------------
# Load Saved Artifacts & Models
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    # Local path on your computer
    local_dir = r"C:\Users\HUT2099\Desktop\internship documents\8th project 17 august 2026\docker_deployment_package"
    
    # Check if local directory exists; otherwise default to script folder (for Streamlit Cloud)
    if os.path.exists(local_dir):
        BASE_DIR = local_dir
    else:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    pkl_path = os.path.join(BASE_DIR, "processed_jobs.pkl")
    npy_path = os.path.join(BASE_DIR, "job_embeddings.npy")

    df = pd.read_pickle(pkl_path)
    embeddings = np.load(npy_path)
    transformer_model = SentenceTransformer('all-MiniLM-L6-v2')
    return df, embeddings, transformer_model

df, job_embeddings, transformer = load_artifacts()
# ---------------------------------------------------------
# Main Navigation Tabs
# ---------------------------------------------------------
tab1, tab2 = st.tabs(["🔍 Recommendation Engine", "📊 Market Analytics Dashboard"])

# =========================================================
# TAB 1: RECOMMENDATION ENGINE (Task 5)
# =========================================================
with tab1:
    st.header("Find Relevant Jobs with Transformer Embeddings")
    
    # 1. Search Interface
    user_query = st.text_input("Enter Job Title or Skill Keyword:", value="Data Scientist")
    
    # 2. Sidebar Filters (Work Type & Country)
    st.sidebar.header("Filter Criteria")
    
    work_type = st.sidebar.radio("Work Type Selector:", ["Both", "Budget", "Hourly"])
    min_budget = st.sidebar.number_input("Minimum Budget ($):", min_value=0, value=500, step=100)
    min_hourly = st.sidebar.number_input("Minimum Hourly Rate ($/hr):", min_value=0, value=20, step=5)
    
    # Extract unique countries for multi-select dropdown
    available_countries = sorted([str(c) for c in df['country'].dropna().unique() if c != ''])
    selected_countries = st.sidebar.multiselect("Country Filter:", available_countries)
    
    top_n = st.sidebar.slider("Number of Recommendations:", 5, 20, 10)

    if st.button("Generate Recommendations") or user_query:
        # Generate query embedding and calculate similarity
        query_embedding = transformer.encode([user_query])
        scores = cosine_similarity(query_embedding, job_embeddings).flatten()
        
        filtered_df = df.copy()
        filtered_df['Match_Score_%'] = (scores * 100).round(2)
        filtered_df = filtered_df.sort_values(by='Match_Score_%', ascending=False)
        
        # Apply Work Type / Budget / Hourly Filters
        if work_type == "Budget" and 'budget' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['budget'] >= min_budget]
        elif work_type == "Hourly" and 'hourly_high' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['hourly_high'] >= min_hourly]
        elif work_type == "Both":
            if 'budget' in filtered_df.columns and 'hourly_high' in filtered_df.columns:
                filtered_df = filtered_df[
                    (filtered_df['budget'] >= min_budget) | 
                    (filtered_df['hourly_high'] >= min_hourly)
                ]
                
        # Apply Country Filter
        if selected_countries and 'country' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['country'].isin(selected_countries)]
            
        # Display Results
        output_cols = [c for c in ['title', 'category', 'budget', 'hourly_high', 'country', 'Match_Score_%'] if c in filtered_df.columns]
        st.subheader(f"Top {top_n} Job Matches for '{user_query}'")
        st.dataframe(filtered_df[output_cols].head(top_n), use_container_width=True)

# =========================================================
# TAB 2: MARKET ANALYTICS DASHBOARD
# =========================================================
with tab2:
    st.header("Job Market Dashboard")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Postings by Category")
        if 'category' in df.columns:
            fig, ax = plt.subplots(figsize=(8, 4))
            df['category'].value_counts().plot(kind='bar', ax=ax, color='skyblue')
            ax.set_ylabel("Count")
            st.pyplot(fig)
            
    with col2:
        st.subheader("Budget Distribution")
        if 'budget' in df.columns:
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(df['budget'].dropna(), bins=30, kde=True, ax=ax, color='teal')
            ax.set_xlim(0, 5000)
            st.pyplot(fig)