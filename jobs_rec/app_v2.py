import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
import pickle
import os
from PIL import Image
import time

# Load models and data
with open('knn_model.pkl', 'rb') as f:
    knn_model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

project = pd.read_csv('/Users/harshavardhan/Downloads/archive (5)/linkedin_job_postings.csv')

# Function to recommend job titles based on search position
def job_title_recommendation(search_position):
    search_position_vector = vectorizer.transform([search_position])
    
    # Finding the nearest neighbors
    distances, indices = knn_model.kneighbors(search_position_vector)

    # Getting the recommended job titles and details
    recommended_jobs = []
    for i, j in enumerate(indices[0]):
        job_details = project.iloc[j]
        similarity_score = 1 - distances[0][i]  #This line converts distance to similarity score
        recommended_jobs.append((job_details, similarity_score))

    return recommended_jobs

# Load image
def load_image(image_file):
    img = Image.open(image_file)
    return img

# Set page layout with background image
def set_background_image(image):
    st.markdown(
        f"""
        <style>
            .reportview-container {{
                background: url(data:image/jpeg;base64,{image});
                background-size: cover;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )

def main():
    # Load background image
    background_image = load_image('image.png')
    set_background_image(background_image)

    st.title("JOB RECOMMENDATION SYSTEM 💰")
    search_position = st.text_input("Enter a search position:")

    if search_position:
        recommended_jobs = job_title_recommendation(search_position)
        # Display recommended jobs
        st.subheader("Recommended Jobs:")
        for job, score in recommended_jobs:
            st.write("Job Title:", job['job_title'])
            st.write("Company:", job['company'])
            st.write("Job Link:", job['job_link'])
            st.write("Search City:", job['search_city'])
            st.write("Search Country:", job['search_country'])
            st.write("Last Processed Time:", job['last_processed_time'])
            st.write("Job Level:", job['job_level'])
            st.write("Job Type:", job['job_type'])
            st.write("Similarity Score:", score)
            st.write("---")

    #st.write("Balloons are BOSDA!")

    while True:
        st.balloons()
        time.sleep(15)  # Adjust the sleep time as needed

if __name__ == "__main__":
    main()
