import os

# Combined app.py: Includes both application logic and auto-downloading for the NLP model
app_code = """
import streamlit as st
import PyPDF2
import spacy
import pandas as pd
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. SETTINGS & MODEL LOADING ---
st.set_page_config(page_title='AI Resume Screener', page_icon='🔍', layout='wide')

@st.cache_resource
def load_nlp_model():
    \"\"\"Ensures the SpaCy model is available in the environment.\"\"\"
    model_name = 'en_core_web_sm'
    try:
        return spacy.load(model_name)
    except OSError:
        # Download the model if not found (crucial for Cloud deployment)
        os.system(f'python -m spacy download {model_name}')
        return spacy.load(model_name)

nlp = load_nlp_model()

# --- 2. HELPER FUNCTIONS ---
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        content = page.extract_text()
        if content: text += content
    return text

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\\s+', ' ', text)
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return ' '.join(tokens)

# --- 3. UI LAYOUT ---
st.title('🤖 AI-Powered Resume Ranking System')
st.markdown('### Project 3: NLP Deployment')

with st.sidebar:
    st.header('Settings')
    jd_input = st.text_area('Target Job Description:', height=200, placeholder='Paste your JD here...')
    uploaded_files = st.file_uploader('Upload Candidate Resumes (PDF)', type=['pdf'], accept_multiple_files=True)

# --- 4. EXECUTION LOGIC ---
if st.button('🚀 Rank Candidates'):
    if jd_input and uploaded_files:
        with st.spinner('Analyzing candidates...'):
            resumes_data = []
            for file in uploaded_files:
                text = extract_text_from_pdf(file)
                resumes_data.append({'name': file.name, 'text': text})

            # Process text
            processed_jd = preprocess_text(jd_input)
            processed_resumes = [preprocess_text(r['text']) for r in resumes_data]

            # TF-IDF & Cosine Similarity
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([processed_jd] + processed_resumes)
            
            # First row is JD, subsequent rows are resumes
            scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

            # Presentation
            results = pd.DataFrame({
                'Candidate Name': [r['name'] for r in resumes_data],
                'Match Percentage (%)': [round(s * 100, 2) for s in scores]
            }).sort_values(by='Match Percentage (%)', ascending=False)

            st.subheader('📊 Top Matching Candidates')
            st.dataframe(results, use_container_width=True)
            
            if not results.empty:
                st.success(f'**Best Fit Identified:** {results.iloc[0]["Candidate Name"]}')
    else:
        st.warning('Please provide a Job Description and at least one Resume.')
"""

with open('app.py', 'w') as f:
    f.write(app_code.strip())

# Updated requirements.txt (No tensorflow to avoid errors)
requirements_content = """
streamlit
PyPDF2
spacy
pandas
scikit-learn
"""

with open('requirements.txt', 'w') as f:
    f.write(requirements_content.strip())

print('✅ Combined app.py and requirements.txt are ready for download.')
