app_code = """
import streamlit as st
import PyPDF2
import spacy
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page configuration
st.set_page_config(page_title='AI Resume Screener', page_icon='🔍', layout='wide')

# Load NLP model with caching for performance
@st.cache_resource
def load_nlp():
    try:
        return spacy.load('en_core_web_sm')
    except:
        import os
        os.system('python -m spacy download en_core_web_sm')
        return spacy.load('en_core_web_sm')

nlp = load_nlp()

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

st.title('🤖 AI-Powered Resume Ranking System')
st.info('Upload candidate resumes and provide a job description to find the best fit.')

# Sidebar Inputs
with st.sidebar:
    st.header('Configuration')
    jd_input = st.text_area('Target Job Description:', height=250, placeholder='Paste JD here...')
    uploaded_files = st.file_uploader('Upload Resumes (PDF only)', type=['pdf'], accept_multiple_files=True)

# Main Execution
if st.button('🚀 Rank Candidates'):
    if jd_input and uploaded_files:
        with st.spinner('Analyzing resumes...'):
            resumes_data = []
            for file in uploaded_files:
                text = extract_text_from_pdf(file)
                resumes_data.append({'name': file.name, 'text': text})

            # NLP Pipeline
            processed_jd = preprocess_text(jd_input)
            processed_resumes = [preprocess_text(r['text']) for r in resumes_data]

            # Vectorization and Similarity
            all_texts = [processed_jd] + processed_resumes
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(all_texts)
            
            # Similarity of JD (index 0) vs Resumes (index 1+)
            scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

            # Prepare Results
            results = pd.DataFrame({
                'Candidate Name': [r['name'] for r in resumes_data],
                'Match Score (%)': [round(s * 100, 2) for s in scores]
            }).sort_values(by='Match Score (%)', ascending=False)

            st.subheader('📊 Top Matches')
            st.dataframe(results, use_container_width=True)
            
            # Visual highlight
            top_candidate = results.iloc[0]['Candidate Name']
            st.success(f'**Recommended Candidate:** {top_candidate}')
    else:
        st.warning('Please provide both the Job Description and Resume files.')
"""

with open('app.py', 'w') as f:
    f.write(app_code.strip())

print('✅ Production-ready app.py created. Download it from the sidebar.')
