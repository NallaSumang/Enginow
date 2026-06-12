import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Ensure nltk resources are downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()
default_stopwords = set(stopwords.words('english'))

# Words that convey negation are critical for emotion/sentiment detection. 
# We remove them from the stopword list so they remain in the text.
negations = {'not', 'no', 'nor', 'dont', 'ain', 'aren', 'arent', 'couldn', 'couldnt', 
             'didn', 'didnt', 'doesn', 'doesnt', 'hadn', 'hadnt', 'hasn', 'hasnt', 
             'haven', 'havent', 'isn', 'isnt', 'mightn', 'mightnt', 'mustn', 'mustnt', 
             'needn', 'neednt', 'shan', 'shant', 'shouldn', 'shouldnt', 'wasn', 'wasnt', 
             'weren', 'werent', 'won', 'wont', 'wouldn', 'wouldnt', 'don', 'do'}

# Include punctuation-free versions of all stopwords
punct_free_stopwords = {word.translate(str.maketrans('', '', string.punctuation)) for word in default_stopwords}
all_stopwords = default_stopwords.union(punct_free_stopwords)

# Remove negations from stopwords
stop_words = all_stopwords - negations

def preprocess_text(text):
    """
    Cleans and preprocesses the input text.
    Steps:
    1. Lowercase text
    2. Remove special characters and punctuation
    3. Tokenization
    4. Stopwords removal
    5. Lemmatization
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Lowercasing
    text = text.lower()
    
    # 2. Removing special characters, numbers, and links
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    
    # 3. Tokenization
    tokens = word_tokenize(text)
    
    # 4. Stopwords removal & 5. Lemmatization
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    return " ".join(cleaned_tokens)
