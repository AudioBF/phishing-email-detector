import nltk
import os

def download_nltk_data():
    nltk_data_dir = os.path.join(os.path.expanduser('~'), 'nltk_data')
    if not os.path.exists(nltk_data_dir):
        os.makedirs(nltk_data_dir)
    
    # Download required NLTK data
    nltk.download('punkt', download_dir=nltk_data_dir)
    nltk.download('stopwords', download_dir=nltk_data_dir)

if __name__ == '__main__':
    download_nltk_data() 