import nltk
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_nltk_data():
    try:
        # Use a directory within the project for NLTK data
        nltk_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "nltk_data")
        if not os.path.exists(nltk_data_dir):
            os.makedirs(nltk_data_dir)
            logger.info(f"Created NLTK data directory at {nltk_data_dir}")
        
        # Add the directory to NLTK's path
        nltk.data.path.append(nltk_data_dir)
        
        # Download required NLTK data
        logger.info("Downloading NLTK resources...")
        nltk.download('punkt', download_dir=nltk_data_dir)
        nltk.download('stopwords', download_dir=nltk_data_dir)
        logger.info("NLTK resources downloaded successfully")
        
    except Exception as e:
        logger.error(f"Error downloading NLTK data: {e}")
        raise

if __name__ == '__main__':
    download_nltk_data() 