import os
from dotenv import load_dotenv

# Load .env variables
load_dotenv()

# Avoid TensorFlow dependency
os.environ["TRANSFORMERS_NO_TF"] = "1"

from langchain_huggingface import HuggingFaceEmbeddings

def get_embedder():
    """Return the HuggingFace embedding model from .env"""
    model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    return HuggingFaceEmbeddings(model_name=model_name)
