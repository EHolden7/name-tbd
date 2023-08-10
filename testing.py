from sentence_transformers import SentenceTransformer, util
import requests
import json

# https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
# https://huggingface.co/tasks/sentence-similarity

API_URL = 'https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2'

def query(payload):
    response = requests.post(API_URL, json=payload)
    return response.json()

data = query(
    {
        "inputs": {
            "source_sentence": "Brain cancer is the leading cause of cancer-related death in children. Early detection and serial monitoring are essential for better therapeutic outcomes",
            "sentences":["This is an example sentence", 
             "Children with an aggressive form of brain cancer known as medulloblasoma may soon be able to have their cancer detected a lot more accuately",
             "Doctors may soon be able to better monitor brain cancer in children and potentially predict the tumour progress and therapy response, thanks to University of Queensland-led research, funded by the Children’s Hospital Foundation.",
             ]
        }
    })

