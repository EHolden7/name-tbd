from sentence_transformers import SentenceTransformer, util
import requests
import json
import PyPDF2

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

## data = [0.050691038370132446, 0.6501005291938782, 0.7971435189247131]


## Extract text from source article
def extract_from_pdf(file_name):
    """"""
    # creating a pdf file object
    pdfFileObj = open(file_name, 'rb')

    pdf_text = ""
    
    # creating a pdf reader object
    pdfReader = PyPDF2.PdfReader(pdfFileObj)
    
    for i in range(len(pdfReader.pages)):
    
        # creating a page object
        pageObj = pdfReader.pages[i]
        
        # extracting text from page
        pdf_text = pdf_text + pageObj.extract_text()
    
    # closing the pdf file object
    pdfFileObj.close()

    return pdf_text

liquidBiopsyArticle = extract_from_pdf('name-tbd\liquidBiopsyArticle.pdf')
liquidBiopsyPaper = extract_from_pdf('name-tbd\liquidBiopsy.pdf')
openDayArticle = extract_from_pdf('name-tbd\openDayArticle.pdf')

data = query(
    {
        "inputs": {
            "source_sentence": liquidBiopsyPaper,
            "sentences":[liquidBiopsyArticle,
                         openDayArticle]
        }
    })

## data = [0.4563927948474884, 0.026159482076764107]
# despite the tragic formatting it's still getting a significantly higher similarity score for the relevant article
# obviously more testing would be needed, in particular for adjacent areas of science
# Could potentially be used 'as is' to provide a confidence score for whether an article is based in truth,
# but the real value would come from being to identify what in the article is accurate and what isn't

# break the individual blocks of text into more valuable individual sentences
# perform semantic similarity on all pairs in source and reference
# scores greater than x are considered a good match and likely indicate accuracy, lack of indicates inaccurate information