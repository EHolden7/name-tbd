from sentence_transformers import SentenceTransformer, util
import requests

import PyPDF2
from cleantext import clean
from nlpretext import Preprocessor
from nlpretext.basic.preprocess import (normalize_whitespace, remove_punct, remove_eol_characters, remove_stopwords, lower_text)
from nlpretext.social.preprocess import remove_mentions, remove_hashtag, remove_emoji
from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt')
from scrapy.crawler import CrawlerProcess
from spiders.google_spider import GoogSpider, QuotesSpider

# https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
# https://huggingface.co/tasks/sentence-similarity

API_URL = 'https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2'
headers = {"Authorization": f"Bearer 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"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
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

liquidBiopsyArticle = extract_from_pdf('liquidBiopsyArticle.pdf')
liquidBiopsyPaper = extract_from_pdf('liquidBiopsy.pdf')
openDayArticle = extract_from_pdf('openDayArticle.pdf')

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

preprocessor = Preprocessor()
preprocessor.pipe(lower_text)
preprocessor.pipe(remove_mentions)
preprocessor.pipe(remove_hashtag)
preprocessor.pipe(remove_emoji)
preprocessor.pipe(remove_eol_characters)
#preprocessor.pipe(remove_stopwords, args={'lang': 'en'})
#preprocessor.pipe(remove_punct)
preprocessor.pipe(normalize_whitespace)

liquidBiopsyArticleClean = preprocessor.run(liquidBiopsyArticle)
liquidBiopsyPaperClean = preprocessor.run(liquidBiopsyPaper)
openDayArticleClean = preprocessor.run(openDayArticle)

dataClean = query(
    {
        "inputs": {
            "source_sentence": liquidBiopsyPaperClean,
            "sentences":[liquidBiopsyArticleClean,
                         openDayArticleClean]
        }
    })

# could get a whole of document score to decide which documents to spend the time doing the sliding window on

liquidBiopsyArticleSent = [x for x in sent_tokenize(liquidBiopsyArticleClean) if len(x) >= 10]
liquidBiopsyPaperSent = [x for x in sent_tokenize(liquidBiopsyPaperClean) if len(x) >= 10]
openDayArticleSent = [x for x in sent_tokenize(openDayArticleClean) if len(x) >= 10]

def sent_score(source, ref, threshold):
    matches = {}
    for i in range(len(source)):
        scores = query({"inputs": {"source_sentence": source[i],
                                  "sentences": ref}})
        #print(scores)
        temp = [{ref[x]: scores[x]} for x in range(len(scores)) if scores[x] > threshold] #or scores[x] == max(scores)??
        if len(temp) != 0:
            matches[source[i]] = temp
        #print(matches)
    print('done')
    return matches

scores = sent_score(liquidBiopsyArticleSent, liquidBiopsyPaperSent, 0.6)

# hard to know what the idea score threshold is
# should I conduct tests to choose? Should I get a better understanding of the theory and calculate some threshold?

## The current output is a dictionary with quote:list pairs, where the list contains quote:score pairs for quotes that match the other quote over some threshold score

# 1 We need the webcrawler to get our reference sources
# 2 We need to clean up the PDFs better
# 3 We need to have some better method for choosing 

if __name__ == '__main__':
    process = CrawlerProcess(
        settings={
            "FEEDS": {
                "items.json": {"format": "json"},
            },
        }
    )

    process.crawl(QuotesSpider)
    process.start()  # the script will block here until the crawling is finished