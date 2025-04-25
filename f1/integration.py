import logging
import nltk
from sentence_transformers import SentenceTransformer

from m1_t5 import M1_T5
from pdf_utils import pdfUtils
from vec_db import vecDB
from m2_rag import M2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataQuest:
    def __init__(self):
        logger.info("Starting the application.")
        self.questions = []
        self.m1 = M1_T5()
        self.pdf_file = ""
        self.pdf_utils = pdfUtils("")
        self.vecDB = vecDB()
        self.M2 = M2()
        
        resources = ['punkt', 'stopwords', 'wordnet', 'punkt_tab']
        for resource in resources:
            try:
                nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else f'corpora/{resource}')
            except LookupError:
                nltk.download(resource)
        

    def setPDFFile(self, file_path):
        self.pdf_file = file_path
        self.pdf_utils.updateFilePath(file_path)
        
    def getPDFFileInfo(self):
        return self.pdf_file
    
    def readPDFFile(self):
        text = self.pdf_utils.extractTextFromPdf()
        return text
    
    def readPDFMetaData(self):
        metaData = self.pdf_utils.getPdfMetadata()
        return metaData
    
    def getPDFSectionHeaders(self):
        structureHeaders = self.pdf_utils.extractSectionHeaders()
        return structureHeaders

    def getPDFStructuredText(self):
        structuredText = self.pdf_utils.extractStructuredText()
        return structuredText
    

    def createQs(self, context, difficulty, top):
        self.questions = self.m1.generateQuestion(context, difficulty, top)
        return self.questions
    
    def createNonRepeatingQs(self, questions, threshold):
        self.questions = self.m1.removeSimilarQuestions(questions, threshold)
        logger.info("Questions generated successfully.")
        return self.questions
    
    def setupVecDB(self):
        if not self.vecDB.setup_milvus():
            print("Failed to connect to Milvus")
            exit(1)

    def checkForCollection(self, collection):
        self.vecDB.checkForCollection(collection)
    
    def setupEnhancedCollection(self, embed_dim):
        return self.vecDB.setup_enhanced_collection(embed_dim=embed_dim)

    def clearCollection(self, collection):
        self.vecDB.clear_collection(collection)
    
    def ingest_document_improved(self, text, metadata, headers, collection):
        return self.vecDB.ingest_document_improved(text, metadata, headers, collection)

    def rag_query(self, question, collection, embedder):
        return self.M2.rag_query(question, collection, embedder)
        

if __name__ == "__main__":
    """
    Below code is only for testing. It is like a sandbox for the developer
    """

    file_path = r"Path to the PDF file"
    app = DataQuest()
    app.setupVecDB()
    app.checkForCollection("rag_docs")
    
    embedder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    collection = app.setupEnhancedCollection(embed_dim=768)

    app.clearCollection(collection)

    app.setPDFFile(file_path)
    text = app.readPDFFile()
    metadata = app.readPDFMetaData()
    headers = app.getPDFSectionHeaders()

    stats = app.ingest_document_improved(text, metadata, headers, collection)
    if stats:
        print(f"Ingestion stats: {stats}")

    top = 7
    qs = []
    
    textParagraphs = [para.strip() for para in text.split('\n\n') if para.strip()]

    qs =[]
    for t in textParagraphs:
        qs += (app.createQs(t, "easy", top))
    questions = app.createNonRepeatingQs(qs, 0.6)

    for i, question in enumerate(questions):
        print(f"{i + 1}. {question}")

    for i, question in enumerate(questions):
        ans = app.rag_query(question, collection, embedder)
        print(f"{i + 1}. {question}")
        print(f"Answer: {ans}")