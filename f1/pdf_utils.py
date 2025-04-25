import fitz     
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class pdfUtils:
    def __init__(self, pdf_path):
        """
        Initialize with the path to the PDF file.
        """
        self.pdf_path = pdf_path
    
    def updateFilePath(self, file_path):
        """
        Update the PDF file path.
        """
        self.pdf_path = file_path
    
    def extractTextFromPdf(self):
        """
        Extracts and returns the full text content from the PDF file,
        with page separators for context.
        """
        pdf_path = self.pdf_path
        logger.info("Attempting to read the file: " + pdf_path)
        try:
            doc = fitz.open(pdf_path)
            full_text = ""
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                page_text = page.get_text("text")
                
                full_text += f"\n\n--- Page {page_num+1} ---\n\n"
                full_text += page_text
            
            logger.info("File data extracted Successfully")
            return full_text
            
        except Exception as e:
            logger.exception(f"Error extracting text from PDF: {e}")
            return ""
        


    def getPdfMetadata(self):
        """
        Extracts and returns basic metadata from the PDF file,
        such as title, author, subject, keywords, and total page count.
        """
        pdf_path = self.pdf_path
        logger.info("Attempting to gett metadata from the file: " + pdf_path)
        try:
            doc = fitz.open(pdf_path)
            metadata = {
                "title": doc.metadata.get("title", ""),
                "author": doc.metadata.get("author", ""),
                "subject": doc.metadata.get("subject", ""),
                "keywords": doc.metadata.get("keywords", ""),
                "total_pages": doc.page_count
            }
            
            if not metadata["title"]:
                metadata["title"] = os.path.basename(pdf_path).replace(".pdf", "")
            
            logger.info("File metadata extracted Successfully")   
            return metadata
        except Exception as e:
            logger.exception(f"Error extracting PDF metadata: {e}")
            return {
                "title": os.path.basename(pdf_path).replace(".pdf", ""),
                "author": "",
                "subject": "",
                "keywords": "",
                "total_pages": 0
            }


    def extractSectionHeaders(self):
        """
        Identifies and returns potential section headers in the PDF
        based on font size and bold style heuristics.
        """
        pdf_path = self.pdf_path
        logger.info("Attempting to extract section headers from the file: " + pdf_path)
        try:
            doc = fitz.open(pdf_path)
            headers = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                blocks = page.get_text("dict")["blocks"]
                
                for block in blocks:
                    if "lines" in block:
                        for line in block["lines"]:
                            if "spans" in line:
                                for span in line["spans"]:
                                    is_bold = span.get("font", "").lower().find("bold") >= 0
                                    font_size = span.get("size", 0)
                                    text = span.get("text", "").strip()
                                    
                                    if (is_bold or font_size > 12) and text and len(text) < 100:
                                        headers.append({
                                            "text": text,
                                            "page": page_num + 1,
                                            "font_size": font_size,
                                            "is_bold": is_bold
                                        })
            
            return headers
        except Exception as e:
            logger.exception(f"Error extracting section headers: {e}")
            return []
    
    def extractStructuredText(self):
        """
        Extracts structured text from the PDF, organizing each text span
        by its section header, font properties, and page number.
        """
        pdf_path = self.pdf_path
        logger.info("Attempting to extract structured text from the file: " + pdf_path)
        try:
            doc = fitz.open(pdf_path)
            structured_content = []
            
            current_page = 1
            current_section = ""
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                blocks = page.get_text("dict")["blocks"]
                
                for block in blocks:
                    if "lines" in block:
                        for line in block["lines"]:
                            if "spans" in line:
                                for span in line["spans"]:
                                    is_bold = span.get("font", "").lower().find("bold") >= 0
                                    font_size = span.get("size", 0)
                                    text = span.get("text", "").strip()
                                    
                                    if (is_bold or font_size > 12) and text and len(text) < 100:
                                        current_section = text
                                    
                                    if text:
                                        structured_content.append({
                                            "text": text,
                                            "page": page_num + 1,
                                            "section": current_section,
                                            "is_bold": is_bold,
                                            "font_size": font_size
                                        })
            
            return structured_content
        except Exception as e:
            logger.exception(f"Error extracting structured text: {e}")
            return []