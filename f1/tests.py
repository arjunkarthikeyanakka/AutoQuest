import unittest
import os
from unittest.mock import MagicMock, patch
from sentence_transformers import SentenceTransformer
from integration import DataQuest

class TestDataQuest(unittest.TestCase):
    """Test cases for the DataQuest class"""
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        self.data_quest = DataQuest()
        self.test_pdf_path = r"C:\Users\Utham\Desktop\NEU\NLP\Project\f1\resources\mb.pdf"
        
        if not os.path.exists(self.test_pdf_path):
            with open(self.test_pdf_path, 'w') as f:
                f.write("Test PDF content")
    
    def tearDown(self):
        """Clean up after each test method"""
        pass
    
    def test_init(self):
        """Test class initialization"""
        self.assertEqual(self.data_quest.pdf_file, "")
        self.assertIsNotNone(self.data_quest.m1)
        self.assertIsNotNone(self.data_quest.pdf_utils)
        self.assertIsNotNone(self.data_quest.vecDB)
        self.assertIsNotNone(self.data_quest.M2)
    
    def test_set_pdf_file(self):
        """Test setting the PDF file path"""
        self.data_quest.setPDFFile(self.test_pdf_path)
        self.assertEqual(self.data_quest.pdf_file, self.test_pdf_path)
    
    def test_get_pdf_file_info(self):
        """Test getting PDF file info"""
        self.data_quest.setPDFFile(self.test_pdf_path)
        self.assertEqual(self.data_quest.getPDFFileInfo(), self.test_pdf_path)
    
    @patch('integration.pdfUtils')
    def test_read_pdf_file(self, mock_pdf_utils):
        """Test reading PDF file content"""
        mock_instance = mock_pdf_utils.return_value
        mock_instance.extractTextFromPdf.return_value = "Sample PDF content"
        
        self.data_quest.pdf_utils = mock_instance
        
        content = self.data_quest.readPDFFile()
        self.assertEqual(content, "Sample PDF content")
        mock_instance.extractTextFromPdf.assert_called_once()
    
    @patch('integration.pdfUtils')
    def test_read_pdf_metadata(self, mock_pdf_utils):
        """Test reading PDF metadata"""
        mock_instance = mock_pdf_utils.return_value
        mock_metadata = {"title": "Test PDF", "pages": 1}
        mock_instance.getPdfMetadata.return_value = mock_metadata
        
        self.data_quest.pdf_utils = mock_instance
        
        metadata = self.data_quest.readPDFMetaData()
        self.assertEqual(metadata, mock_metadata)
        mock_instance.getPdfMetadata.assert_called_once()
    
    @patch('integration.M1_T5')
    def test_create_qs(self, mock_m1_t5):
        """Test creating questions"""
        mock_instance = mock_m1_t5.return_value
        mock_questions = ["What is Python?", "How does Python work?"]
        mock_instance.generateQuestion.return_value = mock_questions
        
        self.data_quest.m1 = mock_instance
        
        context = "Python is a programming language"
        questions = self.data_quest.createQs(context, "medium", 2)
        
        self.assertEqual(questions, mock_questions)
        self.assertEqual(self.data_quest.questions, mock_questions)
        mock_instance.generateQuestion.assert_called_once_with(context, "medium", 2)
    
    @patch('integration.M1_T5')
    def test_create_non_repeating_qs(self, mock_m1_t5):
        """Test creating non-repeating questions"""
        mock_instance = mock_m1_t5.return_value
        input_questions = ["What is Python?", "How does Python work?", "What is Python?"]
        filtered_questions = ["What is Python?", "How does Python work?"]
        mock_instance.removeSimilarQuestions.return_value = filtered_questions
        
        self.data_quest.m1 = mock_instance
        
        questions = self.data_quest.createNonRepeatingQs(input_questions, 0.8)
        
        self.assertEqual(questions, filtered_questions)
        self.assertEqual(self.data_quest.questions, filtered_questions)
        mock_instance.removeSimilarQuestions.assert_called_once_with(input_questions, 0.8)
    
    @patch('integration.vecDB')
    def test_setup_enhanced_collection(self, mock_vec_db):
        """Test setting up enhanced collection"""
        mock_instance = mock_vec_db.return_value
        mock_collection = MagicMock()
        mock_instance.setup_enhanced_collection.return_value = mock_collection
        
        self.data_quest.vecDB = mock_instance
        
        collection = self.data_quest.setupEnhancedCollection(768)
        
        self.assertEqual(collection, mock_collection)
        mock_instance.setup_enhanced_collection.assert_called_once_with(embed_dim=768)
    
    @patch('integration.vecDB')
    def test_clear_collection(self, mock_vec_db):
        """Test clearing collection"""
        mock_instance = mock_vec_db.return_value
        mock_instance.clear_collection.return_value = True
        
        self.data_quest.vecDB = mock_instance
        
        mock_collection = MagicMock()
        result = self.data_quest.clearCollection(mock_collection)
        
        self.assertTrue(result is None)
        mock_instance.clear_collection.assert_called_once_with(mock_collection)
    
    @patch('integration.M2')
    def test_rag_query(self, mock_m2):
        """Test RAG query"""
        mock_instance = mock_m2.return_value
        mock_answer = "Python is a programming language"
        mock_instance.rag_query.return_value = mock_answer
        
        self.data_quest.M2 = mock_instance
        
        question = "What is Python?"
        mock_collection = MagicMock()
        mock_embedder = MagicMock()
        
        answer = self.data_quest.rag_query(question, mock_collection, mock_embedder)
        
        self.assertEqual(answer, mock_answer)
        mock_instance.rag_query.assert_called_once_with(question, mock_collection, mock_embedder)
    
    @patch('integration.pdfUtils')
    def test_get_pdf_section_headers(self, mock_pdf_utils):
        """Test getting PDF section headers"""
        mock_instance = mock_pdf_utils.return_value
        mock_headers = ["Introduction", "Methods", "Results"]
        mock_instance.extractSectionHeaders.return_value = mock_headers
        
        self.data_quest.pdf_utils = mock_instance
        headers = self.data_quest.getPDFSectionHeaders()
        
        self.assertEqual(headers, mock_headers)
        mock_instance.extractSectionHeaders.assert_called_once()

    @patch('integration.pdfUtils')
    def test_get_pdf_structured_text(self, mock_pdf_utils):
        """Test getting PDF structured text"""
        mock_instance = mock_pdf_utils.return_value
        mock_structured_text = [{"text": "Sample", "page": 1, "section": "Intro"}]
        mock_instance.extractStructuredText.return_value = mock_structured_text
        
        self.data_quest.pdf_utils = mock_instance
        structured_text = self.data_quest.getPDFStructuredText()
        
        self.assertEqual(structured_text, mock_structured_text)
        mock_instance.extractStructuredText.assert_called_once()

    def test_read_pdf_file_nonexistent(self):
        """Test reading a non-existent PDF"""
        non_existent_path = r"C:\path\to\nonexistent.pdf"
        self.data_quest.setPDFFile(non_existent_path)
        
        with self.assertRaises(Exception):
            self.data_quest.readPDFFile()
        

    @patch('integration.vecDB')
    def test_clear_collection(self, mock_vec_db):
        """Test clearing collection"""
        mock_instance = mock_vec_db.return_value
        mock_instance.clear_collection.return_value = True
        
        self.data_quest.vecDB = mock_instance
        mock_collection = MagicMock()
        
        result = self.data_quest.clearCollection(mock_collection)
        self.assertTrue(result)

        mock_instance.clear_collection.assert_called_once_with(mock_collection)

    @patch('integration.vecDB')
    def test_setup_vec_db_success(self, mock_vec_db):
        """Test successful vector DB setup"""
        mock_instance = mock_vec_db.return_value
        mock_instance.setup_milvus.return_value = True
        
        self.data_quest.vecDB = mock_instance
        
        self.data_quest.setupVecDB()
        
        mock_instance.setup_milvus.assert_called_once()

    @patch('integration.vecDB')
    def test_setup_vec_db_failure(self, mock_vec_db):
        """Test failed vector DB setup"""
        mock_instance = mock_vec_db.return_value
        mock_instance.setup_milvus.return_value = False
        
        self.data_quest.vecDB = mock_instance
        
        with patch('sys.exit') as mock_exit:
            self.data_quest.setupVecDB()
            mock_exit.assert_called_once_with(1)
        
        mock_instance.setup_milvus.assert_called_once()

    @patch('integration.vecDB')
    def test_check_for_collection(self, mock_vec_db):
        """Test checking for collection"""
        mock_instance = mock_vec_db.return_value
        
        self.data_quest.vecDB = mock_instance
        collection_name = "test_collection"
        
        self.data_quest.checkForCollection(collection_name)
        
        mock_instance.checkForCollection.assert_called_once_with(collection_name)

if __name__ == '__main__':
    unittest.main()