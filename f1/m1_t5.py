import pandas as pd
import numpy as np
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer    
import logging
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class M1_T5:

    def __init__(self):
        self.model = T5ForConditionalGeneration.from_pretrained("t5-small")
        self.tokenizer = T5Tokenizer.from_pretrained("t5-small")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

        self.model.load_state_dict(torch.load(r"C:\Users\Utham\Desktop\NEU\NLP\Project\f1\model_T5_epochs_8.pt",map_location=torch.device('cpu')))
        self.model.to(self.device)


    def generateQuestion(self, context, difficulty, top=5):
        if context == "":
            logger.error("There is no context given.")
            return
        else:
            if difficulty not in ["easy", "medium", "hard"]:
                logger.error("The difficulty level is invalid" + difficulty + ". Expected values: 'easy', 'medium', 'hard'")
                return 
            else:
                logger.info("Generating questions with the given context and difficulty.")
                return self._generate_questions(context, difficulty, top)
    

    def _generate_questions(self, context, difficulty, top):
        model = self.model
        tokenizer = self.tokenizer
        device = self.device

        input_text = (
        f"Generate {top} diverse and non-repetitive QUESTIONS of {difficulty} difficulty "
        f"based on the following context:\n{context}"
        f"Make sure that they are questions. If you are giving a question ending with '.',finish the sentence with '.Explain.'"
        f"Incase you cannot generate any more questions, return 'NO'.")

        input_encoding = tokenizer(
            input_text,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(device)

        output = model.generate(
            input_ids=input_encoding["input_ids"],
            attention_mask=input_encoding["attention_mask"],
            max_length=128,
            num_return_sequences=top,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.8,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3
        )

        questions = []
        
        for i in range(top):
            questions.append(tokenizer.decode(output[i], skip_special_tokens=True))

        logger.info("Questions generated successfully.")

        return questions
    
    
    def removeSimilarQuestions(self, generated_questions, threshold = 0.6):
        embedding_model = self.embedding_model

        embeddings = embedding_model.encode(generated_questions)
        cosine_similarities = cosine_similarity(embeddings)
        to_remove = set()

        for i in range(len(generated_questions)):
            for j in range(i + 1, len(generated_questions)):
                if cosine_similarities[i][j] > threshold:
                    to_remove.add(j)

        unique_questions = [q for i, q in enumerate(generated_questions) if i not in to_remove]
        logger.info("Questions generated successfully.")
        return unique_questions