import json
import pandas as pd
import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sklearn.model_selection import train_test_split
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import logging
import random
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

class QuestionGenerationDataset(Dataset):
    def __init__(self, contexts, questions, difficulties, tokenizer, max_length=512):
        self.contexts = contexts
        self.questions = questions
        self.difficulties = difficulties
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, idx):
        input_text = f"generate question: {self.contexts[idx]} difficulty: {self.difficulties[idx]}"

        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        target_encoding = self.tokenizer(
            self.questions[idx],
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = input_encoding["input_ids"].squeeze()
        attention_mask = input_encoding["attention_mask"].squeeze()
        target_ids = target_encoding["input_ids"].squeeze()
        target_ids[target_ids == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": target_ids,
            "context": self.contexts[idx],
            "question": self.questions[idx],
            "difficulty": self.difficulties[idx]
        }

def preprocess_dataset(file_path):
    """
    Preprocess the dataset from a JSON file.
    Returns lists of contexts, questions, answers, and difficulties.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON data: {e}")
        raise

    contexts = []
    questions = []
    difficulties = []

    for item in data:
        context = item["context"]
        for qa_pair in item["qa_pairs"]:
            contexts.append(context)
            questions.append(qa_pair["question"])
            difficulties.append(qa_pair["difficulty"])

    return contexts, questions, difficulties

def freeze_model_layers(model, num_layers_to_train=2):
    """
    Freeze all layers except the last num_layers_to_train layers in the T5 model.
    """
    all_params = list(model.named_parameters())

    encoder_params = [name for name, _ in all_params if 'encoder.block' in name]
    decoder_params = [name for name, _ in all_params if 'decoder.block' in name]

    encoder_layers = sorted(list(set([int(name.split('encoder.block.')[1].split('.')[0]) for name in encoder_params])))
    decoder_layers = sorted(list(set([int(name.split('decoder.block.')[1].split('.')[0]) for name in decoder_params])))

    encoder_layers_to_freeze = encoder_layers[:-num_layers_to_train] if len(encoder_layers) > num_layers_to_train else []
    decoder_layers_to_freeze = decoder_layers[:-num_layers_to_train] if len(decoder_layers) > num_layers_to_train else []

    for name, param in model.named_parameters():
        freeze = False

        for layer_num in encoder_layers_to_freeze:
            if f'encoder.block.{layer_num}.' in name:
                freeze = True
                break

        for layer_num in decoder_layers_to_freeze:
            if f'decoder.block.{layer_num}.' in name:
                freeze = True
                break

        if 'embed' in name or 'relative_attention_bias' in name:
            freeze = True
        if freeze:
            param.requires_grad = False

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

    return model

def train_model(train_dataloader, val_dataloader, model, tokenizer,
                num_epochs=3, learning_rate=5e-5, warmup_steps=100):
    """
    Train the T5 model for question generation.
    """
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0

        train_progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch in train_progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            total_train_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_progress_bar.set_postfix({"loss": loss.item()})

        avg_train_loss = total_train_loss / len(train_dataloader)

        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            val_progress_bar = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            for batch in val_progress_bar:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                total_val_loss += loss.item()
                val_progress_bar.set_postfix({"loss": loss.item()})

        avg_val_loss = total_val_loss / len(val_dataloader)

        logger.info(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model.pt")
            logger.info("Model saved!")

    return model

def calculate_bleu_score(references, candidates):
    """
    Calculate BLEU score for generated questions.
    """
    smoothie = SmoothingFunction().method1
    scores = []

    for ref, cand in zip(references, candidates):
        ref_tokens = nltk.word_tokenize(ref.lower())
        cand_tokens = nltk.word_tokenize(cand.lower())

        score = sentence_bleu([ref_tokens], cand_tokens, smoothing_function=smoothie)
        scores.append(score)

    return np.mean(scores)

def evaluate_model(model, test_dataloader, tokenizer):
    """
    Evaluate the model on the test set and calculate BLEU score.
    """
    model.eval()
    all_generated_questions = []
    all_reference_questions = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=128,
                num_beams=4,
                early_stopping=True
            )

            generated_questions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            all_generated_questions.extend(generated_questions)
            all_reference_questions.extend(batch["question"])

    bleu_score = calculate_bleu_score(all_reference_questions, all_generated_questions)
    logger.info(f"BLEU Score: {bleu_score:.4f}")

    n_examples = 5
    indices = np.random.choice(len(all_generated_questions), n_examples, replace=False)

    logger.info("Generated Question Examples:")
    for i in indices:
        logger.info(f"Context: {test_dataloader.dataset.contexts[i][:100]}...")
        logger.info(f"Difficulty: {test_dataloader.dataset.difficulties[i]}")
        logger.info(f"Original Question: {all_reference_questions[i]}")
        logger.info(f"Generated Question: {all_generated_questions[i]}")
        logger.info("-" * 50)

    return bleu_score, all_generated_questions, all_reference_questions



def generate_question(model, tokenizer, context, difficulty,top=5):
    """
    Generate a question based on a context, difficulty.
    """
    
    input_text = (
    f"Generate {top} diverse and non-repetitive questions of {difficulty} difficulty "
    f"based on the following context:\n{context}")

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
    no_repeat_ngram_size=3,
    early_stopping=True
    )


    questions = []

    for i in range(top):
        print(f"Generated Question {i+1}: {tokenizer.decode(output[i], skip_special_tokens=True)}")
        questions.append(tokenizer.decode(output[i], skip_special_tokens=True))

    question = tokenizer.decode(output[0], skip_special_tokens=True)

    return questions

file_path = "Json_merged_with_difficulty.json"
contexts, questions, difficulties = preprocess_dataset(file_path)

logger.info(f"Dataset loaded: {len(contexts)} examples")

train_contexts, test_contexts, train_questions, test_questions, train_difficulties, test_difficulties = train_test_split(
    contexts, questions, difficulties, test_size=0.2, random_state=42
)

train_contexts, val_contexts, train_questions, val_questions, train_difficulties, val_difficulties = train_test_split(
    train_contexts, train_questions, train_difficulties, test_size=0.1, random_state=42
)

logger.info(f"Train set: {len(train_contexts)} examples")
logger.info(f"Validation set: {len(val_contexts)} examples")
logger.info(f"Test set: {len(test_contexts)} examples")

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

model = freeze_model_layers(model, num_layers_to_train=2)
model = model.to(device)

train_dataset = QuestionGenerationDataset(
    train_contexts, train_questions, train_difficulties, tokenizer
)
val_dataset = QuestionGenerationDataset(
    val_contexts, val_questions, val_difficulties, tokenizer
)
test_dataset = QuestionGenerationDataset(
    test_contexts, test_questions, test_difficulties, tokenizer
)

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

model = train_model(
    train_dataloader, val_dataloader, model, tokenizer,
    num_epochs=8, learning_rate=5e-5
)

model.load_state_dict(torch.load("best_model.pt"))

bleu_score, generated_questions, reference_questions = evaluate_model(model, test_dataloader, tokenizer)

sample_context = test_contexts[0]

logger.info("Sample generations for different difficulty levels:")

for difficulty in ["easy", "medium", "hard"]:
    generated_question = generate_question(model, tokenizer, sample_context, difficulty,top=5)
    logger.info(f"Difficulty: {difficulty}")
    logger.info(f"Context (truncated): {sample_context[:100]}...")
    logger.info(f"Generated Question: {generated_question}")
    logger.info("-" * 50)

nltk.download('punkt_tab')

nltk.download('punkt')

sample_context_1 = ["Medical imaging has revolutionized the way we diagnose and treat diseases. Techniques such as X-rays, CT scans, MRI scans, and ultrasound allow us to visualize the inside of the body without surgery. X-rays are used to visualize bones and detect fractures. CT scans provide detailed images of the body's internal organs and tissues. MRI scans use magnetic fields and radio waves to create images of soft tissues, such as the brain and spinal cord. Ultrasound uses sound waves to create images of organs and tissues. These imaging techniques are essential for diagnosing a wide range of medical conditions, from broken bones to cancer. They also play a crucial role in guiding surgical procedures and monitoring the effectiveness of treatments. Advances in medical imaging technology are constantly improving the resolution and accuracy of these techniques, allowing for earlier and more accurate diagnoses. The development of new contrast agents is also enhancing the ability to visualize specific tissues and organs. Medical imaging continues to be an indispensable tool in modern medicine, providing invaluable information for diagnosis, treatment planning, and monitoring disease progression."]
for difficulty_1 in ["easy", "medium", "hard"]:
    generated_question = generate_question(model, tokenizer, sample_context_1, difficulty_1,top=3)
    print(f"Difficulty: {difficulty_1}")
    print(f"Context (truncated): {sample_context_1[:100]}...")
    print(f"Generated Question: {generated_question}")
    print("-" * 50)

output = []
model = T5ForConditionalGeneration.from_pretrained("t5-small")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model.load_state_dict(torch.load("/content/model_T5_epochs_8.pt",map_location=torch.device('cpu')))
model.to(device)
sample_context = ["Artificial Intelligence (AI) and Machine Learning (ML) are among the most influential technological advances of the 21st century. These fields involve the development of algorithms that allow machines to learn from data and make decisions. AI is widely used in industries such as healthcare, finance, and autonomous driving. For example, AI models can now diagnose diseases like cancer with accuracy rivaling that of human doctors. Natural Language Processing (NLP), a subfield of AI, powers virtual assistants like Siri and ChatGPT, enabling seamless human-computer interaction."]
for difficulty in ["easy", "medium", "hard"]:
  generated_question = generate_question(model, tokenizer, sample_context, difficulty,top=7)
  print(f"Difficulty: {difficulty}")
  print(f"Context (truncated): {sample_context[:100]}...")
  output.append(generated_question)
  print("-" * 50)

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def removeSimilarQuestions(generated_questions):

  embedding_model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
  embeddings = embedding_model.encode(generated_questions)


  cosine_similarities = cosine_similarity(embeddings)

  threshold = 0.6

  to_remove = set()

  for i in range(len(generated_questions)):
     for j in range(i + 1, len(generated_questions)):
          if cosine_similarities[i][j] > threshold:
              to_remove.add(j)

  unique_questions = [q for i, q in enumerate(generated_questions) if i not in to_remove]
  return unique_questions
print(f"output:{output}")
uniqueQuestions = removeSimilarQuestions(output)
print(f"Unique Quiestions :{uniqueQuestions}")