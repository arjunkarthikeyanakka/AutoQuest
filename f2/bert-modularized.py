from transformers import BertTokenizer, EncoderDecoderModel, BertConfig
import torch
import json
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm

def initialize_model_and_tokenizer():
    """
    Initialize and configure the BERT encoder-decoder model and tokenizer.
    
    Returns:
        tuple: Configured tokenizer and model
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    special_tokens = {'additional_special_tokens': ['[EASY]', '[MEDIUM]', '[HARD]']}
    tokenizer.add_special_tokens(special_tokens)
    
    model = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-uncased', 'bert-base-uncased')
    
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.eos_token_id = tokenizer.sep_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    
    model.encoder.resize_token_embeddings(len(tokenizer))
    model.decoder.resize_token_embeddings(len(tokenizer))
    
    return tokenizer, model

def freeze_encoder_layers(model, num_unfrozen_layers=2):
    """
    Freeze encoder layers except for the specified number of layers at the end.
    
    Args:
        model: The encoder-decoder model
        num_unfrozen_layers (int): Number of layers to leave unfrozen
    """
    for param in model.encoder.encoder.layer[:-num_unfrozen_layers]:
        for p in param.parameters():
            p.requires_grad = False
    
    for i in range(1, num_unfrozen_layers + 1):
        for p in model.encoder.encoder.layer[-i].parameters():
            p.requires_grad = True


class QuestionGenerationDataset(Dataset):
    """
    Dataset class for question generation from context with difficulty levels.
    """
    def __init__(self, data, tokenizer, max_length=512):
        """
        Initialize the dataset.
        
        Args:
            data (list): List of dictionaries containing context and QA pairs
            tokenizer: Tokenizer to encode the inputs and targets
            max_length (int): Maximum length for tokenization
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.inputs = []
        self.targets = []
        self.difficulty_labels = []
        
        difficulty_map = {'easy': 0, 'medium': 1, 'hard': 2}
        
        for item in data:
            context = item['context']
            for qa in item['qa_pairs']:
                difficulty = qa['difficulty']
                difficulty_num = difficulty_map[difficulty]
                context_with_difficulty = f"[{difficulty.upper()}] {context}"
                
                self.inputs.append(context_with_difficulty)
                self.targets.append(qa['question'])
                self.difficulty_labels.append(difficulty_num)

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.inputs)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            dict: Dictionary containing encoded inputs and targets
        """
        input_encoding = self.tokenizer(
            self.inputs[idx],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        target_encoding = self.tokenizer(
            self.targets[idx],
            max_length=self.max_length//2,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': input_encoding['input_ids'].squeeze(),
            'attention_mask': input_encoding['attention_mask'].squeeze(),
            'decoder_input_ids': target_encoding['input_ids'].squeeze(),
            'decoder_attention_mask': target_encoding['attention_mask'].squeeze(),
            'labels': target_encoding['input_ids'].squeeze(),
            'difficulty': torch.tensor(self.difficulty_labels[idx])
        }


def load_dataset(file_path, tokenizer, batch_size=8):
    """
    Load dataset from a JSON file and create a DataLoader.
    
    Args:
        file_path (str): Path to the JSON file containing the dataset
        tokenizer: Tokenizer to use for encoding
        batch_size (int): Batch size for the DataLoader
        
    Returns:
        DataLoader: DataLoader for the dataset
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    dataset = QuestionGenerationDataset(data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader


def train_model(model, dataloader, optimizer, device, num_epochs=1):
    """
    Train the model.
    
    Args:
        model: Model to train
        dataloader: DataLoader containing training data
        optimizer: Optimizer to use for training
        device: Device to use for training (CPU or GPU)
        num_epochs (int): Number of epochs to train for
        
    Returns:
        model: Trained model
    """
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": total_loss / (progress_bar.n + 1)})
        
        print(f"Epoch {epoch+1} - Average loss: {total_loss / len(dataloader)}")
    
    return model


def save_model(model, tokenizer, model_dir="question_generator_model"):
    """
    Save the model and tokenizer to disk.
    
    Args:
        model: Model to save
        tokenizer: Tokenizer to save
        model_dir (str): Directory to save the model and tokenizer to
    """
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    torch.save(model.state_dict(), f'{model_dir}.pth')


def generate_questions(context, difficulty='easy', model=None, tokenizer=None, device=None):
    """
    Generate questions based on the context and difficulty.
    
    Args:
        context (str): Context to generate questions from
        difficulty (str): Difficulty level ('easy', 'medium', or 'hard')
        model: Model to use for generation
        tokenizer: Tokenizer to use for encoding and decoding
        device: Device to use for generation
        
    Returns:
        str: Generated question
    """
    context_with_difficulty = f"[{difficulty.upper()}] {context}"
    
    inputs = tokenizer(
        context_with_difficulty,
        return_tensors='pt',
        max_length=512,
        padding='max_length',
        truncation=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    outputs = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=64,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=3,
        decoder_start_token_id=tokenizer.cls_token_id,
        eos_token_id=tokenizer.sep_token_id,
        pad_token_id=tokenizer.pad_token_id
    )
    
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    first_question = decoded.split('?')[0].strip() + '?' if '?' in decoded else decoded
    
    return first_question


def main():
    """
    Main function to run the entire pipeline.
    """
    tokenizer, model = initialize_model_and_tokenizer()
    
    freeze_encoder_layers(model, num_unfrozen_layers=2)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    dataloader = load_dataset('/content/Json_merged_with_difficulty.json', tokenizer)
    
    optimizer = AdamW(model.parameters(), lr=5e-5)
    
    model = train_model(model, dataloader, optimizer, device)
    
    save_model(model, tokenizer)
    
    test_context = "Parkinson's disease is a progressive neurodegenerative disorder affecting primarily the motor system."
    
    easy_question = generate_questions(test_context, difficulty='easy', model=model, tokenizer=tokenizer, device=device)
    medium_question = generate_questions(test_context, difficulty='medium', model=model, tokenizer=tokenizer, device=device)
    hard_question = generate_questions(test_context, difficulty='hard', model=model, tokenizer=tokenizer, device=device)
    
    print(f"Easy: {easy_question}")
    print(f"Medium: {medium_question}")
    print(f"Hard: {hard_question}")
    
    print(tokenizer.convert_tokens_to_ids(['[EASY]', '[MEDIUM]', '[HARD]']))


if __name__ == "__main__":
    main()