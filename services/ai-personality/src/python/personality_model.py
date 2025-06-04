"""
AI Personality Model for EthernalEcho
This module handles personality modeling and response generation
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from sentence_transformers import SentenceTransformer
import faiss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PersonalityModel:
    """Main personality modeling class"""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        """Initialize personality model
        
        Args:
            model_name: Base model to use for fine-tuning
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base_model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Memory bank components
        self.sentence_encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.memory_index = None
        self.memories = []
        
        logger.info(f"Initialized PersonalityModel on {self.device}")
    
    def extract_personality_traits(self, texts: List[str]) -> Dict[str, float]:
        """Extract Big Five personality traits from text samples
        
        Args:
            texts: List of text samples from user
            
        Returns:
            Dictionary of personality trait scores
        """
        traits = {
            "openness": 0.0,
            "conscientiousness": 0.0,
            "extraversion": 0.0,
            "agreeableness": 0.0,
            "neuroticism": 0.0
        }
        
        # Personality indicators (simplified)
        indicators = {
            "openness": ["creative", "curious", "imaginative", "artistic", "adventure"],
            "conscientiousness": ["organized", "careful", "disciplined", "efficient", "responsible"],
            "extraversion": ["social", "outgoing", "energetic", "talkative", "friendly"],
            "agreeableness": ["helpful", "trusting", "kind", "cooperative", "sympathetic"],
            "neuroticism": ["anxious", "worried", "nervous", "emotional", "stressed"]
        }
        
        for text in texts:
            text_lower = text.lower()
            for trait, keywords in indicators.items():
                score = sum(1 for keyword in keywords if keyword in text_lower)
                traits[trait] += score / len(keywords)
        
        # Normalize scores
        num_texts = len(texts) if texts else 1
        for trait in traits:
            traits[trait] = min(1.0, traits[trait] / num_texts)
        
        return traits
    
    def extract_communication_patterns(self, conversations: List[Dict]) -> Dict:
        """Extract communication patterns from conversations
        
        Args:
            conversations: List of conversation dictionaries
            
        Returns:
            Dictionary of communication patterns
        """
        patterns = {
            "avg_response_length": 0,
            "vocabulary_richness": 0,
            "formality_level": 0,
            "response_time_tendency": "moderate",
            "emotion_expression_level": 0.5,
            "question_frequency": 0
        }
        
        if not conversations:
            return patterns
        
        all_words = []
        total_responses = 0
        total_words = 0
        question_count = 0
        
        for conv in conversations:
            for message in conv.get("messages", []):
                if message.get("sender") == "user":
                    text = message.get("text", "")
                    words = text.split()
                    all_words.extend(words)
                    total_words += len(words)
                    total_responses += 1
                    
                    if "?" in text:
                        question_count += 1
        
        if total_responses > 0:
            patterns["avg_response_length"] = total_words / total_responses
            patterns["vocabulary_richness"] = len(set(all_words)) / len(all_words) if all_words else 0
            patterns["question_frequency"] = question_count / total_responses
        
        return patterns
    
    def build_memory_index(self, memories: List[Dict]) -> None:
        """Build FAISS index for memory retrieval
        
        Args:
            memories: List of memory dictionaries with 'text' field
        """
        if not memories:
            return
        
        self.memories = memories
        texts = [m.get("text", "") for m in memories]
        
        # Encode all memories
        embeddings = self.sentence_encoder.encode(texts)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.memory_index = faiss.IndexFlatL2(dimension)
        self.memory_index.add(embeddings.astype('float32'))
        
        logger.info(f"Built memory index with {len(memories)} memories")
    
    def retrieve_memories(self, query: str, k: int = 5) -> List[Dict]:
        """Retrieve relevant memories for a query
        
        Args:
            query: Query text
            k: Number of memories to retrieve
            
        Returns:
            List of relevant memories
        """
        if not self.memory_index or not self.memories:
            return []
        
        # Encode query
        query_embedding = self.sentence_encoder.encode([query])
        
        # Search in index
        distances, indices = self.memory_index.search(
            query_embedding.astype('float32'), 
            min(k, len(self.memories))
        )
        
        # Return relevant memories
        return [self.memories[idx] for idx in indices[0]]
    
    def fine_tune_model(
        self,
        user_id: str,
        training_data: List[Dict],
        traits: Dict[str, float],
        output_dir: str
    ) -> str:
        """Fine-tune the model on user's data
        
        Args:
            user_id: User ID
            training_data: List of training examples
            traits: Personality traits
            output_dir: Directory to save model
            
        Returns:
            Path to saved model
        """
        # Prepare dataset
        train_texts = []
        for example in training_data:
            if "input" in example and "response" in example:
                # Format as conversation
                text = f"User: {example['input']}\nAssistant: {example['response']}"
                train_texts.append(text)
        
        if not train_texts:
            raise ValueError("No valid training examples found")
        
        # Tokenize
        encodings = self.tokenizer(
            train_texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Create dataset class
        class PersonalityDataset(torch.utils.data.Dataset):
            def __init__(self, encodings):
                self.encodings = encodings
            
            def __len__(self):
                return len(self.encodings.input_ids)
            
            def __getitem__(self, idx):
                return {
                    key: val[idx] for key, val in self.encodings.items()
                }
        
        dataset = PersonalityDataset(encodings)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            save_strategy="epoch",
            evaluation_strategy="no",
            load_best_model_at_end=False,
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.base_model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=self.tokenizer,
        )
        
        # Train
        trainer.train()
        
        # Save model
        model_path = f"{output_dir}/model_{user_id}"
        trainer.save_model(model_path)
        
        # Save personality profile
        profile_path = f"{output_dir}/profile_{user_id}.json"
        with open(profile_path, 'w') as f:
            json.dump({
                "traits": traits,
                "model_path": model_path
            }, f)
        
        return model_path
    
    def generate_response(
        self,
        user_input: str,
        conversation_history: List[Dict],
        personality_traits: Dict[str, float],
        max_length: int = 150
    ) -> str:
        """Generate a personality-consistent response
        
        Args:
            user_input: User's message
            conversation_history: Previous messages
            personality_traits: User's personality traits
            max_length: Maximum response length
            
        Returns:
            Generated response
        """
        # Retrieve relevant memories
        memories = self.retrieve_memories(user_input, k=3)
        
        # Build context
        context = "Personality context:\n"
        
        # Add personality traits
        for trait, value in personality_traits.items():
            if value > 0.7:
                context += f"- High {trait}\n"
            elif value < 0.3:
                context += f"- Low {trait}\n"
        
        # Add recent conversation
        context += "\nConversation:\n"
        for msg in conversation_history[-3:]:  # Last 3 messages
            role = "User" if msg.get("sender") == "human" else "Assistant"
            context += f"{role}: {msg.get('text', '')}\n"
        
        # Add current input
        context += f"User: {user_input}\nAssistant:"
        
        # Generate response
        inputs = self.tokenizer.encode(context, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.base_model.generate(
                inputs,
                max_length=inputs.shape[1] + max_length,
                temperature=0.7 + (personality_traits.get("openness", 0.5) * 0.3),
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the assistant's response
        if "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()
        
        return response


def train_personality_model(user_id: str, training_data: Dict) -> Dict:
    """Main training function called from Node.js
    
    Args:
        user_id: User ID
        training_data: Dictionary containing texts, conversations, etc.
        
    Returns:
        Dictionary with model path and metrics
    """
    try:
        model = PersonalityModel()
        
        # Extract personality traits
        texts = training_data.get("texts", [])
        traits = model.extract_personality_traits(texts)
        
        # Extract communication patterns
        conversations = training_data.get("conversations", [])
        patterns = model.extract_communication_patterns(conversations)
        
        # Build memory index
        memories = training_data.get("memories", [])
        model.build_memory_index(memories)
        
        # Prepare training examples
        training_examples = []
        for conv in conversations:
            messages = conv.get("messages", [])
            for i in range(len(messages) - 1):
                if messages[i].get("sender") == "human" and messages[i+1].get("sender") == "assistant":
                    training_examples.append({
                        "input": messages[i].get("text", ""),
                        "response": messages[i+1].get("text", "")
                    })
        
        # Fine-tune model
        output_dir = f"/models/personality/{user_id}"
        os.makedirs(output_dir, exist_ok=True)
        
        model_path = model.fine_tune_model(
            user_id,
            training_examples,
            traits,
            output_dir
        )
        
        return {
            "success": True,
            "model_path": model_path,
            "traits": traits,
            "patterns": patterns,
            "memories_indexed": len(memories)
        }
        
    except Exception as e:
        logger.error(f"Error training personality model: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def generate_response(user_id: str, message: str, context: Dict) -> Dict:
    """Generate response for user message
    
    Args:
        user_id: User ID
        message: User's message
        context: Conversation context
        
    Returns:
        Dictionary with generated response
    """
    try:
        # Load user's model
        model_path = f"/models/personality/{user_id}/model_{user_id}"
        profile_path = f"/models/personality/{user_id}/profile_{user_id}.json"
        
        # Load profile
        with open(profile_path, 'r') as f:
            profile = json.load(f)
        
        # Initialize model
        model = PersonalityModel()
        model.base_model = AutoModelForCausalLM.from_pretrained(model_path)
        
        # Load memories if available
        memories = context.get("memories", [])
        if memories:
            model.build_memory_index(memories)
        
        # Generate response
        response = model.generate_response(
            message,
            context.get("conversation_history", []),
            profile["traits"]
        )
        
        return {
            "success": True,
            "response": response,
            "traits_used": profile["traits"]
        }
        
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return {
            "success": False,
            "error": str(e)
        }


if __name__ == "__main__":
    # Test the model
    test_model = PersonalityModel()
    test_traits = test_model.extract_personality_traits([
        "I love exploring new places and trying new things!",
        "I'm very organized and always plan ahead.",
        "I enjoy spending time with friends and family."
    ])
    print("Test traits:", test_traits)