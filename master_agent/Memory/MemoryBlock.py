import os
import torch
import openai
import numpy as np
from typing import List
from dotenv import load_dotenv
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load OpenAI API Key
load_dotenv() 
openai.api_key = os.getenv('API_KEY')

########################################################################################################################
################################################# MemoryBlock Class ########################################################
########################################################################################################################
class MemoryBlock:
    """Handles a single MemoryBlock of sentences (summarization & classification)."""

    def __init__(self, sentences: List[str], categories: List[str], use_openai=True, model_type="t5", open_ai_model="gpt-4o", logger=None):
        """
        - sentences: List of sentences in the MemoryBlock
        - categories: List of category labels
        - model_type: 'bart' or 't5' for Hugging Face models
        - use_openai: Whether to use OpenAI API instead of local models
        """
        ######################## Configurations
        self.use_openai = use_openai
        self.device = 0 if torch.cuda.is_available() else -1  # Use GPU if available
        self.logger = logger
        self.logger_name = "MEMORY"
        ######################## Models Init
        # If not using OpenAI, load local Hugging Face models
        self.openai_model = open_ai_model
        if not use_openai:
            if model_type == "t5":
                self.title_generator = pipeline("text2text-generation", model="t5-large", device=self.device)
            else:# model_type == "bart":
                self.title_generator = pipeline("summarization", model="facebook/bart-large-cnn", device=self.device)

            self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=self.device)

        # Load sentence transformer model for embeddings
        #self.embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Faster & optimized for similarity

        ######################## Placeholder for results
        joint_sentences = " ".join(sentences)
        self.summary = self.__summarize(joint_sentences)
        self.category, self.confidence = self.__classify(joint_sentences, categories)
        self.embedding = self.__compute_summary_embedding(self.summary)
        ######################################################################################

    def measure_similarity(self, new_text: str) -> float:
        """Checks if a new sentence is related to the memory block using cosine similarity."""
        new_embedding = self.embedding_model.encode([new_text])[0]  # Encode new sentence
        similarity = cosine_similarity([self.embedding], [new_embedding])[0][0]  # Compute similarity
        return similarity
    
    def update_summary(self, new_text: str):
        """If the new text is related, update the memory block summary and re-compute embedding."""
        # Combine existing summary with the new text
        combined_text = f"{self.summary} {new_text}"

        # Generate a new summary
        self.summary = self.__summarize(combined_text)

        # Update memory block embedding
        self.embedding = self.__compute_summary_embedding(self.summary)
            
    def __openai_request(self, prompt: str, max_tokens=50) -> str:
        """Helper function to send API requests to OpenAI."""
        client = openai.OpenAI(api_key=openai.api_key)  # Create OpenAI client

        response = client.chat.completions.create(
            model=self.openai_model,
            messages=[{"role": "system", "content": "You are a helpful AI assistant."},
                      {"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()

    def __compute_summary_embedding(self, summary: str):
        """Computes and stores only the summary's embedding (memory efficient)."""
        # Split the summary into sentences
        sentences = summary.split(".")
        # Encode each sentence
        embeddings = []
        for sentence in sentences:
            embeddings.append(self.embedding_model.encode(sentence))
        # Average the sentence embeddings
        mean_embedding = np.mean(embeddings, axis=0)
        return mean_embedding
    
    def __summarize(self, joint_sentences, max_length=100, min_length=10):
        """Generates a summary using OpenAI API or local model."""
        summary = "Summary unavailable"
        try:
            if self.use_openai:
                prompt = f"Summarize the following text in one concise sentence:\n\n{joint_sentences}"
                summary = self.__openai_request(prompt, max_tokens=50)
            else:
                if "t5" in str(self.title_generator.model.config._name_or_path):
                    input_text = f"summarize: {joint_sentences}"
                    summary = self.title_generator(input_text, max_length=max_length, min_length=min_length, do_sample=False)
                    summary = summary[0]['generated_text']
                else:
                    summary = self.title_generator(joint_sentences, max_length=max_length, min_length=min_length, do_sample=False)
                    summary = summary[0]['summary_text']
        except Exception as e:
            summary = "Summary unavailable"
            if self.logger is not None:
                message = f"MemBlock: Error in summarization: {e}"
                self.logger.info(f"[{self.logger_name}] {message}")
        return summary

    def __classify(self, joint_sentences, categories):
        """Classifies the Memory Block into a category using OpenAI API or local model."""
        category = "Unknown"
        confidence = 0.0
        try:
            if self.use_openai:
                prompt = f"""Categorize the following text into one of these categories: {", ".join(categories)}.
                Only return the best category name, nothing else.
                Text: {joint_sentences}"""
                
                category = self.__openai_request(prompt, max_tokens=10)
                confidence = 0.95  # Assume high confidence for OpenAI
            else:
                result = self.classifier(joint_sentences, candidate_labels=categories)
                category = result["labels"][0]  # Best category
                confidence = result["scores"][0]  # Confidence score
        except Exception as e:
            category = "Unknown"
            confidence = 0.0
            if self.logger is not None:
                message = f"MemBlock: Error in classification: {e}"
                self.logger.info(f"[{self.logger_name}] {message}")

        return category, confidence

    def __repr__(self):
        return f"Memory Block(Category={self.category}, Confidence={self.confidence:.2f}, Summary='{self.summary}')"