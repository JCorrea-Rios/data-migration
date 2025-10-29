#!/usr/bin/env python3
"""
GLiNER-based classifier for company name entity recognition.
"""
import spacy
from gliner_spacy.pipeline import GlinerSpacy
import pandas as pd
import torch
from typing import List, Dict, Any, Tuple
from tqdm import tqdm

def get_optimal_device() -> str:
    """
    Automatically detect and return the best available device for inference.
    Priority: MPS (Apple Silicon) > CUDA (NVIDIA) > CPU
    """
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

# Configuration for GLiNER integration with expanded labels
GLINER_CONFIG = {
    "gliner_model": "urchade/gliner_medium-v2.1",
    "chunk_size": 250,
    "labels": # [
        # # Industry categories
        # "professional_services", "legal", "technology", 
        # "telecommunications", "financial", "healthcare", 
        # "retail", "energy", #"industrial",
        # "logistics",
        
        # # Organization types
        # "corporation", 
        # "government", 
        # "educational",
        # "non_profit"
    # ],
    ["company name", "company identifier", "address", 
    "financial instrument", "nonprofit entity", "government entity", 
    "individual person", "product or service name", 
    "educational and research institutions", "ambiguous"],
    "threshold": 0.5,
    "map_location": get_optimal_device()  # Auto-detect: MPS, CUDA, or CPU
}

class GLiNERClassifier:
    def __init__(self, config=None, context_strategy: str = "neutral"):
        """
        Initialize the GLiNER classifier with the given configuration.
        
        Args:
            config: GLiNER configuration dictionary
            context_strategy: How to add context ("neutral", "explicit", "minimal", "none")
        """
        self.config = config or GLINER_CONFIG
        self.device = self.config.get("map_location", "cpu")
        self.context_strategy = context_strategy
        
        print(f"[GLiNER] Initializing with device: {self.device}")
        print(f"[GLiNER] Context strategy: {context_strategy}")
        if self.device == "mps":
            print(f"[GLiNER] Using Apple Silicon GPU acceleration (Metal Performance Shaders)")
        elif self.device == "cuda":
            print(f"[GLiNER] Using NVIDIA CUDA GPU acceleration")
        else:
            print(f"[GLiNER] Using CPU (consider upgrading PyTorch for GPU support)")
        
        self.nlp = spacy.blank("en")
        self.nlp.add_pipe("gliner_spacy", config=self.config)
        print(f"[GLiNER] Model loaded successfully on {self.device}")
        
    def add_context(self, text: str) -> str:
        """
        Add context to help GLiNER classify diverse entity types.
        
        Args:
            text: The text to classify
            
        Returns:
            Text with appropriate context based on strategy
        """
        if self.context_strategy == "neutral":
            return f"Classify the following: {text}"
        elif self.context_strategy == "explicit":
            return (f"{text} could be a company name, person's name, address, "
                    f"financial instrument, or other type of entity.")
        elif self.context_strategy == "minimal":
            return f"Entity: {text}"
        elif self.context_strategy == "none":
            return text
        else:
            return f"Classify the following: {text}"  # Default to neutral
    
    def classify_single(self, text: str) -> Tuple[str, float]:
        """Classify a single text and return the predicted type and confidence."""
        if not text or not isinstance(text, str):
            return "unknown", 0.0
            
        text_with_context = self.add_context(text)
        doc = self.nlp(text_with_context)
        
        if not doc.ents:
            return "ambiguous", 0.0  # Default fallback when no entities detected
            
        # Find the entity with highest confidence
        best_ent = max(doc.ents, key=lambda e: getattr(e._, 'score', 0))
        label = best_ent.label_.lower()
        score = getattr(best_ent._, 'score', 0.0)
        
        return label, score
        
    def classify_batch(self, texts: List[str], batch_size: int = 100) -> List[Tuple[str, float]]:
        """Classify a batch of texts for efficiency."""
        results = []
        
        # Calculate total batches for progress bar
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        # Process in smaller batches to avoid memory issues
        with tqdm(total=len(texts), desc="GLiNER Classification", unit="records", 
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                batch_with_context = [self.add_context(text) for text in batch]
                
                docs = list(self.nlp.pipe(batch_with_context))
                
                for doc in docs:
                    if not doc.ents:
                        results.append(("ambiguous", 0.0))  # Default fallback when no entities detected
                        continue
                        
                    best_ent = max(doc.ents, key=lambda e: getattr(e._, 'score', 0))
                    label = best_ent.label_.lower()
                    score = getattr(best_ent._, 'score', 0.0)
                    results.append((label, score))
                
                # Update progress bar
                pbar.update(len(batch))
                
        return results
        
    def process_dataframe(self, df: pd.DataFrame, name_column: str = "norm", 
                         batch_size: int = 100) -> pd.DataFrame:
        """Process a dataframe containing texts to classify."""
        print(f"  Processing {len(df):,} records from column '{name_column}'...")
        texts = df[name_column].tolist()
        classifications = self.classify_batch(texts, batch_size)
        
        df["gliner_type"] = [c[0] for c in classifications]
        df["gliner_confidence"] = [c[1] for c in classifications]
        
        return df