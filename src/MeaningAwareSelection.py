#!/usr/local/bin/python
import os
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
# Assuming LLMInterface and its classes (like HFSequenceClassifierLLM) are in the same directory or accessible
from LLMInterface import create_llm # Changed from relative to direct import
from LLMInterface import HFNLIRelevanceLLM, GenerativeRelevanceLLM # Import specific types for isinstance checks

class MeaningAwareSelection:
    """
    Meaning-Aware Selection module that filters out irrelevant/ambiguous text
    before it reaches the TSU component of PEN using LLM capabilities.
    """
    
    def __init__(self, llm_interface: Optional[Any] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Meaning-Aware Selection module.
        
        Args:
            llm_interface: LLM interface for relevance filtering. Can be an instance or None.
            config: Configuration dictionary. If llm_interface is None, config must contain llm_config.
        """
        self.config = config or {}
        self.cache = {}  # Cache for storing relevance judgments
        
        # The prompt_template is no longer directly used by HFNLIRelevanceLLM as it has its own hypothesis_template.
        # However, keeping it here for potential use with other LLM types or if a user wants to override.
        self.prompt_template = self.config.get('prompt_template', 
            "Placeholder: Given the sentence: \"{text}\", is it relevant to predicting {ticker} stock price movement? "
        )
        
        self.confidence_threshold = self.config.get('confidence_threshold', 0.5) 

        if llm_interface:
            self.llm = llm_interface
        elif "llm_config" in self.config:
            self.llm = create_llm(self.config["llm_config"]) 
        else:
            print("Warning: MeaningAwareSelection initialized without an LLM interface or llm_config. It will treat all texts as relevant.")
            self.llm = None
        
    def filter_texts(self, texts: List[str], ticker: str) -> Tuple[List[str], List[bool]]:
        """
        Filter out irrelevant texts using the LLM.
        
        Args:
            texts: List of text strings to filter
            ticker: Stock ticker symbol
            
        Returns:
            Tuple of (filtered_texts, relevance_mask)
        """
        if not texts:
            return [], []
            
        if not self.llm:
            return texts, [True] * len(texts)  # If no LLM, return all texts as relevant
            
        relevance_judgments = []
        cache_hits = 0
        
        # Create cache key based on ticker
        cache_key = f"ticker_{ticker}"
        if cache_key not in self.cache:
            self.cache[cache_key] = {}
            
        # Check which texts are in cache
        texts_to_evaluate = []
        text_indices = []
        
        for i, text in enumerate(texts):
            if text in self.cache[cache_key]:
                relevance_judgments.append(self.cache[cache_key][text])
                cache_hits += 1
            else:
                texts_to_evaluate.append(text)
                text_indices.append(i)
                
        # Only query LLM for texts not in cache
        if texts_to_evaluate:
            new_judgments = self._batch_evaluate_relevance(texts_to_evaluate, ticker)
            
            # Update cache with new judgments
            for text, judgment in zip(texts_to_evaluate, new_judgments):
                self.cache[cache_key][text] = judgment
                
            # Insert new judgments back into the original order
            for idx, judgment in zip(text_indices, new_judgments):
                relevance_judgments.insert(idx, judgment)
                
        # Create filtered list and relevance mask
        filtered_texts = [text for text, is_relevant in zip(texts, relevance_judgments) if is_relevant]
        relevance_mask = relevance_judgments
        
        return filtered_texts, relevance_mask
        
    def _batch_evaluate_relevance(self, texts: List[str], ticker: str) -> List[bool]:
        """
        Evaluate the relevance of multiple texts in batch if possible.
        
        Args:
            texts: List of text strings to evaluate
            ticker: Stock ticker symbol
            
        Returns:
            List of boolean relevance judgments
        """
        judgments = []
        
        # Process each text individually
        for text_content in texts: # Renamed text to text_content to avoid conflict with method argument
            if isinstance(self.llm, (HFNLIRelevanceLLM, GenerativeRelevanceLLM)):
                # These LLMs expect original_text and ticker, and handle their specific prompting/logic internally.
                response = self.llm(original_text=text_content, ticker=ticker)
            # Example for a future dedicated sequence classifier that might also fit this pattern:
            # elif isinstance(self.llm, HFSequenceClassifierLLM_legacy): 
            #     response = self.llm(original_text=text_content, ticker=ticker) # If it supports this call signature
            else: # Fallback for other LLM types that might still expect a single pre-formatted prompt
                # This path might be for simpler LLMs or if the prompt_template is central to their use.
                formatted_prompt = self.prompt_template.format(text=text_content, ticker=ticker)
                response = self.llm(formatted_prompt)
            
            is_relevant = response.strip().lower() == "relevant"
            judgments.append(is_relevant)
            
        return judgments
        
    def get_embedding_mask(self, embeddings: np.ndarray, relevance_mask: List[bool]) -> np.ndarray:
        """
        Apply the relevance mask to the embeddings.
        
        Args:
            embeddings: Original embeddings of shape [batch_size, n_texts, embedding_dim]
            relevance_mask: Boolean mask indicating relevance
            
        Returns:
            Filtered embeddings with zeros for irrelevant texts
        """
        # Convert relevance mask to numpy array
        mask = np.array(relevance_mask, dtype=np.float32)
        
        # Reshape mask for broadcasting
        mask_reshaped = mask.reshape(-1, 1)
        
        # Apply mask - multiply by 0 for irrelevant texts
        masked_embeddings = embeddings * mask_reshaped
        
        return masked_embeddings 