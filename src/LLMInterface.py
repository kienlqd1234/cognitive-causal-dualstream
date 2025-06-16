#!/usr/local/bin/python
import os
import openai
import time
import random
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
# It's good practice to have a logger available, assuming ConfigLoader.logger is set up
# If not, a basic logger can be configured here for LLMInterface specific logs.
# from ConfigLoader import logger # If available and configured
import logging
logger = logging.getLogger(__name__) # Use a module-specific logger

class OpenAILLM:
    """Simple interface to OpenAI API"""
    
    def __init__(self, model: str = "gpt-3.5-turbo", max_tokens: int = 150, temperature: float = 0.7, api_key: Optional[str] = None):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Set API key
        if api_key:
            openai.api_key = api_key
        elif "OPENAI_API_KEY" in os.environ:
            openai.api_key = os.environ["OPENAI_API_KEY"]
        else:
            raise ValueError("OpenAI API key not provided and not found in environment variables")
    
    def __call__(self, prompt: str) -> str:
        """Call OpenAI API and return response"""
        return self.generate(prompt)
    
    def generate(self, prompt: str, max_retries: int = 3) -> str:
        """Generate text using OpenAI API with retry logic"""
        for attempt in range(max_retries):
            try:
                # Using ChatCompletion for models like gpt-3.5-turbo
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."}, # Optional system message
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    stop=None
                )
                return response.choices[0].message['content'].strip() # Adjusted for ChatCompletion response structure
                
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                wait_time = (2 ** attempt) * 1.0  # Exponential backoff
                print(f"Error calling OpenAI API: {str(e)}")
                print(f"Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
    
    def batch_generate(self, prompts: List[str]) -> List[str]:
        """Generate text for multiple prompts"""
        # Note: For true batching with ChatCompletion, you might explore asynchronous calls 
        # or structure your requests differently if the API supports batching for this endpoint.
        # This implementation processes them sequentially.
        return [self.generate(prompt) for prompt in prompts]


class MockLLM:
    """Mock LLM for testing without API calls"""
    
    def __init__(self):
        self.responses = {
            "reflection": [
                "This explanation lacks specific details about why the price movement is expected.",
                "The explanation doesn't connect the news to the predicted outcome clearly.",
                "The explanation could be improved by adding more context about the industry trends."
            ],
            "improvement": [
                "Based on the information, the stock price is likely to {direction} because the earnings report exceeded analyst expectations, showing strong revenue growth of 15% year-over-year.",
                "The stock will probably {direction} as the company announced a new partnership with a major industry player, which analysts see as a strategic advantage.",
                "Given the information about recent product launches and positive market reception, the stock is expected to {direction} in the short term."
            ]
        }
    
    def __call__(self, prompt: str) -> str:
        """Return mock response based on prompt content"""
        time.sleep(0.5)  # Simulate API delay
        
        if "reflect" in prompt.lower():
            return random.choice(self.responses["reflection"])
        
        if "improve" in prompt.lower() or "provide an improved explanation" in prompt.lower():
            direction = "increase" if "increase" in prompt.lower() else "decrease"
            response = random.choice(self.responses["improvement"])
            return response.format(direction=direction)
        
        # Default response
        return "The stock price is likely to change based on the provided information."


class TransformerLLM:
    """Interface to local transformer models"""
    
    def __init__(self, model, tokenizer, max_tokens=500, temperature=0.7):
        self.model = model
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.temperature = temperature
    
    def __call__(self, prompt: str) -> str:
        """Generate response using local transformer model"""
        try:
            # Use encode instead of calling tokenizer directly
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            if hasattr(self.model, 'device'):
                inputs = inputs.to(self.model.device)
                
            outputs = self.model.generate(
                input_ids=inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=0.95,
                do_sample=True
            )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part, not the prompt
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
                
            return response
        
        except Exception as e:
            print(f"Error generating response: {e}")
            return "Error: Failed to generate response"


class HFNLIRelevanceLLM:
    """Interface to local Hugging Face NLI models for relevance classification."""
    def __init__(self, 
                 model_name: str = "microsoft/deberta-v3-small-mnli", 
                 hypothesis_template: str = "This text is relevant for predicting {ticker} stock price movement.",
                 device: Optional[str] = None):
        logger.info(f"HFNLIRelevanceLLM.__init__ called with model_name: {model_name}, hypothesis_template: {hypothesis_template}, device: {device}") # DIAGNOSTIC LOG
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name 
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval() 
        self.hypothesis_template = hypothesis_template

        self.entailment_id = -1
        model_name_lower = model_name.lower()

        if "textattack/bert-base-uncased-mnli".lower() in model_name_lower:
            self.entailment_id = 2 
            logger.info(f"Applied specific entailment ID {self.entailment_id} for model {model_name}.")
        elif "albert-base-v2-mnli".lower() in model_name_lower: 
            # Common MNLI scheme: 0: contradiction, 1: neutral, 2: entailment
            self.entailment_id = 2 
            logger.info(f"Applied specific entailment ID {self.entailment_id} for ALBERT model {model_name}.")
        elif "roberta-base-mnli".lower() in model_name_lower: # Added for completeness if user tries RoBERTa
            self.entailment_id = 2
            logger.info(f"Applied specific entailment ID {self.entailment_id} for RoBERTa model {model_name}.")
        
        # General entailment ID detection logic (attempt if not hardcoded)
        if self.entailment_id == -1 and hasattr(self.model.config, 'label2id') and isinstance(self.model.config.label2id, dict):
            for label, id_val in self.model.config.label2id.items():
                if label.lower() == "entailment":
                    self.entailment_id = id_val
                    break
        if self.entailment_id == -1 and hasattr(self.model.config, 'id2label') and isinstance(self.model.config.id2label, dict):
            for id_val, label in self.model.config.id2label.items():
                if label.lower() == "entailment":
                    self.entailment_id = id_val
                    break
        
        if self.entailment_id == -1:
            logger.warning(f"Could not automatically determine 'entailment' label ID for {model_name}. Defaulting to 0. This might be incorrect for many NLI models.")
            self.entailment_id = 0 

        print(f"HFNLIRelevanceLLM initialized with model {self.model_name} on {self.device}")
        print(f"Hypothesis template: {self.hypothesis_template}")
        print(f"Entailment ID (for 'Relevant'): {self.entailment_id}")
        if hasattr(self.model.config, 'id2label') and self.model.config.id2label:
             print(f"Model's original id2label: {self.model.config.id2label}")
        if hasattr(self.model.config, 'label2id') and self.model.config.label2id:
             print(f"Model's original label2id: {self.model.config.label2id}")

    def __call__(self, original_text: str, ticker: str) -> str:
        """
        Classify the original_text as 'Relevant' or 'Irrelevant' to the ticker 
        using NLI (premise = original_text, hypothesis = formatted template).
        """
        hypothesis = self.hypothesis_template.format(ticker=ticker)
        premise = original_text

        try:
            encoded_inputs = self.tokenizer.encode_plus(
                premise, 
                hypothesis, 
                add_special_tokens=True,
                return_attention_mask=True, 
                return_tensors="pt",
                padding='max_length',
                truncation=True,
                max_length=getattr(self.tokenizer, 'model_max_length', 512)
            )
            inputs = {k: v.to(self.device) for k, v in encoded_inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs) # Get the full output tuple
                # In older transformers, logits are often the first element of the output tuple
                logits = outputs[0] 
            
            predicted_class_id = torch.argmax(logits, dim=1).item()
            
            if predicted_class_id == self.entailment_id:
                return "Relevant"
            else:
                return "Irrelevant"

        except Exception as e:
            logger.error(f"Error during HFNLIRelevanceLLM inference: {e}", exc_info=True) # Add exc_info for traceback
            return "Irrelevant" 

    def batch_generate(self, prompts: List[Tuple[str, str]]) -> List[str]:
        """Classify a batch of (original_text, ticker) pairs."""
        return [self.__call__(text, ticker_sym) for text, ticker_sym in prompts]


class GenerativeRelevanceLLM:
    """Interface for using generative LLMs (like TinyLlama) for relevance classification via prompting."""
    def __init__(self, 
                 model_name_or_path: str, 
                 tokenizer_name_or_path: Optional[str] = None,
                 max_new_tokens: int = 10, 
                 device: Optional[str] = None,
                 **kwargs):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing GenerativeRelevanceLLM with model {model_name_or_path} on {self.device}")
        
        # Use specific tokenizer if model is OpenFinAL/GPT2_FINGPT_QA, otherwise use model_name_or_path
        actual_tokenizer_path = tokenizer_name_or_path or model_name_or_path
        if model_name_or_path == "OpenFinAL/GPT2_FINGPT_QA" and not tokenizer_name_or_path:
            actual_tokenizer_path = "gpt2" # Default tokenizer for this specific FinGPT model
            logger.info(f"Using tokenizer '{actual_tokenizer_path}' for model '{model_name_or_path}'")

        logger.info(f"Attempting to load tokenizer: {actual_tokenizer_path}")
        # self.tokenizer = GPT2Tokenizer.from_pretrained(actual_tokenizer_path) # Old
        #Re-enable tokenizer loading block
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(actual_tokenizer_path)
            logger.info(f"Successfully loaded tokenizer using AutoTokenizer for {actual_tokenizer_path}")
        except Exception as e_auto_tok:
            logger.warning(f"AutoTokenizer failed for {actual_tokenizer_path}: {e_auto_tok}. Falling back to GPT2Tokenizer.")
            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained(actual_tokenizer_path)
                logger.info(f"Successfully loaded tokenizer using GPT2Tokenizer for {actual_tokenizer_path} as fallback.")
            except Exception as e_gpt2_tok:
                logger.error(f"GPT2Tokenizer also failed for {actual_tokenizer_path}: {e_gpt2_tok}", exc_info=True)
                raise e_gpt2_tok # Re-raise the gpt2 tokenizer error if fallback also fails

        logger.info(f"Attempting to load config for GPT2LMHeadModel from {model_name_or_path}")
        # Attempt to load model using only GPT2LMHeadModel
        try:
            logger.info(f"Attempting to load config for GPT2LMHeadModel from {model_name_or_path}")
            config = GPT2Config.from_pretrained(model_name_or_path)
            logger.info(f"Successfully loaded GPT2Config from {model_name_or_path}. Architecture: {config.model_type}")
            
            logger.info(f"Attempting to load model with GPT2LMHeadModel from {model_name_or_path} using from_pretrained.")
            self.model = GPT2LMHeadModel.from_pretrained(
                model_name_or_path, # Config will be loaded automatically from this path
                # config=config, # Ensure this is removed
                from_tf=False, 
                force_download=False, 
                local_files_only=True
            ).to(self.device)
            logger.info(f"Successfully loaded model with GPT2LMHeadModel.from_pretrained from {model_name_or_path}.")
        except Exception as e:
            logger.error(f"Failed to load model {model_name_or_path} with GPT2LMHeadModel: {e}", exc_info=True)
            # If there's an error, self.model might not be set. Ensure it's handled before eval.
            raise ValueError(f"Could not load model from {model_name_or_path} using GPT2LMHeadModel.") from e

        self.model.eval()
        
        self.max_new_tokens = max_new_tokens
        self.generation_kwargs = kwargs

        # Set a default pad token if not already set
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token # This should also set pad_token_id
                logger.info(f"Set tokenizer.pad_token to tokenizer.eos_token ('{self.tokenizer.eos_token}') for {model_name_or_path}")
            else:
                logger.warning(f"Tokenizer for {model_name_or_path} has no pad_token and no eos_token. Generation requiring padding may fail.")
        
        # Ensure model config has pad_token_id, falling back to eos_token_id if necessary
        if self.model.config.pad_token_id is None:
            if self.tokenizer.eos_token_id is not None:
                self.model.config.pad_token_id = self.tokenizer.eos_token_id
                logger.info(f"Set model.config.pad_token_id to tokenizer.eos_token_id ({self.tokenizer.eos_token_id}) for {model_name_or_path}")
            # If eos_token_id is also None, model.generate will likely still complain or use a default, but we've done our best here.

        print(f"GenerativeRelevanceLLM initialized with model {model_name_or_path} on {self.device}")
        print(f"Max new tokens for generation: {self.max_new_tokens}")

    def _build_prompt(self, text: str, stock_symbol: str) -> str:
        # Few-shot prompt with examples
        # Ensure the examples are representative and the desired output format is clear.
        # Using triple quotes for the overall f-string to handle inner quotes more easily.
        return f"""Context: News article: "Apple Inc. (AAPL) today announced record iPhone sales for the fourth quarter, significantly exceeding analyst expectations. The company also provided strong guidance for the upcoming holiday season." Stock ticker: AAPL
Question: Is this news article relevant for predicting the stock price movement of AAPL? Answer by responding with only the word 'Relevant' or 'Irrelevant'.
Answer: Relevant

Context: News article: "The city's annual flower show will take place next weekend, featuring roses and tulips from around the world." Stock ticker: MSFT
Question: Is this news article relevant for predicting the stock price movement of MSFT? Answer by responding with only the word 'Relevant' or 'Irrelevant'.
Answer: Irrelevant

Context: News article: "{text}" Stock ticker: {stock_symbol}
Question: Is this news article relevant for predicting the stock price movement of {stock_symbol}? Answer by responding with only the word 'Relevant' or 'Irrelevant'.
Answer:"""
        
    def __call__(self, original_text: str, ticker: str) -> str:
        prompt = self._build_prompt(original_text, ticker)
        
        try:
            inputs = self.tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=getattr(self.tokenizer, 'model_max_length', 2000) - self.max_new_tokens) # Ensure space for generation
            inputs = inputs.to(self.device)
            
            logger.debug(f"GenerativeRelevanceLLM __call__ with generation_kwargs: {self.generation_kwargs}") # Log generation_kwargs
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=self.max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id, 
                    eos_token_id=self.tokenizer.eos_token_id, # Ensure EOS is used
                    **self.generation_kwargs # e.g., temperature, do_sample
                )
            
            # Decode only the generated part
            generated_ids = outputs[0, inputs.shape[-1]:]
            logger.debug(f"GenerativeRelevanceLLM raw generated token IDs for ticker {ticker}: {generated_ids.tolist()}") # Log raw token IDs
            response_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            
            # Log the prompt and decoded output for debugging
            logger.debug(f"GenerativeRelevanceLLM prompt for ticker {ticker}: '{prompt}'")
            logger.debug(f"GenerativeRelevanceLLM decoded output for ticker {ticker}: '{response_text}'")

            # Normalize and parse the response
            response_lower = response_text.lower()
            
            if "not relevant" in response_lower:
                return "Irrelevant"
            elif "relevant" in response_lower: # Check "relevant" after "not relevant" to prioritize "not relevant" if both appear
                return "Relevant"
            else:
                logger.warning(f"GenerativeRelevanceLLM for ticker {ticker} produced ambiguous output: '{response_text}'. Defaulting to Irrelevant.")
                return "Irrelevant" # Default to irrelevant if output is unclear

        except Exception as e:
            logger.error(f"Error during GenerativeRelevanceLLM inference for ticker {ticker}: {e}", exc_info=True)
            return "Irrelevant" # Default to irrelevant on error

    def batch_generate(self, prompts_data: List[Tuple[str, str]]) -> List[str]:
        """
        Classify a batch of (original_text, ticker) pairs.
        Currently processes sequentially. True batch generation for CausalLM with diverse prompts
        requires careful padding and attention mask handling, which can be complex for this specific use case.
        """
        return [self.__call__(text, ticker_sym) for text, ticker_sym in prompts_data]


# Factory function to create appropriate LLM based on configuration
def create_llm(config: Dict[str, Any]) -> Any:
    """Create LLM instance based on configuration"""
    llm_type = config.get("type", "mock").lower()
    logger.info(f"create_llm called with type: {llm_type} and config: {config}") # DIAGNOSTIC LOG
    
    if llm_type == "openai":
        return OpenAILLM(
            model=config.get("model", "gpt-3.5-turbo"),
            max_tokens=config.get("max_tokens", 150),
            temperature=config.get("temperature", 0.7),
            api_key=config.get("api_key")
        )
    
    elif llm_type == "transformer_generator": # Renamed for clarity
        # This assumes model and tokenizer are passed in the config, or model_name to load them
        model_name_or_path = config.get("model_name_or_path")
        if not model_name_or_path:
            raise ValueError("TransformerLLM requires 'model_name_or_path' in config")
        
        # Load model and tokenizer here
        # This part might need adjustment based on how TransformerLLM is intended to be used
        # (e.g., if it's always generative or could be for classification)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        # Determine if it's a causal LM or seq-to-seq for generate, or seq classification
        # For now, assuming it's a generative model if it reaches here and isn't 'hf_nli_relevance' etc.
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path) # Or AutoModelForSeq2SeqLM
        except Exception as e:
            logger.error(f"Failed to load {model_name_or_path} as AutoModelForCausalLM: {e}. Attempting AutoModelForSequenceClassification.")
            try:
                model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
            except Exception as e2:
                raise ValueError(f"Could not load model {model_name_or_path} as either CausalLM or SequenceClassification model: {e2}")

        return TransformerLLM(
            model=model,
            tokenizer=tokenizer,
            max_tokens=config.get("max_tokens", 500), # max_new_tokens for generate
            temperature=config.get("temperature", 0.7)
        )
        
    elif llm_type == "hf_nli_relevance":
        model_to_load = config.get("model_name_or_path", "microsoft/deberta-v3-small-mnli")
        logger.info(f"DEBUG create_llm (hf_nli_relevance): input config['model_name_or_path'] was '{config.get('model_name_or_path')}', resolved to model_to_load: '{model_to_load}'") # DIAGNOSTIC LOG
        return HFNLIRelevanceLLM(
            model_name=model_to_load, 
            hypothesis_template=config.get("hypothesis_template", "This text is relevant for predicting {ticker} stock price movement."),
            device=config.get("device")
        )
    
    elif llm_type == "tinyllama_relevance": # New type for TinyLlama
        return GenerativeRelevanceLLM(
            model_name_or_path=config.get("model_name_or_path", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
            max_new_tokens=config.get("max_new_tokens", 10),
            device=config.get("device"),
            temperature=config.get("temperature", 0.1), 
            do_sample=config.get("do_sample", False)  # Changed default to False
        )

    elif llm_type == "fingpt_gpt2_qa_generative": # New type for OpenFinAL/GPT2_FINGPT_QA
        model_path = config.get("model_name_or_path")
        if not model_path:
            raise ValueError("'model_name_or_path' must be provided for 'fingpt_gpt2_qa_generative' or similar LLM types.")
        
        # tokenizer_name_or_path can be None, GenerativeRelevanceLLM will handle defaulting logic
        tokenizer_path_from_config = config.get("tokenizer_name_or_path")

        return GenerativeRelevanceLLM(
            model_name_or_path=model_path,
            tokenizer_name_or_path=tokenizer_path_from_config, 
            max_new_tokens=config.get("max_new_tokens", 10), 
            device=config.get("device"),
            temperature=config.get("temperature", 0.2),
            do_sample=config.get("do_sample", False)    # Changed default to False
        )

    elif llm_type == "hf_sequence_classifier_legacy": # Ensure this matches what's used if HFSequenceClassifierLLM is separate
        # This class HFSequenceClassifierLLM_legacy was defined at the end of the file.
        # Make sure it's correctly handled or integrated if it's the one intended for "relevance_classifier" type.
        return HFSequenceClassifierLLM_legacy( # Assuming this is the intended class
            model_name=config.get("model_name_or_path", "ProsusAI/finbert"), # Corrected to model_name_or_path
            device=config.get("device")
        )
        
    elif llm_type == "mock":
        return MockLLM()
    
    else:  # Default to mock for testing
        raise ValueError(f"Unknown LLM type: {llm_type}")

# Placeholder for the old HFSequenceClassifierLLM if needed, or could be removed if not used elsewhere
class HFSequenceClassifierLLM_legacy:
    def __init__(self, model_name: str = "ProsusAI/finbert", device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.label_map = {}
        if hasattr(self.model.config, 'id2label'):
            id2label = self.model.config.id2label
            label_to_id = {v.lower(): k for k, v in id2label.items()}
            if "positive" in label_to_id: self.label_map[label_to_id["positive"]] = "Relevant"
            if "negative" in label_to_id: self.label_map[label_to_id["negative"]] = "Relevant"
            if "neutral" in label_to_id: self.label_map[label_to_id["neutral"]] = "Irrelevant"
        else:
            self.label_map = {0: "Irrelevant", 1: "Relevant"}
        print(f"HFSequenceClassifierLLM_legacy initialized with {model_name} on {self.device}. Map: {self.label_map}")

    def __call__(self, prompt: str) -> str:
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(self.device)
            with torch.no_grad(): logits = self.model(**inputs).logits
            predicted_class_id = torch.argmax(logits, dim=1).item()
            return self.label_map.get(predicted_class_id, "Irrelevant")
        except Exception as e:
            print(f"Error in HFSequenceClassifierLLM_legacy: {e}")
            return "Irrelevant"
    def batch_generate(self, prompts: List[str]) -> List[str]:
        return [self.__call__(p) for p in prompts] 