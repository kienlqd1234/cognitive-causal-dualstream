#!/usr/bin/python
import os
# Ensure the script can find other modules in the src directory
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from MeaningAwareSelection import MeaningAwareSelection
# from LLMInterface import MockLLM # Comment out or remove MockLLM
from LLMInterface import create_llm, HFSequenceClassifierLLM_legacy, HFNLIRelevanceLLM, GenerativeRelevanceLLM # Import specific class for clarity
from ConfigLoader import config as global_config, logger # Renamed config to global_config

# Attempt to import sentencepiece to check if it's installed - less critical for TinyLlama but good to keep for now
try:
    import sentencepiece
    logger.info("SentencePiece library found.")
except ImportError:
    logger.warning("SentencePiece library not found. May be required for some tokenizers. Please install it: pip install sentencepiece")

def test_meaning_aware_filter_bert_nli():
    """
    Tests the MeaningAwareSelection filter with sample inputs, using a BERT-style NLI model (e.g., bert-base-uncased).
    """
    logger.info("Initializing MeaningAwareSelection for relevance testing with a BERT-style NLI model...")

    # User changed to textattack/bert-base-uncased-mnli, now changing to bert-base-uncased
    nli_model_name = global_config.get('LLM_SETTINGS', {}).get('nli_model_name', "bert-base-uncased")
    logger.info(f"Attempting to load NLI model ({nli_model_name})...")
    try:
        nli_llm_config = {
            "type": "hf_nli_relevance",
            "model_name_or_path": nli_model_name,
            # "device": "cuda" # Optional: force cuda
        }
        llm_interface = create_llm(nli_llm_config)
        # Ensure the loaded LLM is of the expected type
        if not isinstance(llm_interface, HFNLIRelevanceLLM):
            logger.error(f"LLM interface is not of type HFNLIRelevanceLLM, but {type(llm_interface).__name__}. Check config.")
            return
        logger.info(f"Using LLM: {type(llm_interface).__name__} with model {llm_interface.model_name}")

    except Exception as e:
        logger.error(f"Failed to initialize NLI LLM interface ({nli_model_name}): {e}", exc_info=True)
        logger.error("Ensure the model name is correct and transformers library is installed.")
        return

    # Initialize MeaningAwareSelection
    try:
        meaning_selector = MeaningAwareSelection(
            llm_interface=llm_interface,
            config=global_config
        )
        logger.info("MeaningAwareSelection initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize MeaningAwareSelection: {e}", exc_info=True)
        return

    sample_stock_symbol = "AAPL"
    sample_texts = [
        "Apple's new iPhone 15 is expected to boost sales significantly this quarter.", # Likely positive -> Relevant
        "The weather today is sunny with a high of 75 degrees.", # Likely neutral -> Irrelevant
        "AAPL stock price surged by 5% after the product announcement.", # Likely positive -> Relevant
        "Analysts are bullish on Apple (AAPL) following strong earnings reports.", # Likely positive -> Relevant
        "A local bakery won the award for the best croissant in town.", # Likely neutral -> Irrelevant
        "SEC is investigating trading activities related to several tech stocks, not including Apple.", # Potentially neutral/negative context for others, but neutral for AAPL -> Irrelevant for AAPL focus by FinBERT sentiment mapping
        "Rumors suggest Apple might be developing a new VR headset.", # Could be neutral/positive -> Irrelevant/Relevant depending on FinBERT's take
        "The global chip shortage continues to affect production across industries.", # Likely negative for AAPL -> Relevant
        "Apple (AAPL) faces new lawsuits regarding app store policies.", # Likely negative -> Relevant
        "The market remains volatile due to inflation concerns, AAPL unaffected." # Likely neutral for AAPL -> Irrelevant
    ]

    logger.info(f"\n--- Testing filter for stock: {sample_stock_symbol} with {nli_model_name} (as NLI) ---")
    logger.info("Original texts:")
    for i, text in enumerate(sample_texts):
        logger.info(f"{i+1}. {text}")

    try:
        # Note: HFSequenceClassifierLLM_legacy in MeaningAwareSelection will use the prompt_template
        # from MeaningAwareSelection config, which combines text and ticker.
        relevant_texts_list, relevance_mask = meaning_selector.filter_texts(
            texts=sample_texts,
            ticker=sample_stock_symbol
        )
        irrelevant_texts_list = [sample_texts[i] for i, relevant in enumerate(relevance_mask) if not relevant]

    except Exception as e:
        logger.error(f"Error during text filtering: {e}", exc_info=True)
        return

    logger.info("\n--- Filter Results (bert-base-uncased as NLI) ---")
    logger.info(f"Model '{nli_model_name}' used with HFNLIRelevanceLLM. If it's a base model, its NLI head is randomly initialized.")
    logger.info(f"HFNLIRelevanceLLM maps a detected 'entailment' label to 'Relevant'. Check entailment ID log during init.")
    logger.info(f"Found {len(relevant_texts_list)} RELEVANT texts for {sample_stock_symbol}:")
    if relevant_texts_list:
        for i, text in enumerate(relevant_texts_list):
            logger.info(f"  R{i+1}. {text}")
    else:
        logger.info("  None")

    logger.info(f"\nFound {len(irrelevant_texts_list)} IRRELEVANT texts for {sample_stock_symbol}:")
    if irrelevant_texts_list:
        for i, text in enumerate(irrelevant_texts_list):
            logger.info(f"  IR{i+1}. {text}")
    else:
        logger.info("  None")

    logger.info("\n--- {nli_model_name} (as NLI) Test Complete ---")

def test_meaning_aware_filter_fingpt_gpt2_qa():
    """
    Tests the MeaningAwareSelection filter with sample inputs, using OpenFinAL/GPT2_FINGPT_QA.
    """
    model_name = r"D:\FinalYear\KLTN\PEN\PEN-main\PEN-main\LLM_Model\finance-gpt2" # Use local path for finance-gpt2
    logger.info(f"Initializing MeaningAwareSelection for relevance testing with local Hugging Face model: {model_name}...")

    try:
        llm_config = {
            "type": "fingpt_gpt2_qa_generative",
            "model_name_or_path": model_name,
            "max_new_tokens": 30,
            "temperature": 0.1,
            # "device": "cuda" # Optional: force cuda
        }
        llm_interface = create_llm(llm_config)
        if not isinstance(llm_interface, GenerativeRelevanceLLM):
            logger.error(f"LLM interface is not of type GenerativeRelevanceLLM, but {type(llm_interface).__name__}.")
            return
        logger.info(f"Using LLM: {type(llm_interface).__name__} with model {model_name}")

    except ImportError as e:
        logger.error(f"ImportError for {model_name}: {e}. This often means the 'transformers' library is too old or a component like 'AutoModelForCausalLM' is missing. Please update transformers: pip install --upgrade transformers")
        return        
    except Exception as e:
        logger.error(f"Failed to initialize LLM interface ({model_name}): {e}", exc_info=True)
        return

    try:
        meaning_selector = MeaningAwareSelection(llm_interface=llm_interface, config=global_config)
        logger.info("MeaningAwareSelection initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize MeaningAwareSelection: {e}", exc_info=True)
        return

    sample_stock_symbol = "MSFT"
    sample_texts = [
        "Microsoft reported strong earnings this quarter, beating analyst expectations.",
        "The new Xbox series is selling out worldwide.",
        "A new cafe opened downtown, unrelated to any tech company.",
        "MSFT stock is up 3% in pre-market trading."
    ]

    logger.info(f"\n--- Testing filter for stock: {sample_stock_symbol} with {model_name} ---")
    logger.info("Original texts:")
    for i, text in enumerate(sample_texts):
        logger.info(f"{i+1}. {text}")

    try:
        relevant_texts_list, relevance_mask = meaning_selector.filter_texts(
            texts=sample_texts,
            ticker=sample_stock_symbol
        )
        irrelevant_texts_list = [sample_texts[i] for i, relevant in enumerate(relevance_mask) if not relevant]
    except Exception as e:
        logger.error(f"Error during text filtering: {e}", exc_info=True)
        return

    logger.info(f"\n--- Filter Results ({model_name}) ---")
    logger.info(f"Model '{model_name}' used with GenerativeRelevanceLLM. Output depends on prompting.")
    logger.info(f"Found {len(relevant_texts_list)} RELEVANT texts for {sample_stock_symbol}:")
    if relevant_texts_list:
        for i, text in enumerate(relevant_texts_list):
            logger.info(f"  R{i+1}. {text}")
    else:
        logger.info("  None")

    logger.info(f"\nFound {len(irrelevant_texts_list)} IRRELEVANT texts for {sample_stock_symbol}:")
    if irrelevant_texts_list:
        for i, text in enumerate(irrelevant_texts_list):
            logger.info(f"  IR{i+1}. {text}")
    else:
        logger.info("  None")

    logger.info(f"\n--- {model_name} Test Complete ---")

if __name__ == "__main__":
    import logging
    # Set logging level to DEBUG to see detailed logs from LLMInterface
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s') 
    
    # Create a dummy preprocessed path for ConfigLoader if needed
    # This is a simplified version, ConfigLoader might have more complex path needs
    default_data_path = global_config.get('DEFAULT', {}).get('data_path', '.') # Default to current dir if not in config
    dummy_preprocessed_path = os.path.join(default_data_path, 'preprocessed')
    try:
        os.makedirs(dummy_preprocessed_path, exist_ok=True)
        logger.info(f"Ensured dummy preprocessed path exists: {dummy_preprocessed_path}")
    except Exception as e:
        logger.warning(f"Could not create dummy preprocessed path {dummy_preprocessed_path}: {e}")

    # test_meaning_aware_filter_bert_nli() # Keep this if you want to run both
    test_meaning_aware_filter_fingpt_gpt2_qa() # Call the new test function 