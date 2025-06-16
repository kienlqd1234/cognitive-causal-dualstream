# Meaning-Aware Selection for PEN

This module integrates SEP (Socratic Explanation Pipeline) as a preprocessing step for PEN (Predictive Explainable Network) to filter out noisy/ambiguous/irrelevant text before it reaches the TSU (Time Series Understanding) component of PEN.

## Overview

Meaning-Aware Selection uses the language understanding capabilities of LLMs in SEP to determine the relevance of input texts for stock price prediction. By filtering out irrelevant texts early in the pipeline, the TSU component of PEN can focus on processing only the most relevant information, leading to improved prediction accuracy and explanation quality.

## Key Components

1. **MeaningAwareSelection** - The core module that interfaces with LLM to filter texts based on relevance
2. **RelevanceClassifier** - A lightweight classifier that can be trained on LLM judgments for faster inference
3. **SEP_PEN_Integration** - The integration class that combines PEN's VoS explainability with SEP's reflection approach and the new Meaning-Aware Selection preprocessing

## How It Works

1. **Prompt-based Relevance Filtering**
   
   The system uses SEP or an LLM like GPT/Vicuna to determine relevance:

   ```
   Given the sentence: "I love Apple's new iPhone wallpaper", is it relevant to predicting AAPL stock price?
   Answer: Irrelevant
   ```

2. **Input Data Filtering for PEN**
   
   Only embeddings of sentences labeled as "Relevant" are used:

   etfiltered = ei | LLM(ci) = Relevant

   This ensures the TSU of PEN only processes important text, reducing noise and improving accuracy.

3. **Relevance Classifier Training**
   
   The system collects LLM judgments to train a lightweight classifier for faster inference during runtime.

## Usage

To run PEN with Meaning-Aware Selection:

```bash
python src/run_sep_meaning_aware.py --mode train --llm_type openai --llm_model gpt-3.5-turbo --epochs 5 --train_classifier
```

Options:
- `--mode`: Choose from 'train', 'evaluate', or 'optimize'
- `--llm_type`: Type of LLM to use ('mock', 'openai', 'transformer')
- `--llm_model`: Model name for the LLM
- `--epochs`: Number of epochs to train for
- `--train_classifier`: Flag to train a relevance classifier
- `--output_dir`: Directory to save results
- `--eval_phase`: Phase to evaluate on ('dev' or 'test')

## Benefits

1. **Improved Prediction Accuracy**: By filtering out noise and irrelevant information, the model can focus on truly relevant signals.
2. **Better Explanations**: Explanations are based on more relevant and focused information.
3. **Efficiency**: The TSU component processes less data, potentially leading to faster inference times.
4. **Interpretability**: The filtering process provides an additional layer of transparency, showing which inputs were deemed relevant or irrelevant.

## Implementation Details

### Relevance Filtering Approaches

1. **Direct LLM Judgments**: Uses LLM like GPT to directly judge the relevance of each text.
2. **Distilled Classifier**: A lightweight model trained on LLM judgments for faster inference.

### Integration with PEN

The Meaning-Aware Selection acts as a preprocessing step:

1. Raw text inputs → MeaningAwareSelection → Filtered texts
2. Filtered texts → PEN's TSU component → Prediction + Explanation

The module can be easily enabled or disabled for comparative experiments.

## Requirements

See the updated `requirements.txt` file for all dependencies. Key requirements include:
- TensorFlow
- PyTorch
- Transformers
- scikit-learn (for the relevance classifier)
- OpenAI API (optional, for using GPT models) 