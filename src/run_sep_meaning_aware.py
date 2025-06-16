#!/usr/local/bin/python
import os
import argparse
import json
from SEP_Integration import SEP_PEN_Integration

def main():
    parser = argparse.ArgumentParser(description='Run SEP-PEN with Meaning-Aware Selection')
    parser.add_argument('--mode', type=str, default='evaluate',
                      choices=['train', 'evaluate', 'optimize'],
                      help='Mode to run in')
    parser.add_argument('--llm_type', type=str, default='mock',
                      choices=['mock', 'openai', 'transformer'],
                      help='Type of LLM to use')
    parser.add_argument('--llm_model', type=str, default='gpt-3.5-turbo-instruct',
                      help='Model name for the LLM')
    parser.add_argument('--epochs', type=int, default=5,
                      help='Number of epochs to train for')
    parser.add_argument('--train_classifier', action='store_true',
                      help='Train a relevance classifier')
    parser.add_argument('--output_dir', type=str, default='results/meaning_aware_selection',
                      help='Directory to save results')
    parser.add_argument('--eval_phase', type=str, default='test',
                      choices=['dev', 'test'],
                      help='Phase to evaluate on')
    parser.add_argument('--openai_api_key', type=str,
                      help='OpenAI API key (if not set in environment)')
    
    args = parser.parse_args()
    
    # Set OpenAI API key if provided
    if args.openai_api_key:
        os.environ['OPENAI_API_KEY'] = args.openai_api_key
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize SEP-PEN integration with OpenAI configuration
    llm_config = {
        'type': args.llm_type,
        'model': args.llm_model,
        'max_tokens': 500,
        'temperature': 0.7,
        'api_key': os.environ.get('OPENAI_API_KEY')
    }
    
    sep_pen = SEP_PEN_Integration(
        config_path="src/config.yml",
        llm_config=llm_config
    )
    
    if args.mode == 'train':
        print(f"Training for {args.epochs} epochs...")
        sep_pen.train(n_epochs=args.epochs, train_relevance_classifier=args.train_classifier)
        
    elif args.mode == 'evaluate':
        print(f"Evaluating on {args.eval_phase} set...")
        results = sep_pen.evaluate(phase=args.eval_phase)
        
        # Save results
        results_file = os.path.join(args.output_dir, f'{args.eval_phase}_evaluation.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {results_file}")
        
    elif args.mode == 'optimize':
        print("Optimizing LLM with PPO...")
        sep_pen.optimize_llm_with_ppo(n_iterations=5)

if __name__ == "__main__":
    main() 