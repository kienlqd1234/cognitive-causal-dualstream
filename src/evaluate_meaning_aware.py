#!/usr/local/bin/python
import os
import sys
import argparse
import tensorflow as tf
import json
from typing import List, Dict, Any, Tuple
import numpy as np
import metrics as pen_metrics

# Add the src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from MeaningAwareSelection import MeaningAwareSelection
from RelevanceClassifier import RelevanceClassifier
from LLMInterface import MockLLM, LLMInterface
from Model import Model
from DataPipe import DataPipe
from ConfigLoader import config, config_model, path_parser

class MeaningAwareEvaluator:
    """
    Evaluator class for testing Meaning-Aware Selection with PEN model.
    Evaluates the effectiveness of text filtering on model performance.
    """
    
    def __init__(self, device: str = None):
        """
        Initialize the evaluator.
        
        Args:
            device: Device to run evaluation on
        """
        # Set up TensorFlow device
        if device == 'cuda' and tf.test.is_built_with_cuda():
            self.device = '/gpu:0'
        else:
            self.device = '/cpu:0'
            
        # Initialize PEN model
        self.pen_model = Model()
        self.pen_model.assemble_graph()
        self.saver = tf.train.Saver()
        
        # Initialize data pipeline
        self.data_pipe = DataPipe()
        
    def setup_llm(self) -> None:
        """
        Set up the mock language model for Meaning-Aware Selection.
        """
        try:
            # Initialize mock LLM interface
            self.llm = MockLLM()
            self.meaning_aware = MeaningAwareSelection(llm_interface=self.llm, config=config)
            print("Using MockLLM for evaluation")
            
        except Exception as e:
            print(f"Error setting up language model: {str(e)}")
            raise
        
    def setup_real_llm(self, llm_api_key: str, llm_model_name: str) -> None:
        """
        Set up a real language model for Meaning-Aware Selection.
        Args:
            llm_api_key: API key for the LLM service.
            llm_model_name: Name of the LLM model to use.
        """
        try:
            # Placeholder for initializing a real LLM interface
            # You'll need to replace this with actual LLM client initialization
            # Example: self.llm = LLMInterface(api_key=llm_api_key, model_name=llm_model_name)
            self.llm = LLMInterface() # Replace with actual LLMInterface initialization
            self.meaning_aware = MeaningAwareSelection(llm_interface=self.llm, config=config)
            print(f"Using real LLM ({llm_model_name}) for evaluation")

        except Exception as e:
            print(f"Error setting up real language model: {str(e)}")
            raise
        
    def setup_classifier(self, classifier_type: str, classifier_path: str = None) -> None:
        """
        Set up the relevance classifier if available.
        
        Args:
            classifier_type: Type of classifier to use
            classifier_path: Path to classifier model
        """
        if classifier_path and os.path.exists(classifier_path):
            self.classifier = RelevanceClassifier(
                model_type=classifier_type,
                model_path=classifier_path,
                device=self.device
            )
        else:
            self.classifier = None
            
    def load_pen_model(self, checkpoint_path: str) -> None:
        """
        Load a pre-trained PEN model from a checkpoint.
        Args:
            checkpoint_path: Path to the model checkpoint.
        """
        try:
            with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
                sess.run(tf.global_variables_initializer())
                # Ensure the path is just the directory for get_checkpoint_state
                checkpoint_dir = os.path.dirname(checkpoint_path)
                ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    self.saver.restore(sess, ckpt.model_checkpoint_path)
                    print(f"PEN model restored from {ckpt.model_checkpoint_path}")
                else:
                    raise FileNotFoundError(f"Checkpoint not found at {checkpoint_dir}")
        except Exception as e:
            print(f"Error loading PEN model: {str(e)}")
            raise
            
    def evaluate_batch(self, batch_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a single batch of data.
        
        Args:
            batch_data: Dictionary containing batch data
            
        Returns:
            Dictionary with evaluation metrics for the batch
        """
        # Extract data from batch
        word_batch = batch_data['word_batch']
        stock_batch = batch_data['stock_batch']
        y_batch = batch_data['y_batch']
        T_batch = batch_data['T_batch']
        n_words_batch = batch_data['n_words_batch']
        n_msgs_batch = batch_data['n_msgs_batch']
        mv_percent_batch = batch_data['mv_percent_batch']
        price_batch = batch_data['price_batch']
        ss_index_batch = batch_data['ss_index_batch']
        
        # Convert batch data to texts and labels
        texts = self._convert_batch_to_texts(word_batch, n_words_batch)
        labels = y_batch[:, -1]  # Get labels for the last day
        ticker = stock_batch[0]  # Get ticker for the batch
        
        # Filter texts using Meaning-Aware Selection
        filtered_texts, relevance_mask = self.meaning_aware.filter_texts(texts, ticker)
        
        # Create a new word_batch based on filtered_texts for PEN model
        # This is a simplification. In a real scenario, you'd need to map filtered_texts
        # back to the PEN model's expected input format (e.g., word IDs, adjusted n_words_batch, etc.)
        # For now, we'll proceed assuming the PEN model can take the filtered texts (or their embeddings)
        # and the original y_batch, T_batch etc. are still relevant for the filtered items.

        # Create a mask for y_batch based on relevance_mask if PEN expects only relevant items' labels
        # This depends on how PEN model handles partial batches or masked inputs.
        # Assuming y_batch, T_batch etc. are indexed in a way that relevance_mask can be applied.
        # This part needs careful implementation based on PEN model's input structure.
        
        # For simplicity in this example, we'll assume that the PEN model's feed_dict
        # can handle the original batch data and we'll use the relevance_mask
        # to select the relevant predictions and labels for accuracy calculation.
        
        batch_metrics = {
            'total_texts': len(texts),
            'relevant_texts': sum(relevance_mask),
            'relevance_rate': sum(relevance_mask) / len(texts) if texts else 0,
            'y_pred_relevant': [], # Store predictions for relevant items
            'y_true_relevant': []  # Store true labels for relevant items
        }
        
        if sum(relevance_mask) > 0: # Proceed only if there are relevant texts
            # Get embeddings for filtered texts
            # embeddings = self._get_embeddings(filtered_texts) # Placeholder for now

            # Prepare feed_dict for PEN model
            # IMPORTANT: The PEN model might expect inputs aligned with filtered_texts.
            # This example uses original batch data, which might not be correct
            # if the model expects a batch derived *only* from filtered_texts.
            # You may need to create a new batch_data_filtered based on relevance_mask.
            feed_dict = {
                        self.pen_model.word_ph: word_batch,
                        self.pen_model.stock_ph: stock_batch,
                        self.pen_model.T_ph: T_batch,
                        self.pen_model.n_words_ph: n_words_batch,
                        self.pen_model.n_msgs_ph: n_msgs_batch,
                        self.pen_model.y_ph: y_batch,
                        self.pen_model.mv_percent_ph: mv_percent_batch,
                        self.pen_model.price_ph: price_batch,
                        self.pen_model.ss_index_ph: ss_index_batch,
                self.pen_model.is_training_phase: False,
                # Add other placeholders like dropout if necessary, set to eval mode (e.g., 0.0)
                self.pen_model.dropout_mel_in: 0.0,
                self.pen_model.dropout_mel: 0.0,
                self.pen_model.dropout_ce: 0.0,
                self.pen_model.dropout_vmd_in: 0.0,
                self.pen_model.dropout_vmd: 0.0,
            }

            with tf.device(self.device):
                # We need a persistent session for evaluation if model is loaded once
                # For now, creating session per batch, assuming model is not loaded via load_pen_model
                # If using load_pen_model, session management needs to be handled at a higher level.
                # This is a simplified version.
                with tf.Session() as sess: # This session should ideally be managed outside if model is pre-loaded
                    sess.run(tf.global_variables_initializer()) # Initialize if not loaded
                    # If a model is loaded via load_pen_model, this init might not be needed or could conflict.
                    # Consider how the PEN model's graph and session are managed.

                    # Get predictions (y_T_ for final day prediction as in Executor.py)
                    predictions = sess.run(self.pen_model.y_T_, feed_dict=feed_dict)
                    
                    # Assuming predictions and labels correspond to the original texts' order
                    # We need to map relevance_mask to the predictions and labels
                    # The shape of predictions and labels needs to be compatible with relevance_mask
                    
                    # Example: if predictions are [pred_text1, pred_text2, ...]
                    # and relevance_mask is [True, False, True, ...]
                    # Then relevant_predictions = [pred_text1, pred_text3]
                    # And relevant_labels = [label_text1, label_text3]

                    # This mapping is crucial and depends on how texts, relevance_mask, predictions,
                    # and labels are structured and aligned.
                    # For now, let's assume relevance_mask can be directly applied to select
                    # the corresponding elements from predictions and labels.
                    # This assumes that each item in relevance_mask corresponds to an item in predictions/labels.
                    
                    # y_batch[:, -1] are the labels for the last day, as used before.
                    # Ensure this aligns with what self.pen_model.y_T_ predicts.
                    if predictions.shape[0] == len(relevance_mask) and y_batch[:, -1].shape[0] == len(relevance_mask):
                        relevant_predictions = predictions[relevance_mask]
                        relevant_labels = y_batch[:, -1][relevance_mask]

                        batch_metrics['y_pred_relevant'].extend(relevant_predictions.tolist())
                        batch_metrics['y_true_relevant'].extend(relevant_labels.tolist())
                    else:
                        # This case needs careful handling. It means the relevance_mask
                        # cannot be directly applied to filter predictions/labels.
                        # This could happen if PEN model operates on a different granularity
                        # than the texts filtered by MeaningAwareSelection.
                        print("Warning: Shape mismatch between predictions/labels and relevance_mask. Skipping accuracy calculation for this batch.")
                        # Fallback or error handling needed here.
                        # For now, accuracy will be 0 if this happens.
                
        return batch_metrics
        
    def _convert_batch_to_texts(self, word_batch: np.ndarray, n_words_batch: np.ndarray) -> List[str]:
        """Convert word IDs to text strings"""
        texts = []
        for i in range(word_batch.shape[0]):  # For each sample in batch
            for j in range(word_batch.shape[1]):  # For each day
                for k in range(word_batch.shape[2]):  # For each message
                    n_words = n_words_batch[i, j, k]
                    if n_words > 0:
                        word_ids = word_batch[i, j, k, :n_words]
                        text = ' '.join([str(wid) for wid in word_ids])
                        texts.append(text)
        return texts
        
    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Convert texts to embeddings.
        This is a placeholder. You MUST implement this based on your model's requirements.
        For example, using a Sentence Transformer model or an LLM's embedding endpoint.
        The output shape should be (len(texts), embedding_dim).
        """
        print("Warning: _get_embeddings is a placeholder. Returning zero embeddings.")
        # Placeholder: replace with actual embedding generation logic
        # Example: embedding_dim = self.pen_model.word_embedding_dim or similar
        embedding_dim = 768 # Replace with actual embedding dimension
        return np.zeros((len(texts), embedding_dim))
        
    def run_evaluation(self, num_batches: int, batch_size: int, pen_model_checkpoint_path: str = None) -> Dict[str, Any]:
        """
        Run evaluation for specified number of batches.
        
        Args:
            num_batches: Number of batches to evaluate
            batch_size: Size of each batch
            
        Returns:
            Dictionary with overall evaluation metrics
        """
        total_metrics = {
            'total_batches': 0,
            'total_texts': 0,
            'total_relevant_texts': 0,
            'all_y_pred_relevant': [], # Collect all relevant predictions
            'all_y_true_relevant': []  # Collect all relevant true labels
        }

        # Initialize TensorFlow session for the duration of the evaluation if a model is loaded
        self.eval_session = None
        if pen_model_checkpoint_path:
            try:
                # This simplified load_pen_model creates and closes a session.
                # For a persistent session, tf.Session() should be created here and passed around,
                # or self.pen_model should be loaded into a session managed by the evaluator.
                # For now, we'll rely on evaluate_batch to handle its session,
                # which is not ideal if a checkpoint is loaded once.
                # A better approach:
                self.tf_config_eval = tf.ConfigProto(allow_soft_placement=True)
                self.tf_config_eval.gpu_options.allow_growth = True
                self.eval_session = tf.Session(graph=self.pen_model.graph, config=self.tf_config_eval) # Use PEN model's graph
                self.eval_session.run(tf.global_variables_initializer()) # Initialize vars in this session
                
                checkpoint_dir = os.path.dirname(pen_model_checkpoint_path)
                ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    self.saver.restore(self.eval_session, ckpt.model_checkpoint_path)
                    print(f"PEN model restored from {ckpt.model_checkpoint_path} for evaluation run.")
                else:
                    print(f"Warning: PEN model checkpoint not found at {checkpoint_dir}. Predictions will be from an uninitialized model.")

            except Exception as e:
                print(f"Error setting up TensorFlow session or loading model: {str(e)}")
                # Decide if evaluation should proceed with an uninitialized model or stop
                # For now, it will proceed but predictions might be meaningless.
                if self.eval_session:
                    self.eval_session.close()
                self.eval_session = None # Ensure it's None if setup failed

        print(f"Starting evaluation with {num_batches} batches...")
        
        # Get batch generator
        batch_gen = self.data_pipe.batch_gen('test')  # Use test phase for evaluation
        
        for batch_idx in range(num_batches):
            try:
                # Get next batch
                batch_data = next(batch_gen)
                if not batch_data:
                    break
                    
                # Evaluate batch
                # Pass the session to evaluate_batch if it's managed here
                batch_metrics = self.evaluate_batch_with_session(batch_data, self.eval_session) # New method
                
                # Update total metrics
                total_metrics['total_batches'] += 1
                total_metrics['total_texts'] += batch_metrics['total_texts']
                total_metrics['total_relevant_texts'] += batch_metrics['relevant_texts']
                total_metrics['all_y_pred_relevant'].extend(batch_metrics.get('y_pred_relevant', []))
                total_metrics['all_y_true_relevant'].extend(batch_metrics.get('y_true_relevant', []))
                
                # Print progress
                if (batch_idx + 1) % 10 == 0:
                    self._print_progress(batch_idx + 1, total_metrics)
                    
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {str(e)}")
                continue
                
        # Calculate final metrics
        final_metrics = self._calculate_final_metrics(total_metrics)
        self._print_final_results(final_metrics)

        if self.eval_session:
            self.eval_session.close() # Close the session after evaluation
            print("TensorFlow session closed.")
        
        return final_metrics
        
    def evaluate_batch_with_session(self, batch_data: Dict[str, Any], sess: tf.Session = None) -> Dict[str, Any]:
        """
        Evaluate a single batch of data using a provided TensorFlow session.
        Args:
            batch_data: Dictionary containing batch data
            sess: Existing TensorFlow session to use for running the PEN model.
                  If None, a new session will be created and closed per call (not recommended for performance).
        Returns:
            Dictionary with evaluation metrics for the batch
        """
        # Extract data from batch
        word_batch = batch_data['word_batch']
        stock_batch = batch_data['stock_batch']
        y_batch = batch_data['y_batch']
        T_batch = batch_data['T_batch']
        n_words_batch = batch_data['n_words_batch']
        n_msgs_batch = batch_data['n_msgs_batch']
        mv_percent_batch = batch_data['mv_percent_batch']
        price_batch = batch_data['price_batch']
        ss_index_batch = batch_data['ss_index_batch']
        
        texts = self._convert_batch_to_texts(word_batch, n_words_batch)
        # Using y_batch[:, -1] as true labels for the final day prediction.
        # Ensure this aligns with what pen_model.y_T_ predicts.
        true_labels_full = y_batch[:, -1] 
        ticker = stock_batch[0] 
        
        filtered_texts, relevance_mask = self.meaning_aware.filter_texts(texts, ticker)
        
        batch_metrics = {
            'total_texts': len(texts),
            'relevant_texts': sum(relevance_mask),
            'relevance_rate': sum(relevance_mask) / len(texts) if texts else 0,
            'y_pred_relevant': [],
            'y_true_relevant': []
        }
        
        if sum(relevance_mask) > 0:
            # IMPORTANT: The PEN model feed_dict here uses the *original* full batch data.
            # `relevance_mask` is then used to pick out predictions and labels *after* the model run.
            # This assumes the PEN model can process the full batch and its output aligns row-wise
            # with the original texts/items such that `relevance_mask` can be applied.
            # If MeaningAwareSelection changes the structure/number of items fed to PEN,
            # then `word_batch`, `y_batch`, etc., would need to be filtered *before* `sess.run`.

            feed_dict = {
                self.pen_model.word_ph: word_batch,
                self.pen_model.stock_ph: stock_batch,
                self.pen_model.T_ph: T_batch,
                self.pen_model.n_words_ph: n_words_batch,
                self.pen_model.n_msgs_ph: n_msgs_batch,
                self.pen_model.y_ph: y_batch, # Original y_batch
                self.pen_model.mv_percent_ph: mv_percent_batch,
                self.pen_model.price_ph: price_batch,
                self.pen_model.ss_index_ph: ss_index_batch,
                self.pen_model.is_training_phase: False,
                self.pen_model.dropout_mel_in: 0.0,
                self.pen_model.dropout_mel: 0.0,
                self.pen_model.dropout_ce: 0.0,
                self.pen_model.dropout_vmd_in: 0.0,
                self.pen_model.dropout_vmd: 0.0,
            }

            predictions = None
            if sess: # Use provided session
                predictions = sess.run(self.pen_model.y_T_, feed_dict=feed_dict)
            else: # Create temporary session (less efficient)
                print("Warning: Creating temporary TF session in evaluate_batch_with_session. For better performance, provide a session to run_evaluation.")
                with tf.Session(graph=self.pen_model.graph) as temp_sess: # Use PEN model's graph
                    temp_sess.run(tf.global_variables_initializer()) # Required if model not loaded into this graph structure before
                    # This assumes self.pen_model.graph is the correct graph.
                    # If a checkpoint was loaded into a different session, this won't work as expected
                    # unless self.pen_model.graph is the one with loaded weights.
                    predictions = temp_sess.run(self.pen_model.y_T_, feed_dict=feed_dict)
            
            if predictions is not None:
                # Assuming predictions and true_labels_full are NumPy arrays.
                # And that their first dimension corresponds to the items in `texts`.
                # `relevance_mask` should be a boolean array of the same length.
                if predictions.shape[0] == len(relevance_mask) and true_labels_full.shape[0] == len(relevance_mask):
                    relevant_indices = np.where(relevance_mask)[0]
                    
                    # Filter predictions and true labels using these indices
                    y_pred_relevant = predictions[relevant_indices]
                    y_true_relevant = true_labels_full[relevant_indices]

                    batch_metrics['y_pred_relevant'].extend(y_pred_relevant.tolist())
                    batch_metrics['y_true_relevant'].extend(y_true_relevant.tolist())
                else:
                    print(f"Warning: Shape mismatch. Predictions shape: {predictions.shape}, True labels shape: {true_labels_full.shape}, Relevance mask length: {len(relevance_mask)}. Skipping metric calculation for this batch.")
        
        return batch_metrics
        
    def _print_progress(self, batch_idx: int, metrics: Dict[str, int]) -> None:
        """Print progress during evaluation"""
        print(f"Processed {batch_idx} batches...")
        print(f"Relevance rate: {metrics['total_relevant_texts']/metrics['total_texts']:.2%}")
        if metrics['total_relevant_texts'] > 0:
            print(f"Accuracy on relevant texts: {metrics['y_pred_relevant'][-1]/metrics['y_true_relevant'][-1] if metrics['y_true_relevant'][-1] > 0 else 0:.2%}")
        print("---")
        
    def _calculate_final_metrics(self, metrics: Dict[str, int]) -> Dict[str, float]:
        """Calculate final evaluation metrics using pen_metrics.eval_res"""
        y_true_relevant = np.array(metrics.get('all_y_true_relevant', []))
        y_pred_relevant = np.array(metrics.get('all_y_pred_relevant', []))

        # Placeholder for loss_list if not calculated per batch or not needed for eval_res
        # eval_res from Executor.py expects gen_loss_list, y_list, y_list_
        # Here, y_list corresponds to y_true_relevant and y_list_ to y_pred_relevant (binarized if needed)
        
        num_relevant_predictions = len(y_true_relevant)
        # eval_res expects y_pred_relevant to be binarized probabilities (0 or 1)
        # Assuming y_pred_relevant from PEN model are probabilities, threshold them.
        y_pred_relevant_binary = (y_pred_relevant > 0.5).astype(float)

        # `gen_n_acc` and `gen_size` for pen_metrics.eval_res
        # n_accurate expects y (true) and y_ (predicted probabilities)
        # For relevant items:
        # This part needs the raw probabilities (y_pred_relevant) and true labels (y_true_relevant)
        # before binarization for n_accurate.
        
        # We need to compute n_acc based on relevant items
        # pen_metrics.n_accurate seems to take raw predictions (probabilities) and true labels
        # Let's simulate how it might be used or adapt.
        # Directly calculate accuracy for relevant items here for clarity.
        correct_relevant_predictions = 0
        if num_relevant_predictions > 0:
             correct_relevant_predictions = np.sum(y_pred_relevant_binary == y_true_relevant)

        # The `pen_metrics.eval_res` function might require specific inputs like loss list.
        # If we don't have batch losses, we can't directly use it as is.
        # We'll compute metrics directly here based on collected predictions.
        
        # For now, calculating accuracy and MCC manually.
        # You might want to adapt or use parts of pen_metrics.eval_res if applicable.
        mcc = 0.0
        if num_relevant_predictions > 0:
            # MCC calculation requires true positives, true negatives, false positives, false negatives
            # This can be done using sklearn.metrics.matthews_corrcoef or manually
            try:
                # Using a simple way to get TP, TN, FP, FN for binary classification
                tp = np.sum((y_true_relevant == 1) & (y_pred_relevant_binary == 1))
                tn = np.sum((y_true_relevant == 0) & (y_pred_relevant_binary == 0))
                fp = np.sum((y_true_relevant == 0) & (y_pred_relevant_binary == 1))
                fn = np.sum((y_true_relevant == 1) & (y_pred_relevant_binary == 0))
                
                numerator = (tp * tn) - (fp * fn)
                denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
                if denominator == 0:
                    mcc = 0.0 # Or handle as per standard practice (e.g., if only one class predicted)
                else:
                    mcc = numerator / denominator
            except Exception as e:
                print(f"Could not calculate MCC: {e}")
                mcc = 0.0 # Default to 0 if calculation fails


        return {
            'total_batches': metrics['total_batches'],
            'total_texts': metrics['total_texts'],
            'total_relevant_texts': metrics['total_relevant_texts'],
            'relevance_rate': metrics['total_relevant_texts'] / metrics['total_texts'] if metrics['total_texts'] > 0 else 0,
            'accuracy_on_relevant': correct_relevant_predictions / num_relevant_predictions if num_relevant_predictions > 0 else 0,
            'mcc_on_relevant': mcc, # Added MCC
            'total_relevant_predictions': num_relevant_predictions
        }
        
    def _print_final_results(self, metrics: Dict[str, float]) -> None:
        """Print final evaluation results"""
        print("\nFinal Results:")
        print(f"Total batches processed: {metrics['total_batches']}")
        print(f"Total texts: {metrics['total_texts']}")
        print(f"Total relevant texts: {metrics['total_relevant_texts']}")
        print(f"Relevance rate: {metrics['relevance_rate']:.2%}")
        if metrics['total_relevant_predictions'] > 0:
            print(f"Accuracy on relevant texts: {metrics['accuracy_on_relevant']:.2%}")
            print(f"MCC on relevant texts: {metrics['mcc_on_relevant']:.4f}")
        else:
            print("No relevant texts were processed for PEN model evaluation.")

def main():
    parser = argparse.ArgumentParser(description='Evaluate Meaning-Aware Selection with PEN')
    parser.add_argument('--classifier_type', type=str, default='tfidf_logreg',
                      choices=['tfidf_logreg', 'tfidf_rf', 'tfidf_svm', 'bert_small'],
                      help='Type of classifier to use')
    parser.add_argument('--classifier_path', type=str, default=None,
                      help='Path to classifier model')
    parser.add_argument('--batch_size', type=int, default=config.batch_size,
                      help='Batch size for evaluation')
    parser.add_argument('--num_batches', type=int, default=100,
                      help='Number of batches to evaluate')
    parser.add_argument('--device', type=str, default='cpu',
                      help='Device to run on (e.g., \'cpu\' or \'cuda\')')
    parser.add_argument('--use_real_llm', action='store_true', help='Use real LLM instead of MockLLM')
    parser.add_argument('--llm_api_key', type=str, default=None, help='API key for the LLM service')
    parser.add_argument('--llm_model_name', type=str, default="gpt-3.5-turbo", help='Name of the LLM model')
    parser.add_argument('--pen_model_checkpoint_path', type=str, default=None, help='Path to the PEN model checkpoint (e.g., ./save/model.ckpt)')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = MeaningAwareEvaluator(device=args.device)
    
    # Set up mock LLM and classifier
    if args.use_real_llm:
        if not args.llm_api_key:
            print("Error: LLM API key must be provided when using real LLM.")
            sys.exit(1)
        evaluator.setup_real_llm(llm_api_key=args.llm_api_key, llm_model_name=args.llm_model_name)
    else:
        evaluator.setup_llm() # Default to MockLLM
        
    # Set up classifier (optional)
    if args.classifier_path:
        evaluator.setup_classifier(classifier_type=args.classifier_type, classifier_path=args.classifier_path)
    
    # Run evaluation
    # Pass PEN model checkpoint path to run_evaluation
    evaluator.run_evaluation(
        num_batches=args.num_batches, 
        batch_size=args.batch_size,
        pen_model_checkpoint_path=args.pen_model_checkpoint_path
    )

if __name__ == "__main__":
    main() 