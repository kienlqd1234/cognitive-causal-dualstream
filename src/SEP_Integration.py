#!/usr/local/bin/python
import os
import tensorflow as tf
import numpy as np
import sys
import json
from datetime import datetime
from tqdm import tqdm

# Import PEN components
from DataPipe import DataPipe
from SEP_Integrated_Model import Model
from metrics import eval_res, n_accurate

# Import new SEP integration components
from PENReflectAgent import PENReflectAgent
from RewardModel import SimpleRewardModel, TransformerRewardModel, create_explanation_pairs
from LLMInterface import create_llm, MockLLM
from PPOTrainer import PPOTrainer

# Import Meaning-Aware Selection module
from MeaningAwareSelection import MeaningAwareSelection
from RelevanceClassifier import RelevanceClassifier

class SEP_PEN_Integration:
    """
    Integration class that combines PEN's Vector of Salience (VoS) explainability
    with SEP's self-reflection and automated evaluation approach.
    """
    
    def __init__(self, config_path="src/config.yml", llm_config=None, device=None):
        self.config_path = config_path
        
        # Default LLM config if none provided
        if llm_config is None:
            self.llm_config = {
                "type": "mock",  # Can be "mock", "openai", "transformer"
                "model": "gpt-3.5-turbo",
                "max_tokens": 500,
                "temperature": 0.7
            }
        else:
            self.llm_config = llm_config
            
        # Initialize components
        self.model = Model(config_path)
        self.model.assemble_graph()  # Assemble the TensorFlow graph
        self.pipe = DataPipe()
        self.llm = create_llm(self.llm_config)
        self.reward_model = SimpleRewardModel()  # Start with simple reward model
        
        # Initialize Meaning-Aware Selection module
        self.meaning_aware_selection = MeaningAwareSelection(self.llm, {
            'prompt_template': "Given the sentence: \"{text}\", is it relevant to predicting {ticker} stock price movement?\nAnswer with only 'Relevant' or 'Irrelevant'.",
            'confidence_threshold': 0.3  # Lower threshold for relevance
        })
        
        # Initialize Relevance Classifier (for faster inference)
        self.relevance_classifier = RelevanceClassifier(model_type='tfidf_logreg')
        self.use_relevance_classifier = True  # Enable by default
        
        # Try to load pre-trained classifier if available
        classifier_path = os.path.join("results", "relevance_classifier.pkl")
        if os.path.exists(classifier_path):
            try:
                self.relevance_classifier.load(classifier_path)
                print("Loaded pre-trained relevance classifier")
            except Exception as e:
                print(f"Could not load pre-trained classifier: {e}")
                self.use_relevance_classifier = False
        
        # Create TF saver and session config
        self.saver = tf.train.Saver()
        self.tf_config = tf.ConfigProto(allow_soft_placement=True)
        self.tf_config.gpu_options.allow_growth = True
        
        # Tracking for training
        self.agents = []
        self.train_stats = {
            'epochs': 0,
            'batches': 0,
            'correct_predictions': 0,
            'reflection_improvements': 0,
            'avg_rewards': [],
            'relevance_stats': {
                'total_texts': 0,
                'relevant_texts': 0,
                'irrelevant_texts': 0
            }
        }
        
        # Results storage
        self.results_dir = "results/sep_pen_integration"
        os.makedirs(self.results_dir, exist_ok=True)
        
    def _initialize_session(self):
        """Initialize TensorFlow session"""
        sess = tf.Session(config=self.tf_config)
        
        # Initialize word table
        word_table_init = self.pipe.init_word_table()
        feed_table_init = {self.model.word_table_init: word_table_init}
        sess.run(tf.global_variables_initializer(), feed_dict=feed_table_init)
        
        # Try to restore checkpoint if available
        checkpoint = tf.train.get_checkpoint_state(os.path.dirname(self.model.tf_checkpoint_file_path))
        if checkpoint and checkpoint.model_checkpoint_path:
            # Restore saved vars
            reader = tf.train.NewCheckpointReader(checkpoint.model_checkpoint_path)
            restore_dict = dict()
            for v in tf.all_variables():
                tensor_name = v.name.split(':')[0]
                if reader.has_tensor(tensor_name):
                    print('has tensor: {0}'.format(tensor_name))
                    restore_dict[tensor_name] = v

            checkpoint_saver = tf.train.Saver(restore_dict)
            checkpoint_saver.restore(sess, checkpoint.model_checkpoint_path)
            print(f'Model restored from {checkpoint.model_checkpoint_path}')
        else:
            print('Starting new session')
            
        return sess
    
    def _create_feed_dict(self, batch_dict, is_training=True):
        """Create feed dict for model from batch dict"""
        return {
            self.model.is_training_phase: is_training,
            self.model.batch_size: batch_dict['batch_size'],
            self.model.stock_ph: batch_dict['stock_batch'],
            self.model.T_ph: batch_dict['T_batch'],
            self.model.n_words_ph: batch_dict['n_words_batch'],
            self.model.n_msgs_ph: batch_dict['n_msgs_batch'],
            self.model.y_ph: batch_dict['y_batch'],
            self.model.price_ph: batch_dict['price_batch'],
            self.model.mv_percent_ph: batch_dict['mv_percent_batch'],
            self.model.word_ph: batch_dict['word_batch'],
            self.model.ss_index_ph: batch_dict['ss_index_batch'],
            # Set dropout to 0 when not training
            self.model.dropout_mel_in: 0.0 if not is_training else None,
            self.model.dropout_mel: 0.0 if not is_training else None,
            self.model.dropout_ce: 0.0 if not is_training else None,
            self.model.dropout_vmd_in: 0.0 if not is_training else None,
            self.model.dropout_vmd: 0.0 if not is_training else None,
        }
    
    def _extract_texts_from_batch(self, batch_dict):
        """Extract all texts from a batch for explanation"""
        # This is a simplified placeholder - the actual implementation
        # would depend on how texts are stored in your batches
        texts = []
        for i in range(batch_dict['batch_size']):
            sample_texts = []
            for day in range(min(5, self.model.max_n_days)):  # Limit to 5 days for simplicity
                for msg in range(min(10, self.model.max_n_msgs)):  # Limit to 10 messages
                    if batch_dict['n_words_batch'][i][day][msg] > 0:
                        # Create dummy text for now - in real implementation, 
                        # you'd extract actual text content
                        sample_texts.append(f"Text from day {day}, message {msg}")
            texts.append(sample_texts)
        return texts
    
    def _apply_meaning_aware_selection(self, batch_dict, texts_list):
        """
        Apply Meaning-Aware Selection to filter out irrelevant texts
        
        Args:
            batch_dict: Batch dictionary
            texts_list: List of texts for each sample in the batch
            
        Returns:
            Modified batch dictionary with irrelevant texts filtered out
        """
        if not texts_list:
            return batch_dict
            
        # Deep copy the batch dict to avoid modifying the original
        filtered_batch = dict(batch_dict)
        
        # For each sample in the batch
        for i in range(batch_dict['batch_size']):
            ticker = f"Stock_{batch_dict['stock_batch'][i]}"
            
            # Apply Meaning-Aware Selection
            if self.use_relevance_classifier and self.relevance_classifier and self.relevance_classifier.is_trained:
                # Use trained classifier for faster inference
                judgments, _ = self.relevance_classifier.predict(texts_list[i])
            else:
                # Use LLM for relevance filtering
                _, judgments = self.meaning_aware_selection.filter_texts(texts_list[i], ticker)
                
            # Update stats
            self.train_stats['relevance_stats']['total_texts'] += len(judgments)
            self.train_stats['relevance_stats']['relevant_texts'] += sum(judgments)
            self.train_stats['relevance_stats']['irrelevant_texts'] += len(judgments) - sum(judgments)
            
            # Create a mask for the word embeddings
            # This is a simplified version - you would need to adapt this to your data structure
            for day in range(min(5, self.model.max_n_days)):
                for msg, is_relevant in enumerate(judgments):
                    if msg < self.model.max_n_msgs and not is_relevant:
                        # Set n_words to 0 for irrelevant messages to effectively ignore them
                        filtered_batch['n_words_batch'][i][day][msg] = 0
        
        return filtered_batch
    
    def _create_agents_from_batch(self, batch_dict, predictions, actuals, attention_weights):
        """Create PENReflectAgents from batch data"""
        agents = []
        texts_list = self._extract_texts_from_batch(batch_dict)
        
        for i in range(batch_dict['batch_size']):
            ticker = f"Stock_{batch_dict['stock_batch'][i]}"
            
            # Safely get target value from actuals
            try:
                target_val = actuals[i][0]
                if isinstance(target_val, np.ndarray) or isinstance(target_val, tf.Tensor):
                    # Try to convert tensor to scalar
                    try:
                        if hasattr(target_val, 'numpy'):
                            target_val = float(target_val.numpy())
                        else:
                            target_val = float(target_val)
                    except:
                        target_val = 0.5  # Default value
                target = 1 if target_val > 0.5 else 0
            except:
                target = 0  # Default to negative class
            
            # Safely get price data
            try:
                prices = batch_dict['price_batch'][i]
                if isinstance(prices, tf.Tensor) and hasattr(prices, 'numpy'):
                    prices = prices.numpy()
            except:
                prices = np.array([0.5, 0.3, 0.4])  # Default prices
            
            agent = PENReflectAgent(
                ticker=ticker,
                texts=texts_list[i],
                prices=prices,
                target=target
            )
            
            # Safely set prediction
            try:
                pred_val = predictions[i][0]
                if isinstance(pred_val, np.ndarray) or isinstance(pred_val, tf.Tensor):
                    try:
                        if hasattr(pred_val, 'numpy'):
                            pred_val = float(pred_val.numpy())
                        else:
                            pred_val = float(pred_val)
                    except:
                        pred_val = 0.5  # Default value
                agent.prediction = pred_val
            except:
                agent.prediction = 0.5  # Default to uncertain prediction
            
            # Safely set attention weights
            try:
                weights = attention_weights[i]
                if isinstance(weights, tf.Tensor) and hasattr(weights, 'numpy'):
                    weights = weights.numpy()
                agent.vos_weights = weights
            except:
                # Create dummy weights if needed
                agent.vos_weights = np.ones(len(texts_list[i])) / len(texts_list[i])
            
            # Extract top texts based on attention weights
            try:
                agent.extract_top_texts(n=2)
            except:
                # If extraction fails, manually set top texts
                if texts_list[i]:
                    agent.top_texts = texts_list[i][:min(2, len(texts_list[i]))]
                else:
                    agent.top_texts = ["No relevant texts found"]
            
            agents.append(agent)
            
        return agents
    
    def train(self, n_epochs=10, train_relevance_classifier=True):
        """Train model using SEP's self-reflection approach"""
        # Create TensorFlow session without using 'with'
        sess = self._initialize_session()
        
        # Collect relevance data for training the classifier
        if train_relevance_classifier:
            relevance_data = {
                'texts': [],
                'labels': []
            }
        
        try:
            # Start from current global step
            # Properly pass session to eval
            start_epoch = sess.run(self.model.global_step) // 100  # Assuming 100 steps per epoch
            
            for epoch in range(start_epoch, start_epoch + n_epochs):
                print(f"Epoch {epoch+1}/{start_epoch + n_epochs}")
                
                # Training batch generator
                train_batch_gen = self.pipe.batch_gen(phase='train')
                
                epoch_agents = []
                epoch_correct = 0
                epoch_total = 0
                
                # Process each batch
                for batch_idx, train_batch_dict in enumerate(train_batch_gen):
                    # Extract texts for relevance filtering
                    texts_list = self._extract_texts_from_batch(train_batch_dict)
                    
                    # Apply Meaning-Aware Selection to filter irrelevant texts
                    filtered_batch_dict = self._apply_meaning_aware_selection(train_batch_dict, texts_list)
                    
                    # If collecting relevance data for classifier training
                    if train_relevance_classifier:
                        for i in range(train_batch_dict['batch_size']):
                            ticker = f"Stock_{train_batch_dict['stock_batch'][i]}"
                            _, judgments = self.meaning_aware_selection.filter_texts(texts_list[i], ticker)
                            
                            # Add to training data
                            relevance_data['texts'].extend(texts_list[i])
                            relevance_data['labels'].extend(judgments)
                    
                    # Create feed dict with filtered batch
                    feed_dict = self._create_feed_dict(filtered_batch_dict, is_training=True)
                    
                    # Forward pass with PEN model
                    ops = [
                        self.model.y_T,       # True labels
                        self.model.y_T_,      # Predictions
                        self.model.loss,      # Loss
                        self.model.P,         # Attention weights from MSIN
                    ]
                    
                    actuals, predictions, batch_loss, vos_weights = sess.run(ops, feed_dict)
                    
                    # Create agents for this batch
                    batch_agents = self._create_agents_from_batch(
                        filtered_batch_dict, predictions, actuals, vos_weights
                    )
                    
                    # Process each agent
                    for agent in batch_agents:
                        # Generate explanation
                        agent.generate_explanation(self.llm)
                        
                        # Run reflection loop
                        agent.run_reflection_loop(self.llm, self.reward_model, max_iterations=2)
                        
                        # Update stats
                        if agent.is_correct():
                            epoch_correct += 1
                        epoch_total += 1
                            
                    # Add to epoch agents
                    epoch_agents.extend(batch_agents)
                    
                    # Save model occasionally
                    if (batch_idx + 1) % 20 == 0:
                        step = sess.run(self.model.global_step)
                        save_path = self.saver.save(sess, self.model.tf_saver_path, global_step=step)
                        print(f"Model saved at step {step} to {save_path}")
                        
                        # Print batch stats
                        batch_accuracy = epoch_correct / max(1, epoch_total)
                        print(f"Batch {batch_idx+1}, Loss: {batch_loss:.4f}, Accuracy: {batch_accuracy:.4f}")
                        print(f"Relevance stats: {self.train_stats['relevance_stats']}")
                
                # Save epoch results
                self._save_epoch_results(epoch, epoch_agents)
                
                # Train the reward model with explanation pairs
                explanation_pairs = create_explanation_pairs(epoch_agents)
                if explanation_pairs:
                    self._train_reward_model(explanation_pairs)
                
                # Train the relevance classifier at the end of each epoch
                if train_relevance_classifier and epoch > 0 and len(relevance_data['texts']) > 100:
                    if self.relevance_classifier is None:
                        self.relevance_classifier = RelevanceClassifier(model_type='tfidf_logreg')
                    
                    print(f"Training relevance classifier with {len(relevance_data['texts'])} examples...")
                    self.relevance_classifier.train(
                        texts=relevance_data['texts'],
                        labels=relevance_data['labels']
                    )
                    
                    # Save the classifier
                    classifier_path = os.path.join(self.results_dir, 'relevance_classifier.pkl')
                    self.relevance_classifier.save(classifier_path)
                    
                    # Use the classifier for future epochs
                    self.use_relevance_classifier = True
                    
                    # Clear training data to save memory
                    relevance_data = {
                        'texts': [],
                        'labels': []
                    }
                
                # Save final model for this epoch
                step = sess.run(self.model.global_step)
                save_path = self.saver.save(sess, self.model.tf_saver_path, global_step=step)
                print(f"Epoch {epoch+1} completed. Model saved to {save_path}")
                
        finally:
            # Close session when done
            sess.close()
    
    def _train_reward_model(self, explanation_pairs):
        """Train the reward model on pairs of explanations"""
        if not explanation_pairs:
            return
            
        print(f"Training reward model with {len(explanation_pairs)} explanation pairs...")
        
        # If using simple reward model, no training needed
        if isinstance(self.reward_model, SimpleRewardModel):
            return
            
        # If using transformer reward model, train it
        if isinstance(self.reward_model, TransformerRewardModel):
            self.reward_model.train(explanation_pairs)
    
    def _save_epoch_results(self, epoch, agents):
        """Save results from epoch to file"""
        if not agents:
            return
            
        # Calculate stats
        correct = sum(1 for a in agents if a.is_correct())
        total = len(agents)
        accuracy = correct / total if total > 0 else 0
        
        # Calculate average reward for explanations
        rewards = [self.reward_model(a.explanation) for a in agents]
        avg_reward = sum(rewards) / len(rewards) if rewards else 0
        
        # Update train stats
        self.train_stats['epochs'] += 1
        self.train_stats['correct_predictions'] += correct
        self.train_stats['avg_rewards'].append(avg_reward)
        
        # Create results dictionary
        results = {
            'epoch': epoch,
            'accuracy': accuracy,
            'avg_reward': avg_reward,
            'relevance_stats': self.train_stats['relevance_stats'],
            'examples': []
        }
        
        # Add sample examples
        for i, agent in enumerate(agents[:5]):  # Just add first 5 for brevity
            example = {
                'ticker': agent.ticker,
                'prediction': float(agent.prediction),
                'target': int(agent.target),
                'explanation': agent.explanation,
                'reflections': agent.reflections
            }
            results['examples'].append(example)
            
        # Save to file
        filename = os.path.join(self.results_dir, f'epoch_{epoch}_results.json')
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"Epoch {epoch} results - Accuracy: {accuracy:.4f}, Avg Reward: {avg_reward:.4f}")
        print(f"Relevance stats: Total: {self.train_stats['relevance_stats']['total_texts']}, " +
              f"Relevant: {self.train_stats['relevance_stats']['relevant_texts']} " +
              f"({self.train_stats['relevance_stats']['relevant_texts']/max(1, self.train_stats['relevance_stats']['total_texts']):.2%})")
    
    def optimize_llm_with_ppo(self, n_iterations=5):
        """Optimize LLM explanations using PPO"""
        # Only applicable if using a trainable LLM
        if self.llm_config['type'] not in ['transformer']:
            print("LLM optimization only supported for transformer-based models")
            return
            
        # Create PPO trainer
        trainer = PPOTrainer(
            llm=self.llm,
            reward_model=self.reward_model
        )
        
        # Sample batch for training
        train_batch_gen = self.pipe.batch_gen(phase='train')
        batch_dict = next(train_batch_gen)
        
        # Create session for evaluation
        sess = self._initialize_session()
        
        try:
            # Extract texts and create agents
            feed_dict = self._create_feed_dict(batch_dict, is_training=False)
            actuals, predictions, _, vos_weights = sess.run(
                [self.model.y_T, self.model.y_T_, self.model.loss, self.model.P],
                feed_dict
            )
            
            agents = self._create_agents_from_batch(
                batch_dict, predictions, actuals, vos_weights
            )
            
            # Run PPO optimization
            print(f"Optimizing LLM with PPO for {n_iterations} iterations...")
            trainer.train(agents, n_iterations=n_iterations)
            
        finally:
            sess.close()
    
    def evaluate(self, phase="test"):
        """Evaluate model on validation or test set"""
        # Create session for evaluation
        sess = self._initialize_session()
        
        try:
            # Evaluation batch generator
            eval_batch_gen = self.pipe.batch_gen(phase=phase)
            
            eval_agents = []
            eval_predictions = []
            eval_actuals = []
            eval_losses = []  # Track losses
            
            # Process each batch
            for batch_idx, eval_batch_dict in enumerate(eval_batch_gen):
                # Extract texts for relevance filtering
                texts_list = self._extract_texts_from_batch(eval_batch_dict)
                
                # Apply Meaning-Aware Selection
                filtered_batch_dict = self._apply_meaning_aware_selection(eval_batch_dict, texts_list)
                
                # Create feed dict
                feed_dict = self._create_feed_dict(filtered_batch_dict, is_training=False)
                
                # Forward pass
                actuals, predictions, batch_loss, vos_weights = sess.run(
                    [self.model.y_T, self.model.y_T_, self.model.loss, self.model.P],
                    feed_dict
                )
                
                # Store predictions, actuals, and loss
                eval_predictions.extend(predictions)
                eval_actuals.extend(actuals)
                eval_losses.append(batch_loss)  # Store batch loss
                
                # Create agents for this batch
                batch_agents = self._create_agents_from_batch(
                    filtered_batch_dict, predictions, actuals, vos_weights
                )
                
                # Generate explanations
                for agent in batch_agents:
                    agent.generate_explanation(self.llm)
                
                # Add to evaluation agents
                eval_agents.extend(batch_agents)
                
                # Print progress
                if (batch_idx + 1) % 10 == 0:
                    print(f"Processed {batch_idx+1} evaluation batches")

                if (batch_idx + 1) == 100:
                    break
            
            # Convert predictions and actuals to numpy arrays if they aren't already
            eval_predictions = np.array(eval_predictions)
            eval_actuals = np.array(eval_actuals)
            
            # Get the predicted class (argmax for one-hot encoded predictions)
            pred_classes = np.argmax(eval_predictions, axis=1)
            actual_classes = np.argmax(eval_actuals, axis=1)
            
            # Calculate evaluation metrics
            gen_n_acc = np.sum(pred_classes == actual_classes)
            gen_size = len(eval_predictions)
            gen_loss_list = eval_losses  # Use collected losses
            
            # Calculate results using eval_res
            results = eval_res(
                gen_n_acc=gen_n_acc,
                gen_size=gen_size,
                gen_loss_list=gen_loss_list,
                y_list=pred_classes,  # Pass class indices instead of one-hot vectors
                y_list_=actual_classes,  # Pass class indices instead of one-hot vectors
                use_mcc=True
            )
            
            # Add explanation rewards
            explanation_rewards = [agent.get_explanation_reward() for agent in eval_agents]
            results['avg_explanation_reward'] = np.mean(explanation_rewards) if explanation_rewards else 0.0
            
            # Add relevance stats
            results['relevance_stats'] = self.train_stats['relevance_stats']
            
            # Print results
            print("\nTest Results:")
            print(f"Accuracy: {results['acc']:.4f}")
            print(f"MCC Score: {results['mcc']:.4f}")
            print(f"Loss: {results['loss']:.4f}")
            print(f"Avg Explanation Reward: {results['avg_explanation_reward']:.4f}")
            print(f"Relevance Filtering: {results['relevance_stats']['relevant_texts']} / " +
                  f"{results['relevance_stats']['total_texts']} texts deemed relevant " +
                  f"({results['relevance_stats']['relevant_texts']/max(1, results['relevance_stats']['total_texts']):.2%})")
            
            return results
            
        finally:
            sess.close()


if __name__ == "__main__":
    # Simple test script
    integration = SEP_PEN_Integration()
    
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        integration.train(n_epochs=5)
    else:
        integration.evaluate(phase="test") 