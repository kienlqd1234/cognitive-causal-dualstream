#!/usr/local/bin/python
import os
import numpy as np
import torch
from typing import List, Dict, Any, Tuple, Optional
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn

class RelevanceClassifier:
    """
    A lightweight classifier that can be trained on relevance judgments from an LLM
    to provide faster inference for the Meaning-Aware Selection module.
    """
    
    def __init__(self, model_type='tfidf_logreg', model_path=None, device=None):
        """
        Initialize the classifier.
        
        Args:
            model_type: Type of model to use ('tfidf_logreg', 'bert_small', etc.)
            model_path: Path to load model from (if None, a new model will be created)
            device: Device to use for inference ('cpu', 'cuda', etc.)
        """
        self.model_type = model_type
        self.model_path = model_path
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        if model_type == 'tfidf_logreg':
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),  # Use both unigrams and bigrams
                min_df=2,  # Minimum document frequency
                max_df=0.95  # Maximum document frequency
            )
            self.model = LogisticRegression(
                C=1.0,
                class_weight='balanced',
                max_iter=1000
            )
        elif model_type == 'tfidf_rf':
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95
            )
            self.model = RandomForestClassifier(
                n_estimators=200,
                class_weight='balanced',
                max_depth=10,
                min_samples_split=5
            )
        elif model_type == 'tfidf_svm':
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95
            )
            self.model = SVC(
                probability=True,
                class_weight='balanced',
                kernel='rbf',
                C=1.0
            )
        elif 'bert' in model_type:
            # Use a small BERT model for efficiency
            model_name = 'prajjwal1/bert-tiny' if model_type == 'bert_small' else 'bert-base-uncased'
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.encoder = AutoModel.from_pretrained(model_name).to(self.device)
            self.classifier = nn.Sequential(
                nn.Linear(self.encoder.config.hidden_size, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, 1),
                nn.Sigmoid()
            ).to(self.device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        # Load model if path is provided
        if model_path and os.path.exists(model_path):
            self.load(model_path)
            
        self.is_trained = False
        
    def train(self, texts: List[str], labels: List[bool], epochs=5, batch_size=32, validation_split=0.1):
        """
        Train the classifier on labeled data with validation.
        
        Args:
            texts: List of text strings
            labels: List of boolean labels (True for relevant, False for irrelevant)
            epochs: Number of epochs for training (only used for deep models)
            batch_size: Batch size for training (only used for deep models)
            validation_split: Fraction of data to use for validation
        """
        if not texts or not labels:
            raise ValueError("Empty training data")
            
        if len(texts) != len(labels):
            raise ValueError(f"Number of texts ({len(texts)}) doesn't match number of labels ({len(labels)})")
            
        # Convert boolean labels to integers
        int_labels = np.array([1 if label else 0 for label in labels])
        
        # Split into train and validation sets
        indices = np.random.permutation(len(texts))
        split_idx = int(len(texts) * (1 - validation_split))
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        train_texts = [texts[i] for i in train_indices]
        train_labels = int_labels[train_indices]
        val_texts = [texts[i] for i in val_indices]
        val_labels = int_labels[val_indices]
        
        # Train model based on type
        if 'tfidf' in self.model_type:
            # TF-IDF + classic ML model
            X_train = self.vectorizer.fit_transform(train_texts)
            X_val = self.vectorizer.transform(val_texts)
            
            self.model.fit(X_train, train_labels)
            
            # Evaluate on validation set
            val_pred = self.model.predict(X_val)
            val_acc = np.mean(val_pred == val_labels)
            print(f"Validation accuracy: {val_acc:.4f}")
            
        elif 'bert' in self.model_type:
            self._train_bert(train_texts, train_labels, val_texts, val_labels, epochs, batch_size)
                
        self.is_trained = True
        
    def _train_bert(self, train_texts, train_labels, val_texts, val_labels, epochs, batch_size):
        """Train BERT model with validation"""
        # Tokenize texts
        train_encodings = self.tokenizer(train_texts, truncation=True, padding=True, return_tensors='pt')
        val_encodings = self.tokenizer(val_texts, truncation=True, padding=True, return_tensors='pt')
        
        # Create datasets
        train_dataset = torch.utils.data.TensorDataset(
            train_encodings['input_ids'].to(self.device),
            train_encodings['attention_mask'].to(self.device),
            torch.tensor(train_labels, dtype=torch.float32).to(self.device)
        )
        val_dataset = torch.utils.data.TensorDataset(
            val_encodings['input_ids'].to(self.device),
            val_encodings['attention_mask'].to(self.device),
            torch.tensor(val_labels, dtype=torch.float32).to(self.device)
        )
        
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
        
        # Set up optimizer and scheduler
        optimizer = torch.optim.AdamW(self.classifier.parameters(), lr=2e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2)
        
        best_val_acc = 0
        
        for epoch in range(epochs):
            # Training
            self.encoder.train()
            self.classifier.train()
            train_loss = 0
            
            for batch in train_dataloader:
                input_ids, attention_mask, batch_labels = batch
                
                optimizer.zero_grad()
                
                with torch.no_grad():
                    outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
                embeddings = outputs.last_hidden_state[:, 0, :]
                
                predictions = self.classifier(embeddings).squeeze()
                loss = nn.BCELoss()(predictions, batch_labels)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            self.encoder.eval()
            self.classifier.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch in val_dataloader:
                    input_ids, attention_mask, batch_labels = batch
                    
                    outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
                    embeddings = outputs.last_hidden_state[:, 0, :]
                    
                    predictions = self.classifier(embeddings).squeeze()
                    loss = nn.BCELoss()(predictions, batch_labels)
                    
                    val_loss += loss.item()
                    val_correct += ((predictions > 0.5) == batch_labels).sum().item()
                    val_total += len(batch_labels)
            
            val_acc = val_correct / val_total
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss/len(train_dataloader):.4f} - "
                  f"Val Loss: {val_loss/len(val_dataloader):.4f} - Val Acc: {val_acc:.4f}")
            
            scheduler.step(val_acc)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                if self.model_path:
                    self.save(self.model_path)
                
    def predict(self, texts: List[str]) -> Tuple[List[bool], List[float]]:
        """
        Predict relevance for a list of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            Tuple of (relevance_judgments, confidence_scores)
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions")
            
        if 'tfidf' in self.model_type:
            # TF-IDF + classic ML model
            X = self.vectorizer.transform(texts)
            probabilities = self.model.predict_proba(X)[:, 1]
            # Use a lower threshold for relevance (0.3 instead of 0.5)
            predictions = probabilities > 0.3
        elif 'bert' in self.model_type:
            # BERT model
            self.encoder.eval()
            self.classifier.eval()
            
            with torch.no_grad():
                # Tokenize texts
                encodings = self.tokenizer(texts, truncation=True, padding=True, return_tensors='pt')
                input_ids = encodings['input_ids'].to(self.device)
                attention_mask = encodings['attention_mask'].to(self.device)
                
                # Get embeddings
                outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
                embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
                
                # Get predictions
                probabilities = self.classifier(embeddings).squeeze().cpu().numpy()
                # Use a lower threshold for relevance (0.3 instead of 0.5)
                predictions = probabilities > 0.3
                
        return predictions.tolist(), probabilities.tolist()
        
    def save(self, path: str):
        """Save model to disk"""
        if 'tfidf' in self.model_type:
            with open(path, 'wb') as f:
                pickle.dump({
                    'model_type': self.model_type,
                    'vectorizer': self.vectorizer,
                    'model': self.model
                }, f)
        elif 'bert' in self.model_type:
            torch.save({
                'model_type': self.model_type,
                'encoder_state': self.encoder.state_dict(),
                'classifier_state': self.classifier.state_dict()
            }, path)
            
    def load(self, path: str):
        """Load model from disk"""
        if 'tfidf' in self.model_type:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.model_type = data['model_type']
                self.vectorizer = data['vectorizer']
                self.model = data['model']
        elif 'bert' in self.model_type:
            checkpoint = torch.load(path, map_location=self.device)
            self.model_type = checkpoint['model_type']
            self.encoder.load_state_dict(checkpoint['encoder_state'])
            self.classifier.load_state_dict(checkpoint['classifier_state'])
            
        self.is_trained = True 