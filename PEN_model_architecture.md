# PEN Dual-Pathway Model Architecture

This document describes the architecture, workflow, and meaning of the `Model_tx_lf_dual_path.py` implementation, which extends the original PEN (Prediction-Explanation Network) framework presented in the paper "PEN: Prediction-Explanation Network to Forecast Stock Price Movement with Better Explainability."

## Overview

The `Model_tx_lf_dual_path.py` implements an enhanced version of PEN that processes both text (tweets/news) and price data to predict stock price movements. The key innovation is the introduction of a dual-pathway architecture that explicitly models the bidirectional relationship between text and price data.

## Architecture Components

### 1. Text Embedding Layer (TEL)

**Purpose**: Encodes raw text messages into dense vector representations.

**Implementation**:
- Uses pre-trained word embeddings as input
- Processes each message with a bidirectional GRU/LSTM
- Applies Zoneout regularization (a more stable alternative to dropout for RNNs)

**Output**: Message embeddings with shape [batch_size, max_n_days, max_n_msgs, msg_embed_size]


### 2. Dual-Pathway Shared Representation Learning  (DP-SRL)

## Overview

The Dual-Pathway Shared Representation Learning (DP-SRL) module enhances the original Shared Representation Learning architecture by explicitly distinguishing between causal (text → price) and responsive (price → text) relationships. This separation enables the model to distinguish between texts that predict price movements and texts that merely describe them.

## Architecture

The DP-SRL architecture consists of four main components:

### 2.1. Dual-Stream Text Selection Unit (DS-TSU)

This component implements two separate attention mechanisms:
- **Causal Attention**: Identifies text that might influence future price movements
- **Responsive Attention**: Identifies text that responds to previous price movements

Each pathway maintains its own attention weights and parameters, ensuring separation of the two different types of relationships.

### 2.2. Cross-Pathway Communication Module

This module enables controlled information exchange between the causal and responsive pathways:
- Projects pathway states into a shared space for cross-attention
- Uses adaptive gating mechanisms to control information flow between pathways
- Computes interaction strength based on memory relevance to current text
- Enhances each pathway with relevant information from the other pathway

The implementation in `MSINModule_dual_path.py` includes:
- Pathway-specific projection matrices (`causal_proj` and `responsive_proj`)
- Memory-to-text projection matrices for cross-pathway influence
- Dynamic interaction gates based on current context and memory relevance
- Configurable pathway interaction strengths

```python
# Cross-pathway communication configuration
cell.cross_communication_enabled = True  # Enable/disable cross-pathway communication
cell.causal_to_responsive_strength = 0.5  # How strongly causal affects responsive
cell.responsive_to_causal_strength = 0.3  # How strongly responsive affects causal
```

The method `_cross_pathway_communication` handles this exchange by:
1. Projecting memory states into a shared representation space
2. Computing relevance scores between text and memory from the opposite pathway
3. Applying relevance-based gates to control information flow
4. Enhancing each pathway's representation with weighted information from the other

### 2.3. Dual-Stream Text Memory Unit (DS-TMU)

The dual memory system maintains separate memory states:
- **Causal Memory (v_causal)**: Stores information from text that predicts prices
- **Responsive Memory (v_responsive)**: Stores information from text that responds to prices

Each memory has its own set of control gates (forget, output, new text) to selectively update and maintain information.

### 2.4. Context-Aware Information Fusion Unit (CA-IFU)

This component dynamically weighs the importance of each information pathway based on the current context:
- **Price Gate (k_price)**: Controls the influence of price information
- **Causal Gate (k_causal)**: Controls the influence of causal text information
- **Responsive Gate (k_responsive)**: Controls the influence of responsive text information

The implementation in `MSINModule_dual_path.py` includes:
- Separate processing streams for price, causal text, and responsive text information
- Context-dependent gating mechanisms using the current price and hidden state
- Sigmoid activation followed by normalization to ensure gates sum to 1
- Weighted fusion of the three information sources

```python
# Context-dependent gates
context_features = tf.concat([X, h], axis=1)
k_price = tf.sigmoid(tf.matmul(context_features, self.W_k_price) + self.b_k_price)
k_causal = tf.sigmoid(tf.matmul(context_features, self.W_k_causal) + self.b_k_causal)
k_responsive = tf.sigmoid(tf.matmul(context_features, self.W_k_responsive) + self.b_k_responsive)

# Normalize gates to sum to 1
k_sum = k_price + k_causal + k_responsive + 1e-10
k_price = k_price / k_sum
k_causal = k_causal / k_sum
k_responsive = k_responsive / k_sum

# Final fusion with adaptive weighting
H = k_price * hx + k_causal * hv_causal + k_responsive * hv_responsive
```

This approach allows the model to adaptively focus on the most relevant information source based on the current market context, emphasizing price information during highly volatile periods, causal text during trend-setting news events, or responsive text during reaction phases.
### 3. Deep Recurrent Generator (DRG)

**Purpose**: Generates predictions from the fused representation.

**Variants**:
3.1. **Generative (VAE-based)**:
   - `_create_vmd_with_h_rec()`: Uses hidden state recurrence
   - `_create_vmd_with_zh_rec()`: Uses both hidden and latent variable recurrence
   - Incorporates KL divergence in the loss function
   
3.2. **Discriminative**:
   - `_create_discriminative_vmd()`: Standard RNN without latent variables
   - Direct mapping from input to output

**Output**: Hidden states `g` and predictions `y`

### 4. Temporal Attention Prediction (TAP)

**Purpose**: Enhances predictions by attending to relevant historical information.

**Implementation**:
- Computes attention scores between current and previous days
- Combines information across time using weighted sum
- Generates final prediction `y_T`

## Learning Objective

The model optimizes a complex learning objective that balances several components:

1. **Primary Prediction Objective**: Maximizes the log-likelihood of the correct price movement prediction.

2. **Variational Regularization**: For generative variants, includes a KL divergence term between prior and posterior latent distributions.

3. **Pathway-Specific Regularization**:
   - **Causal Pathway (λ=0.1)**: Stronger regularization to encourage focused attention on truly predictive signals.
   - **Responsive Pathway (λ=0.05)**: Lighter regularization allowing for broader attention to descriptive text.
   
4. **Pathway Differentiation (λ=0.02)**: Penalizes attention overlap between pathways to ensure they capture distinct aspects of the text-price relationship.

5. **Temporal Attention Weighting**: Applies the parameter α to weight the importance of auxiliary timesteps relative to the target prediction timestep.

The full objective function can be represented as:

```
Loss = -[log p(y_T|x_T) + α·Σ v_t·log p(y_t|x_t)] 
       + KL[q(z|x,y) || p(z|x)]
       + λ_causal·R_causal + λ_responsive·R_responsive + λ_diff·R_diff
```

Where:
- R_causal and R_responsive are regularization terms for each pathway
- R_diff is the pathway differentiation penalty
- v_t represents the temporal attention weights


## Workflow

The model processes data through the following pipeline as implemented in `Model_tx_lf_dual_path.py`:

1. **Input Processing**:
   - Word embeddings → Message embeddings via `_create_msg_embed_layer()`
   - Message embeddings are processed with potential dropout for regularization

2. **Dual-Pathway Processing** via `_build_mie()`:
   - Creates `MSINCell` with dual pathway architecture
   - Configures cross-pathway communication parameters
   - Creates initial state with zero tensors for hidden state and both memory pathways
   - Processes inputs through `dual_msin.dynamic_dual_msin()`:
     ```python
     self.x, self.P_dict, state = dual_msin.dynamic_dual_msin(
         cell=cell,
         inputs=self.price,
         s_inputs=self.msg_embed,
         sequence_length=self.T_ph,
         initial_state=initial_state,
         dtype=tf.float32
     )
     ```
   - Extracts pathway-specific attention weights and gate values:
     ```python
     self.P_causal = self.P_dict['causal']
     self.P_responsive = self.P_dict['responsive']
     self.k_price = self.P_dict['k_price']
     self.k_causal = self.P_dict['k_causal']
     self.k_responsive = self.P_dict['k_responsive']
     ```

3. **Prediction Generation**:
   - Fused representation processed by variant-specific recurrent generators:
     - Generative h-recurrence via `_create_vmd_with_h_rec()`
     - Generative zh-recurrence via `_create_vmd_with_zh_rec()`
     - Discriminative model via `_create_discriminative_vmd()`
   - Hidden states processed by temporal attention in `_build_temporal_att()`
   - Final prediction generated with pathway-specific information

4. **Training** via `_create_optimizer()` and specialized ATA functions:
   - Multi-component loss optimization with pathway-specific regularization:
     ```python
     self.loss = (self.loss + 
                lambda_causal * tf.reduce_mean(-P_causal_reg) + 
                lambda_responsive * tf.reduce_mean(-P_responsive_reg) + 
                lambda_diff * tf.reduce_mean(P_diff))
     ```
   - Gradient clipping for training stability (`self.clip`)
   - Optional learning rate decay for SGD optimization
   - Monitoring of pathway correlation and influence metrics

## Key Innovations Over Original PEN

1. **Explicit Dual-Pathway Architecture**:
   - Original PEN: Implicit fusion of text and price
   - Enhanced model: Explicit modeling of text→price and price→text relationships with separate memory states
   - Implementation: `DualPathwayMSINStateTuple` with separate `v_causal` and `v_responsive` memory components

2. **Cross-Pathway Communication**:
   - Original PEN: No explicit modeling of information exchange between pathways
   - Enhanced model: Controlled information flow between pathways with tunable strengths
   - Implementation: `_cross_pathway_communication()` method with projection matrices and adaptive gating

3. **Pathway-Specific Regularization**:
   - Original PEN: Uniform regularization for all components
   - Enhanced model: Different regularization strengths for causal (λ=0.1) and responsive (λ=0.05) pathways
   - Implementation: Specialized loss components in `_create_generative_ata()` and `_create_discriminative_ata()`

4. **Context-Aware Adaptive Fusion**:
   - Original PEN: Fixed fusion weights
   - Enhanced model: Dynamic weighting of pathways based on market context
   - Implementation: Context-dependent gates using current price and hidden state information

5. **Advanced Monitoring and Metrics**:
   - Original PEN: Basic performance metrics
   - Enhanced model: Tracks pathway correlation and influence during training
   - Implementation: Dedicated metrics section in loss function with TensorFlow summaries

## Practical Applications

The dual-pathway architecture provides several practical benefits:

1. **Better Explainability**:
   - Can distinguish between text that drives prices vs. text that reflects prices
   - Provides pathway-specific explanations

2. **Adaptive to Market Conditions**:
   - Relies more on price-driven pathway during high volatility
   - Relies more on text-driven pathway during stable periods

3. **Robust to Noisy Text**:
   - Noise-aware loss can downweight unreliable text signals
   - Enhances performance in real-world scenarios with varying text quality

## Model Variants and Configuration

### Model Variants

The implementation supports several model variants, as defined in `Model_tx_lf_dual_path.py`:

1. **Generative Models with Variational Approaches**:
   - `h-recurrence` variant (`vmd_rec = 'h'`): Updates hidden state based on previous state and current input
   - `zh-recurrence` variant (`vmd_rec = 'zh'`): Updates both latent variable and hidden state recursively
   - Both include KL divergence terms in the loss function

2. **Discriminative Model** (`variant_type = 'discriminative'`):
   - Uses a simpler objective without KL divergence terms
   - Focuses purely on prediction accuracy
   - Has specialized regularization for dual pathways

### Cell Type Options

The model supports multiple cell types for different components:

```python
# Message Embedding Layer (MEL) cell types
self.mel_cell_type = config_model['mel_cell_type']  # 'ln-lstm', 'gru', or 'basic'

# Variational Movement Decoder (VMD) cell types
self.vmd_cell_type = config_model['vmd_cell_type']  # 'ln-lstm' or 'gru'
```

### Cross-Pathway Configuration

The dual-pathway behavior can be finely tuned:

```python
# Cross-pathway communication settings
self.enable_cross_pathway = config_model.get('enable_cross_pathway', True)
self.c2r_strength = config_model.get('causal_to_responsive_strength', 0.5)
self.r2c_strength = config_model.get('responsive_to_causal_strength', 0.3)
```

### Training Configuration

Various training parameters can be adjusted:

```python
# Optimizer settings
self.opt = config_model['opt']  # 'sgd' or 'adam'
self.lr = config_model['lr']  # Learning rate
self.decay_step = config_model['decay_step']  # For exponential decay
self.decay_rate = config_model['decay_rate']  # For exponential decay
self.momentum = config_model['momentum']  # For SGD with momentum
self.clip = config_model['clip']  # Gradient clipping threshold

# KL divergence annealing (for variational models)
self.kl_lambda_anneal_rate = config_model['kl_lambda_anneal_rate']
self.kl_lambda_start_step = config_model['kl_lambda_start_step']
self.use_constant_kl_lambda = config_model['use_constant_kl_lambda']
self.constant_kl_lambda = config_model['constant_kl_lambda']

# Temporal attention weight
self.alpha = config_model['alpha']  # Weight for auxiliary timesteps
```

This comprehensive configuration system enables detailed experimentation across model architectures, training regimes, and pathway interactions, allowing adaptation to different financial markets and data sources. 