# Dual-Pathway Shared Representation Learning (DP-SRL)

## Overview

The Dual-Pathway Shared Representation Learning (DP-SRL) module enhances the original Shared Representation Learning architecture by explicitly distinguishing between causal (text → price) and responsive (price → text) relationships. This separation enables the model to distinguish between texts that predict price movements and texts that merely describe them.

## Architecture

The DP-SRL architecture consists of four main components:

### 1. Dual-Stream Text Selection Unit (DS-TSU)

This component implements two separate attention mechanisms:
- **Causal Attention**: Identifies text that might influence future price movements
- **Responsive Attention**: Identifies text that responds to previous price movements

Each pathway maintains its own attention weights and parameters, ensuring separation of the two different types of relationships.

### 2. Cross-Pathway Communication Module

This module enables controlled information exchange between the causal and responsive pathways:
- Projects pathway states into a shared space for cross-attention
- Uses adaptive gating mechanisms to control information flow between pathways
- Computes interaction strength based on memory relevance to current text
- Enhances each pathway with relevant information from the other pathway

The cross-pathway communication can be enabled/disabled and has tunable strength parameters for each direction (causal→responsive and responsive→causal).

### 3. Dual-Stream Text Memory Unit (DS-TMU)

The dual memory system maintains separate memory states:
- **Causal Memory (v_causal)**: Stores information from text that predicts prices
- **Responsive Memory (v_responsive)**: Stores information from text that responds to prices

Each memory has its own set of control gates (forget, output, new text) to selectively update and maintain information.

### 4. Context-Aware Information Fusion Unit (CA-IFU)

This component dynamically weighs the importance of each information pathway based on the current context:
- **Price Gate (k_price)**: Controls the influence of price information
- **Causal Gate (k_causal)**: Controls the influence of causal text information
- **Responsive Gate (k_responsive)**: Controls the influence of responsive text information

The gates are normalized to sum to 1, allowing the model to adaptively focus on the most relevant information source.

## Implementation Details

The implementation follows the Adapter Pattern:

1. The core functionality is implemented in `MSINCell` in `MSINModule_dual_path.py`:
   - Multiple attention mechanisms (causal and responsive)
   - Dual memory states in `DualPathwayMSINStateTuple`
   - Context-aware gates for information weighting

2. For compatibility with the existing MSIN architecture:
   - Attention outputs are packed into a single tensor
   - A wrapper class `DualPathwayMSIN` handles unpacking the attention values

3. Usage is simplified through the `dynamic_dual_msin` method, which:
   - Calls the parent's `dynamic_msin` method
   - Unpacks the combined attention values into a structured dictionary
   - Returns outputs, attention dictionary, and final state

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

### Model Variants

The implementation supports several variants:

1. **Generative Model**: Uses a variational approach with two recurrence options:
   - `h` recurrence: Updates hidden state based on previous state and current input
   - `zh` recurrence: Updates both latent variable and hidden state recursively

2. **Discriminative Model**: Uses a simpler objective without KL divergence terms, focusing purely on prediction accuracy.

### Training Metrics

The model tracks several metrics during training to monitor pathway behavior:

```python
# Calculate correlation between pathway attentions
correlation = tf.reduce_mean(
    tf.reduce_sum(tf.multiply(self.P_causal, self.P_responsive), axis=[1, 2])
)

# Track information flow through pathways
causal_influence = tf.reduce_mean(self.k_causal)
responsive_influence = tf.reduce_mean(self.k_responsive)
```

These metrics help understand how the pathways interact and their relative importance in different market contexts.

## Key Benefits

- **Improved Explainability**: Clearly distinguishes between causal and responsive text
- **Enhanced Prediction**: Better feature selection by separating predictive from reactive text
- **Adaptive Weighting**: Dynamic adaptation to different market contexts
- **Cross-Pathway Learning**: Enables controlled information sharing between causal and responsive pathways
- **Tunable Communication**: Configurable strength parameters for cross-pathway information exchange
- **Backwards Compatibility**: Works with existing code through the Adapter Pattern

## Usage Example

```python
# Create the MSINCell with dual pathway architecture
cell = MSINCell(
    input_size=price_size,
    num_units=num_units,
    v_size=embedding_size,
    max_n_msgs=max_n_msgs * 5  # 5x for combined attention values
)

# Configure cross-pathway communication (optional)
cell.cross_communication_enabled = True  # Enable/disable cross-pathway communication
cell.causal_to_responsive_strength = 0.5  # How strongly causal affects responsive
cell.responsive_to_causal_strength = 0.3  # How strongly responsive affects causal

# Create the dual pathway MSIN wrapper
dual_msin = DualPathwayMSIN()

# Create initial state
initial_state = DualPathwayMSINStateTuple(
    h=tf.zeros([batch_size, num_units]),
    v_causal=tf.zeros([batch_size, embedding_size]),
    v_responsive=tf.zeros([batch_size, embedding_size])
)

# Run the dual pathway SRL
outputs, P_dict, final_state = dual_msin.dynamic_dual_msin(
    cell=cell,
    inputs=price_features,
    s_inputs=msg_embeddings,
    sequence_length=sequence_lengths,
    initial_state=initial_state,
    dtype=tf.float32
)

# Access attention components
causal_attention = P_dict['causal']
responsive_attention = P_dict['responsive']
k_price = P_dict['k_price']
k_causal = P_dict['k_causal']
k_responsive = P_dict['k_responsive']
```

For a complete example, see `use_dual_pathway_srl.py` and `test_dual_pathway_srl.py`.

## Reference

For more details on the original Shared Representation Learning architecture, see the paper: "Enhancing Stock Movement Prediction with Adversarial Training". 