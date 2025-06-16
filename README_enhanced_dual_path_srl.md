# Enhanced Dual-Path SRL for PEN

This implementation enhances the Shared Representation Learning (SRL) module in PEN with a dual-pathway architecture that distinguishes between "influential text" (text that drives price movements) and "price-driven text" (text that reflects price patterns).

## Architecture

The enhanced dual-path SRL consists of two main branches:

### 1. Text-Driven (TD) Pathway
- Processes text data to predict future price movements
- Identifies influential text through prediction accuracy
- Features temporal attention to capture important patterns over time
- Calculates text influence scores based on prediction quality

### 2. Price-Driven (PD) Pathway  
- Uses price information to guide attention over text
- Identifies text that responds to or reflects price patterns
- Employs price-guided attention mechanism
- Extracts context vectors that represent price-driven text features

### 3. Adaptive Fusion Mechanism
- Detects market volatility to adaptively weight each pathway
- During high volatility: favors price-driven pathway
- During low volatility: favors text-driven pathway
- Uses a gating mechanism for dynamic fusion

## Added Loss Components

The enhanced model includes additional loss components:

1. **Prediction Loss**: Measures accuracy of text-to-price predictions
2. **Pathway Consistency Loss**: Encourages clear specialization between pathways
3. **Volatility-Aware Regularization**: Adjusts pathway weights based on market conditions
4. **Noise-Aware Loss**: Makes the model more robust to noisy tweets by weighting them based on reliability

## Configuration

The enhanced SRL can be configured through the `config_tx_lf_dual_path.yml` file:

```yaml
model:
  # Enhanced SRL parameters
  use_enhanced_srl: True
  td_temporal_attention: True
  volatility_adaptation_factor: 0.3
  residual_connection_weight: 0.1
  pathway_competition_type: 'dynamic'
  
  # Enhanced SRL loss weights
  prediction_loss_weight: 0.1
  consistency_loss_weight: 0.05
  volatility_reg_weight: 0.05
  
  # Noise-aware loss parameters
  use_noise_aware_loss: True
  noise_aware_weight: 0.5
```

## Running the Model

You can train and evaluate the enhanced dual-path SRL model using the provided script:

```bash
# Make the script executable
chmod +x run_enhanced_dual_path_srl.sh

# Run the script
./run_enhanced_dual_path_srl.sh
```

This will:
1. Train the model
2. Evaluate it on the test set
3. Generate visualizations and metrics

## Testing Noise-Aware Loss

You have multiple options to test the noise-aware loss component:

### Option 1: Quick Test Script

```bash
# Make the script executable
chmod +x run_test_noise_aware.sh

# Run with default parameters
./run_test_noise_aware.sh

# Run test only (no training)
./run_test_noise_aware.sh --test-only

# Explore different noise weight values
./run_test_noise_aware.sh --explore-weights

# Explore different volatility factors
./run_test_noise_aware.sh --explore-volatility
```

### Option 2: Direct Python Script

You can also run the test script directly with custom parameters:

```bash
# Run with default parameters
python src/test_noise_aware_loss.py

# Specify custom parameters
python src/test_noise_aware_loss.py --noise_aware_weight 0.7 --volatility_factor 0.5 --test_steps 200

# Run test only (no training)
python src/test_noise_aware_loss.py --test_only
```

Available parameters:
- `--noise_aware_weight`: Weight for the noise-aware loss (default: 0.5)
- `--volatility_factor`: Factor for volatility adaptation (default: 0.3)
- `--test_steps`: Number of training steps to run (default: 100)
- `--test_only`: Run only evaluation without training

This will:
1. Run a shorter training phase with noise-aware loss enabled
2. Generate detailed visualizations about text reliability
3. Analyze which stocks have more reliable vs. noisy social media text
4. Save results in `results/noise_aware_training/` directory

## Visualizations

After running the model, visualizations will be saved in the `results/enhanced_srl_figures` directory:

- Pathway weight distribution
- Pathway weights over time
- Volatility vs. text influence
- Price attention heatmaps
- Accuracy comparison between high/low influence days

Noise-aware test visualizations in `results/noise_aware_training/`:

- Training loss and accuracy
- Noise-aware loss values
- Reliability weights over time
- Top/bottom stocks by text reliability
- Correlation between text influence and reliability

## Analysis

The enhanced dual-path SRL provides several benefits:

1. **Interpretability**: Clearly distinguishes between different types of text influences
2. **Adaptive Learning**: Adjusts to market conditions automatically
3. **Improved Performance**: Leverages both text-driven and price-driven patterns
4. **Robustness**: Less susceptible to noisy or irrelevant social media data
5. **Noise Awareness**: Automatically identifies and downweights potentially noisy data

## Implementation Details

The implementation is found in the following files:

- `src/Model_tx_lf_dual_path.py`: Main model implementation
- `src/Executor_tx_lf_dual_path.py`: Training and evaluation
- `src/test_noise_aware_loss.py`: Specialized test script for noise-aware loss
- `src/visualize_dual_path_srl.py`: Visualization utilities
- `src/main_tx_lf_dual_path.py`: Command-line interface
- `run_enhanced_dual_path_srl.sh`: Full training script
- `run_test_noise_aware.sh`: Noise-aware testing script 