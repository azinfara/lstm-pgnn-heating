import json
import torch

def load_model(model_path='model/lstm_pgnn_v2_2_baseline.pt', 
               config_path='model/model_config.json'):
    """Load model with configuration"""
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Initialize model
    from src.model import LSTMPINNHeatingV2_2_Enhanced
    model = LSTMPINNHeatingV2_2_Enhanced(
        **config['architecture'],
        **config['physics_parameters']
    )
    
    # Load weights
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    return model, config

# Usage
model, config = load_model()
print(f"Loaded {config['name']} model with {config['performance']['total_parameters']} parameters")