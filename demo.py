import numpy as np
import torch
from model import Network
import joblib

# Load example data
x, y = joblib.load('example_data/example_data.pkl')
tx = torch.as_tensor(x)
ty = torch.as_tensor(y)

# Load model
network = Network(1, 3) 
network.load_state_dict(torch.load('model/model.pt')[0])

# Perform prediction
with torch.no_grad():
    y_p = network(tx)
y_ = torch.argmax(y_p, dim=1)

print(f'True Label: {list(y)}')
print(f'Predicted Result: {list(y_.numpy())}')