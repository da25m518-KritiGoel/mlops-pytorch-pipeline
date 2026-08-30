import torch
from src.model import get_model

def test_resnet18_output_shape():
    model = get_model()
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    assert y.shape == (2, 10)
