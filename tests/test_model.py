import torch

from src.model import get_model


def test_model_output_shape():
    model = get_model("resnet18", 10)
    x = torch.randn(4, 3, 32, 32)

    with torch.no_grad():
        output = model(x)

    assert output.shape == (4, 10)


def test_model_has_ten_classes():
    model = get_model("resnet18", 10)

    assert model.fc.out_features == 10


def test_invalid_architecture():
    try:
        get_model("invalid", 10)
        assert False
    except ValueError:
        assert True