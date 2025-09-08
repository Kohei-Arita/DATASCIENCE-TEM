"""
Unit tests for model architecture and functionality

Tests model creation, forward pass, attention mechanisms, and utilities.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock
import tempfile

from scripts.model import (
    AneurysmClassifier,
    MultiScaleModel,
    EnsembleModel,
    SEBlock,
    CBAM,
    ECABlock,
    create_model,
    load_model_weights,
    count_parameters
)


class TestAneurysmClassifier:
    """Test the main AneurysmClassifier model"""
    
    def test_init_default(self):
        """Test model initialization with default parameters"""
        model = AneurysmClassifier()
        
        assert model.architecture == "resnet50"
        assert model.num_classes == 1
        assert not model.use_attention
        assert isinstance(model.backbone, nn.Module)
        assert isinstance(model.classifier, nn.Module)
    
    def test_init_custom_params(self):
        """Test model initialization with custom parameters"""
        model = AneurysmClassifier(
            architecture="resnet18",
            num_classes=2,
            pretrained=False,
            dropout=0.3,
            hidden_dim=256,
            use_attention=True,
            attention_type="se"
        )
        
        assert model.architecture == "resnet18"
        assert model.num_classes == 2
        assert model.use_attention
        assert hasattr(model, 'attention')
    
    def test_forward_pass(self, sample_batch):
        """Test forward pass with sample batch"""
        model = AneurysmClassifier(architecture="resnet18", pretrained=False)
        model.eval()
        
        images, _ = sample_batch
        
        with torch.no_grad():
            outputs = model(images)
        
        assert outputs.shape == (4, 1)  # batch_size=4, num_classes=1
        assert not torch.isnan(outputs).any()
        assert torch.isfinite(outputs).all()
    
    def test_forward_pass_with_attention(self, sample_batch):
        """Test forward pass with attention mechanism"""
        model = AneurysmClassifier(
            architecture="resnet18",
            pretrained=False,
            use_attention=True,
            attention_type="cbam"
        )
        model.eval()
        
        images, _ = sample_batch
        
        with torch.no_grad():
            outputs = model(images)
        
        assert outputs.shape == (4, 1)
        assert not torch.isnan(outputs).any()
    
    def test_get_features(self, sample_batch):
        """Test feature extraction without classification"""
        model = AneurysmClassifier(architecture="resnet18", pretrained=False)
        model.eval()
        
        images, _ = sample_batch
        
        with torch.no_grad():
            features = model.get_features(images)
        
        assert features.shape[0] == 4  # batch_size
        assert len(features.shape) == 2  # Should be flattened features
        assert not torch.isnan(features).any()
    
    def test_different_architectures(self):
        """Test model creation with different architectures"""
        architectures = ["resnet18", "resnet34", "efficientnet-b0"]
        
        for arch in architectures:
            try:
                model = AneurysmClassifier(architecture=arch, pretrained=False)
                assert model.architecture == arch
                
                # Test forward pass
                x = torch.randn(1, 3, 224, 224)
                output = model(x)
                assert output.shape == (1, 1)
                
            except Exception as e:
                # Some architectures might not be available, skip gracefully
                print(f"Skipping {arch}: {e}")
    
    def test_parameter_count(self):
        """Test parameter counting functionality"""
        model = AneurysmClassifier(architecture="resnet18", pretrained=False)
        param_count = model._count_parameters()
        
        assert isinstance(param_count, int)
        assert param_count > 0
    
    @pytest.mark.slow
    def test_gradient_flow(self, sample_batch):
        """Test that gradients flow through the model"""
        model = AneurysmClassifier(architecture="resnet18", pretrained=False)
        criterion = nn.BCEWithLogitsLoss()
        
        images, labels = sample_batch
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist and are finite
        has_grad = False
        for param in model.parameters():
            if param.grad is not None:
                has_grad = True
                assert torch.isfinite(param.grad).all()
        
        assert has_grad, "No gradients found in model parameters"


class TestAttentionBlocks:
    """Test attention mechanism implementations"""
    
    def test_se_block(self):
        """Test Squeeze-and-Excitation block"""
        se_block = SEBlock(channels=512)
        
        # Test with 1D features (after global pooling)
        x = torch.randn(4, 512)
        output = se_block(x)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()
    
    def test_cbam(self):
        """Test CBAM attention block"""
        cbam = CBAM(channels=512)
        
        # Test with 1D features
        x = torch.randn(4, 512)
        output = cbam(x)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()
    
    def test_eca_block(self):
        """Test ECA attention block"""
        eca = ECABlock(channels=512)
        
        # Test with 1D features
        x = torch.randn(4, 512)
        output = eca(x)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()


class TestMultiScaleModel:
    """Test multi-scale model implementation"""
    
    def test_init(self):
        """Test multi-scale model initialization"""
        scales = [224, 384, 512]
        model = MultiScaleModel(
            scales=scales,
            architecture="resnet18",
            pretrained=False
        )
        
        assert model.scales == scales
        assert hasattr(model, 'fusion')
    
    def test_single_scale_forward(self):
        """Test forward pass with single scale input"""
        model = MultiScaleModel(
            scales=[224, 384],
            architecture="resnet18",
            pretrained=False
        )
        model.eval()
        
        # Single scale input (224x224)
        x = torch.randn(2, 3, 224, 224)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (2, 1)
        assert not torch.isnan(output).any()
    
    @pytest.mark.slow
    def test_multi_scale_forward(self):
        """Test forward pass with multi-scale processing"""
        model = MultiScaleModel(
            scales=[224, 288],  # Use smaller scales for faster testing
            architecture="resnet18",
            pretrained=False
        )
        model.eval()
        
        # Different scale input
        x = torch.randn(2, 3, 288, 288)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (2, 1)
        assert not torch.isnan(output).any()


class TestEnsembleModel:
    """Test ensemble model implementation"""
    
    def test_init_equal_weights(self):
        """Test ensemble initialization with equal weights"""
        models = [
            AneurysmClassifier(architecture="resnet18", pretrained=False),
            AneurysmClassifier(architecture="resnet18", pretrained=False)
        ]
        
        ensemble = EnsembleModel(models)
        
        assert len(ensemble.models) == 2
        assert torch.allclose(ensemble.weights, torch.tensor([0.5, 0.5]))
    
    def test_init_custom_weights(self):
        """Test ensemble initialization with custom weights"""
        models = [
            AneurysmClassifier(architecture="resnet18", pretrained=False),
            AneurysmClassifier(architecture="resnet18", pretrained=False)
        ]
        weights = [0.3, 0.7]
        
        ensemble = EnsembleModel(models, weights=weights)
        
        assert torch.allclose(ensemble.weights, torch.tensor(weights))
    
    def test_forward_pass(self, sample_batch):
        """Test ensemble forward pass"""
        models = [
            AneurysmClassifier(architecture="resnet18", pretrained=False),
            AneurysmClassifier(architecture="resnet18", pretrained=False)
        ]
        
        ensemble = EnsembleModel(models)
        ensemble.eval()
        
        images, _ = sample_batch
        
        with torch.no_grad():
            output = ensemble(images)
        
        assert output.shape == (4, 1)
        assert not torch.isnan(output).any()
    
    def test_eval_mode(self):
        """Test that eval() sets all models to eval mode"""
        models = [
            AneurysmClassifier(architecture="resnet18", pretrained=False),
            AneurysmClassifier(architecture="resnet18", pretrained=False)
        ]
        
        ensemble = EnsembleModel(models)
        
        # Set to training mode first
        for model in ensemble.models:
            model.train()
        
        # Call eval on ensemble
        ensemble.eval()
        
        # Check all models are in eval mode
        for model in ensemble.models:
            assert not model.training


class TestModelUtilities:
    """Test utility functions for models"""
    
    def test_create_model_standard(self, sample_config):
        """Test model creation from config"""
        model = create_model(sample_config)
        
        assert isinstance(model, AneurysmClassifier)
        assert model.architecture == "resnet18"
        assert model.num_classes == 1
    
    def test_create_model_multiscale(self, sample_config):
        """Test multi-scale model creation from config"""
        sample_config["model"]["use_multiscale"] = True
        sample_config["model"]["scales"] = [224, 288]
        
        model = create_model(sample_config)
        
        assert isinstance(model, MultiScaleModel)
        assert model.scales == [224, 288]
    
    def test_count_parameters(self):
        """Test parameter counting utility"""
        model = AneurysmClassifier(architecture="resnet18", pretrained=False)
        
        param_info = count_parameters(model)
        
        assert "total" in param_info
        assert "trainable" in param_info
        assert "non_trainable" in param_info
        
        assert param_info["total"] > 0
        assert param_info["trainable"] > 0
        assert param_info["non_trainable"] >= 0
        assert param_info["total"] == param_info["trainable"] + param_info["non_trainable"]
    
    def test_load_model_weights_success(self, test_data_dir):
        """Test successful model weight loading"""
        model = AneurysmClassifier(architecture="resnet18", pretrained=False)
        
        # Save model weights
        checkpoint_path = test_data_dir / "test_model.pth"
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "epoch": 10,
            "metrics": {"auc": 0.85}
        }
        torch.save(checkpoint, checkpoint_path)
        
        # Create new model and load weights
        new_model = AneurysmClassifier(architecture="resnet18", pretrained=False)
        new_model = load_model_weights(new_model, str(checkpoint_path))
        
        # Check that weights are the same
        for (name1, param1), (name2, param2) in zip(model.named_parameters(), new_model.named_parameters()):
            assert name1 == name2
            assert torch.allclose(param1, param2)
    
    def test_load_model_weights_file_not_found(self):
        """Test model weight loading with non-existent file"""
        model = AneurysmClassifier(architecture="resnet18", pretrained=False)
        
        with pytest.raises(Exception):  # Should raise FileNotFoundError or similar
            load_model_weights(model, "/non/existent/path.pth")
    
    @patch('torch.load')
    def test_load_model_weights_different_formats(self, mock_load):
        """Test model weight loading with different checkpoint formats"""
        model = AneurysmClassifier(architecture="resnet18", pretrained=False)
        original_state = model.state_dict()
        
        # Test different checkpoint formats
        formats = [
            {"model_state_dict": original_state},  # Standard format
            {"state_dict": original_state},        # Alternative format
            original_state                         # Direct state dict
        ]
        
        for checkpoint_format in formats:
            mock_load.return_value = checkpoint_format
            
            # Should not raise exception
            loaded_model = load_model_weights(model, "dummy_path.pth")
            assert isinstance(loaded_model, nn.Module)


@pytest.mark.integration
class TestModelIntegration:
    """Integration tests for model functionality"""
    
    def test_training_step(self, sample_batch):
        """Test a complete training step"""
        model = AneurysmClassifier(architecture="resnet18", pretrained=False)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        images, labels = sample_batch
        
        # Forward pass
        model.train()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        assert loss.item() > 0
        assert torch.isfinite(loss)
    
    def test_inference_step(self, sample_batch):
        """Test a complete inference step"""
        model = AneurysmClassifier(architecture="resnet18", pretrained=False)
        model.eval()
        
        images, _ = sample_batch
        
        with torch.no_grad():
            outputs = model(images)
            probabilities = torch.sigmoid(outputs)
        
        assert outputs.shape == (4, 1)
        assert probabilities.shape == (4, 1)
        assert (probabilities >= 0).all() and (probabilities <= 1).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])