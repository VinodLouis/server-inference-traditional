import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import Optional


class ModelLoader:
    """Load and cache ML models with flexible device support"""
    
    def __init__(self):
        # Default device for initialization only
        self.default_device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model_cache = {}
        self.processor_cache = {}
        self.tokenizer_cache = {}
        print(f"Default device: {self.default_device}")
    
    def get_device(self, device: str = "auto") -> torch.device:
        """
        Get device based on parameter
        
        Args:
            device: "auto", "cpu", "mps", or "cuda"
        
        Returns:
            torch.device
        """
        if device == "auto":
            return self.default_device
        elif device == "cpu":
            return torch.device("cpu")
        elif device == "mps":
            if torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                print("⚠️ MPS not available, falling back to CPU")
                return torch.device("cpu")
        elif device == "cuda":
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                print("⚠️ CUDA not available, falling back to CPU")
                return torch.device("cpu")
        else:
            raise ValueError(f"Unknown device: {device}")
    
    def load_mobilenetv2(self,  device: str = "auto"):
        """
        Load MobileNetV2 for image classification
        
        Args:
            num_classes: Number of output classes (default: 10 for CIFAR-10)
            device: Device to load model on ("auto", "cpu", "mps", "cuda")
        
        Returns:
            model on specified device
        """
        target_device = self.get_device(device)
        cache_key = f"mobilenetv2_{device}"
        
        if cache_key in self.model_cache:
            return self.model_cache[cache_key]
        
        try:
            print(f"Loading MobileNetV2 on {target_device}...")
            model = torch.hub.load(
                "chenyaofo/pytorch-cifar-models", 
                "cifar10_mobilenetv2_x1_0", 
                pretrained=True
            )
            model = model.to(target_device)
            model.eval()
            self.model_cache[cache_key] = model
            print(f"✅ MobileNetV2 loaded on {target_device}")
            return model
        except Exception as e:
            print(f"❌ Error loading MobileNetV2: {e}")
            raise
    
    def load_resnet20(self, device: str = "auto"):
        """
        Load ResNet-20 pretrained on CIFAR-10
        
        Args:
            device: Device to load model on ("auto", "cpu", "mps", "cuda")
        
        Returns:
            model on specified device
        """
        target_device = self.get_device(device)
        cache_key = f"resnet20_{device}"
        
        if cache_key in self.model_cache:
            return self.model_cache[cache_key]
        
        try:
            print(f"Loading ResNet-20 on {target_device}...")
            model = torch.hub.load(
                "chenyaofo/pytorch-cifar-models", 
                "cifar10_resnet20", 
                pretrained=True
            )
            model = model.to(target_device)
            model.eval()
            self.model_cache[cache_key] = model
            print(f"✅ ResNet-20 loaded on {target_device}")
            return model
        except Exception as e:
            print(f"❌ Error loading ResNet-20: {e}")
            raise
    
    def load_distilbert(
        self, 
        model_name="mansoorhamidzadeh/ag-news-bert-classification",
        device: str = "auto"
    ):
        """
        Load a BERT classifier fine-tuned on the AG News dataset
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to load model on ("auto", "cpu", "mps", "cuda")
        
        Returns:
            (model, tokenizer) on specified device
        """
        target_device = self.get_device(device)
        cache_key = f"agnews_{device}"
        
        # Return cached if available
        if cache_key in self.model_cache:
            return self.model_cache[cache_key], self.tokenizer_cache[cache_key]
        
        try:
            print(f"Loading DistilBERT on {target_device}...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            model = model.to(target_device)
            model.eval()
            
            self.model_cache[cache_key] = model
            self.tokenizer_cache[cache_key] = tokenizer
            print(f"✅ DistilBERT loaded on {target_device}")
            return model, tokenizer
        except Exception as e:
            print(f"❌ Error loading DistilBERT: {e}")
            raise
    
    def clear_cache(self):
        """Clear model cache to free memory"""
        self.model_cache.clear()
        self.processor_cache.clear()
        self.tokenizer_cache.clear()
        print("✅ Model cache cleared")
    
    def get_cache_info(self):
        """Get information about cached models"""
        return {
            "models": list(self.model_cache.keys()),
            "tokenizers": list(self.tokenizer_cache.keys()),
            "count": len(self.model_cache),
        }