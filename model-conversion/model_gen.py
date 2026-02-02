import torch
import torch.onnx
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path

def convert_all():
    """Convert all 3 models to ONNX"""
    
    # Create models folder
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    print(f"Models will be saved to: {models_dir}/\n")
    
    # ======================================================================
    # 1. MOBILENETV2 x1_0
    # ======================================================================
    print("1. Converting MobileNetV2 x1_0...")
    try:
        model1 = torch.hub.load(
            "chenyaofo/pytorch-cifar-models",
            "cifar10_mobilenetv2_x1_0",
            pretrained=True
        )
        model1.eval()
        
        dummy1 = torch.randn(1, 3, 32, 32)
        
        torch.onnx.export(
            model1,
            dummy1,
            models_dir / "mobilenetv2_x1_0.onnx",
            input_names=['input'],
            output_names=['output'],
            opset_version=12,
            do_constant_folding=True
        )
        
        size_mb = (models_dir / "mobilenetv2_x1_0.onnx").stat().st_size / (1024*1024)
        print(f"   ✓ Saved: models/mobilenetv2_x1_0.onnx ({size_mb:.1f} MB)\n")
    except Exception as e:
        print(f"   ✗ Error: {e}\n")
    
    # ======================================================================
    # 2. RESNET20
    # ======================================================================
    print("2. Converting ResNet20...")
    try:
        model2 = torch.hub.load(
            "chenyaofo/pytorch-cifar-models",
            "cifar10_resnet20",
            pretrained=True
        )
        model2.eval()
        
        dummy2 = torch.randn(1, 3, 32, 32)
        
        torch.onnx.export(
            model2,
            dummy2,
            models_dir / "resnet20.onnx",
            input_names=['input'],
            output_names=['output'],
            opset_version=12,
            do_constant_folding=True
        )
        
        size_mb = (models_dir / "resnet20.onnx").stat().st_size / (1024*1024)
        print(f"   ✓ Saved: models/resnet20.onnx ({size_mb:.1f} MB)\n")
    except Exception as e:
        print(f"   ✗ Error: {e}\n")
    
    # ======================================================================
    # 3. DISTILBERT
    # ======================================================================
    print("3. Converting DistilBERT (AG News)...")
    try:
        model_name = "mansoorhamidzadeh/ag-news-bert-classification"
        model3 = AutoModelForSequenceClassification.from_pretrained(model_name)
        model3.eval()
        
        dummy_ids = torch.randint(0, 30522, (1, 128))
        dummy_mask = torch.ones(1, 128, dtype=torch.long)
        
        torch.onnx.export(
            model3,
            (dummy_ids, dummy_mask),
            models_dir / "distilbert_ag_news.onnx",
            input_names=['input_ids', 'attention_mask'],
            output_names=['logits'],
            opset_version=14,
            do_constant_folding=True
        )
        
        size_mb = (models_dir / "distilbert_ag_news.onnx").stat().st_size / (1024*1024)
        print(f"   ✓ Saved: models/distilbert_ag_news.onnx ({size_mb:.1f} MB)\n")
    except Exception as e:
        print(f"   ✗ Error: {e}\n")
    
    # ======================================================================
    # SUMMARY
    # ======================================================================
    print("="*60)
    print("CONVERSION COMPLETE")
    print("="*60)
    
    files = list(models_dir.glob("*.onnx"))
    total_size = sum(f.stat().st_size for f in files) / (1024*1024)
    
    print(f"\nModels saved in: {models_dir.absolute()}\n")
    for f in sorted(files):
        size_mb = f.stat().st_size / (1024*1024)
        print(f"  • {f.name} ({size_mb:.1f} MB)")
    
    print(f"\nTotal size: {total_size:.1f} MB")
    print(f"Total models: {len(files)}")
    print("\n✓ All models converted and stored!")

if __name__ == "__main__":
    convert_all()