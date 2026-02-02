# export_tokenizer_json.py
from transformers import AutoTokenizer
from pathlib import Path

def export_tokenizer_json(
    model_name: str = "mansoorhamidzadeh/ag-news-bert-classification",
    out_dir: str = "tokenizer/data",
):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    tokenizer_path = out_dir / "tokenizer.json"

    print(f"Loading tokenizer from: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"Saving tokenizer to: {tokenizer_path}")
    tokenizer.save_pretrained(out_dir)   # writes tokenizer.json (and vocab files)

    if tokenizer_path.exists():
        size_kb = tokenizer_path.stat().st_size / 1024
        print(f"✓ Saved tokenizer.json ({size_kb:.1f} KB)")
    else:
        print("✗ tokenizer.json not found after save_pretrained")

if __name__ == "__main__":
    export_tokenizer_json()
