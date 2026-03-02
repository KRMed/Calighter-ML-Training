import os
import json
import numpy as np
import onnxruntime as ort
from transformers import AutoConfig
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import Dataset
from seqeval.metrics import classification_report, f1_score
from itertools import islice
from transformers import AutoTokenizer, AutoModelForTokenClassification
from onnxruntime.quantization.calibrate import CalibrationDataReader 
from optimum.exporters.onnx import main_export
from optimum.onnxruntime import ORTModelForTokenClassification
from onnxruntime.quantization import quantize_dynamic, QuantType, quantize_static, QuantFormat, CalibrationMethod

model_path = "./local_minilm_model"
onnx_optimized_path = "./onnx-exported/model.onnx"
onnx_quantized_path = "./onnx-exported/model_quantized.onnx"
MAX_LENGTH = 128
BATCH_SIZE = 16

# ------- Exporting the model to ONNX format -------
def export_onnx_model(model_path, output_dir, opset=17):
    print("Saving tokenizer and config files...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer.save_pretrained(output_dir)

    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    config.save_pretrained(output_dir)

    print("Exporting model to ONNX format...")
    main_export(
        model_name_or_path=model_path,
        output=output_dir,
        opset=opset,
        task="token-classification",
        do_validation=True
    )
    print(f"Model exported to {output_dir}")

def quantize_onnx_model(onnx_optimized_path, onnx_quantized_path):
    print("Quantizing ONNX model...")
    quantize_dynamic(
        model_input=onnx_optimized_path,
        model_output=onnx_quantized_path,
        weight_type=QuantType.QInt8
    )
    print(f"Quantized model saved to {onnx_quantized_path}")

def batchify(dataset, batch_size):
    """Yield successive batches from the dataset."""
    it = iter(dataset)
    while True:
        batch = list(islice(it, batch_size))
        if not batch:
            break
        yield batch

def evaluate_onnx_model():
    print("🧪 Evaluating quantized model on test dataset...")

    test_data_path = "./eventkg_bio_sentences.json"
    label_list = ["O", "B-EVENT", "I-EVENT", "B-TIME", "I-TIME", "B-LOCATION", "I-LOCATION"]
    id_to_label = {i: label for i, label in enumerate(label_list)}
    batch_size = 16
    max_length = 128

    # --- LOAD MODEL + TOKENIZER ---
    try:
        session = ort.InferenceSession(onnx_quantized_path)
    except Exception as e:
        print(f"❌ Failed to load ONNX model: {e}")
        return
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

    # --- LOAD + FILTER TEST DATA ---
    try:
        with open(test_data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except Exception as e:
        print(f"❌ Failed to load test data: {e}")
        return

    original_len = len(raw_data)
    test_data = [ex for ex in raw_data if "tokens" in ex and "tags" in ex]
    print(f"Filtered {original_len - len(test_data)} examples missing 'tokens' or 'tags'.")

    # --- BATCHING ---
    def batchify(data, batch_size):
        for i in range(0, len(data), batch_size):
            yield data[i:i + batch_size]

    all_preds = []
    all_refs = []

    for batch in batchify(test_data, batch_size):
        tokens_batch = [ex["tokens"] for ex in batch]
        labels_batch = [ex["tags"] for ex in batch]

        # Tokenize
        encodings = tokenizer(
            tokens_batch,
            is_split_into_words=True,
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=max_length
        )

        # Convert to ONNX format
        ort_inputs = {k: v.astype(np.int64) for k, v in encodings.items()}
        outputs = session.run(None, ort_inputs)
        batch_preds = outputs[0].argmax(-1)

        # Iterate over batch
        for i in range(len(tokens_batch)):
            word_ids = encodings.word_ids(batch_index=i)
            preds = batch_preds[i]

            pred_tags = []
            ref_tags = []

            for j, word_id in enumerate(word_ids):
                if word_id is None:
                    continue
                # First subtoken only
                if j == 0 or word_id != word_ids[j - 1]:
                    try:
                        true_tag = labels_batch[i][word_id]  # already a string like "B-TIME"
                        pred_tag = id_to_label[preds[j]]
                        ref_tags.append(true_tag)
                        pred_tags.append(pred_tag)
                    except IndexError:
                        continue  # skip if token mismatch

            all_preds.append(pred_tags)
            all_refs.append(ref_tags)

    # --- METRICS ---
    print("🔍 Classification Report:")
    print(classification_report(all_refs, all_preds))
    print(f"🎯 F1 Score: {f1_score(all_refs, all_preds):.4f}")

def print_model_size(path, label="Model"):
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"{label} size: {size_mb:.2f} MB ({path})")
    else:
        print(f"{label} not found at: {path}")

if __name__ == "__main__":
    export_onnx_model(model_path, "./onnx-exported", opset=17)
    print_model_size(onnx_optimized_path, "Optimized ONNX Model")

    quantize_onnx_model(onnx_optimized_path, onnx_quantized_path)
    print_model_size(onnx_quantized_path, "Quantized ONNX Model")

    evaluate_onnx_model()

    print("Optimized ONNX model created successfully.")