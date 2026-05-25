import argparse
import json
import os
import re
import torch
from tqdm import tqdm
from pydantic import BaseModel
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
from transformers import (
    LlavaForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
from dataset import ArtifactDataset
from prompts import SYSTEM_PROMPT, generate_evaluation_prompt


class EvaluationOutput(BaseModel):
    skill: str
    dimension: str
    score: int
    max: int


def parse_args():
    parser = argparse.ArgumentParser(description="VLM Evaluation Pipeline")
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to dataset JSON"
    )
    parser.add_argument("--model_name", type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--quantize", action="store_true", default=True)
    parser.add_argument("--no_quantize", action="store_false", dest="quantize")
    parser.add_argument("--output_path", type=str, default="results.json")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    return parser.parse_args()


def load_model(model_name, quantize=True):
    if quantize:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        quantization_config = None

    model = LlavaForConditionalGeneration.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor


def extract_score(text):
    try:
        # Parse the JSON directly instead of using Regex
        parsed = json.loads(text)
        score = parsed.get("score")
        if isinstance(score, int) and 1 <= score <= 5:
            return score
    except json.JSONDecodeError:
        pass
    return None


def compute_metrics(predictions, ground_truths):
    total = len(ground_truths)
    if total == 0:
        return {}

    exact = sum(1 for p, g in zip(predictions, ground_truths) if p == g)
    within_1 = sum(1 for p, g in zip(predictions, ground_truths) if abs(p - g) <= 1)
    mae = sum(abs(p - g) for p, g in zip(predictions, ground_truths)) / total
    parsed = sum(1 for p in predictions if p is not None)

    return {
        "total_samples": total,
        "exact_accuracy": round(exact / total * 100, 2),
        "within_1_accuracy": round(within_1 / total * 100, 2),
        "mean_absolute_error": round(mae, 4),
        "parse_rate": round(parsed / total * 100, 2),
    }


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        print("Warning: CUDA not available. Inference will be slow on CPU.")

    print(f"Loading dataset from {args.data_path}...")
    dataset = ArtifactDataset(args.data_path)
    if len(dataset) == 0:
        print("Dataset is empty. Exiting.")
        return

    print(f"Loading model {args.model_name} (quantize={args.quantize})...")
    model, processor = load_model(args.model_name, quantize=args.quantize)

    results = []
    preds = []
    truths = []

    for i in tqdm(range(len(dataset)), desc="Evaluating"):
        sample = dataset[i]
        meta = sample["metadata"]
        image = sample["image"]

        if image is None:
            continue

        prompt_text = generate_evaluation_prompt(
            student_id=meta.get("student_id", "unknown"),
            artifact_type=meta.get("artifact_type", "unknown"),
            rubric=meta.get("rubric", {}),
        )

        inputs = processor(text=prompt_text, images=image, return_tensors="pt").to(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        try:
            schema = EvaluationOutput.model_json_schema()
        except AttributeError:
            schema = EvaluationOutput.schema()
            
        parser = JsonSchemaParser(schema)
        prefix_function = build_transformers_prefix_allowed_tokens_fn(processor.tokenizer, parser)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                prefix_allowed_tokens_fn=prefix_function,
            )

        decoded = processor.decode(output_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        response = decoded.strip()

        predicted_score = extract_score(response)
        ground_truth = meta.get("ground_truth_score")

        results.append(
            {
                "student_id": meta.get("student_id", "unknown"),
                "predicted_score": predicted_score,
                "ground_truth_score": ground_truth,
                "raw_response": response,
                "artifact_type": meta.get("artifact_type", "unknown"),
            }
        )

        if predicted_score is not None and ground_truth is not None:
            preds.append(predicted_score)
            truths.append(ground_truth)

    metrics = compute_metrics(preds, truths)

    output = {
        "config": {
            "model_name": args.model_name,
            "quantize": args.quantize,
            "dataset": args.data_path,
        },
        "metrics": metrics,
        "results": results,
    }

    with open(args.output_path, "w") as f:
        json.dump(output, f, indent=2)

    print("\n" + "=" * 50)
    print("EVALUATION METRICS")
    print("=" * 50)
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print("=" * 50)
    print(f"Results saved to {args.output_path}")


if __name__ == "__main__":
    main()
