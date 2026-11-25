"""spaCy NER Training and Fine-Tuning Pipeline for Technical Skills.

Converts annotated spans from dataset/annotations.json into spaCy DocBin
binary format, trains/fine-tunes the named entity recognizer for the 'SKILL' label,
evaluates precision/recall/F1 metrics, and serializes artifacts to models/skill_ner/.
"""

import argparse
import json
import os
import random
import sys
from typing import Any, Dict, List, Tuple


def load_annotations(file_path: str) -> List[Dict[str, Any]]:
    """Loads annotated training examples from JSON file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Annotations file not found at {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} annotated records from {file_path}")
    return data


def create_docbins(
    training_data: List[Dict[str, Any]],
    output_dir: str = "models",
    seed: int = 42,
) -> Tuple[str, str, str]:
    """Creates reproducible 70:15:15 train/dev/test DocBin splits."""
    random.seed(seed)
    shuffled = list(training_data)
    random.shuffle(shuffled)
    train_end = int(len(shuffled) * 0.70)
    dev_end = int(len(shuffled) * 0.85)
    splits = {
        "train": shuffled[:train_end],
        "dev": shuffled[train_end:dev_end],
        "test": shuffled[dev_end:],
    }
    os.makedirs(output_dir, exist_ok=True)
    paths = {name: os.path.join(output_dir, f"{name}.spacy") for name in splits}
    try:
        import spacy
        from spacy.tokens import DocBin
        nlp = spacy.blank("en")
        for name, records in splits.items():
            doc_bin = DocBin()
            for item in records:
                doc = nlp.make_doc(item["text"])
                spans = []
                for start, end, label in item.get("entities", []):
                    span = doc.char_span(start, end, label=label, alignment_mode="strict")
                    if span is not None:
                        spans.append(span)
                doc.ents = spans
                doc_bin.add(doc)
            doc_bin.to_disk(paths[name])
            print(f"Saved {len(records)} {name} docs to {paths[name]}")
    except ImportError:
        for name, records in splits.items():
            json_path = os.path.join(output_dir, f"{name}_split.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2)
            paths[name] = json_path
            print(f"Saved {len(records)} {name} records to {json_path}")
    return paths["train"], paths["dev"], paths["test"]


def train_model(
    annotations_path: str = "dataset/annotations.json",
    model_output_dir: str = "models/skill_ner",
    epochs: int = 30,
    dropout: float = 0.2,
) -> None:
    """Trains or fine-tunes spaCy NER model for SKILL detection."""
    data = load_annotations(annotations_path)
    # Create one deterministic split shared by training, evaluation, and metadata.
    random.seed(42)
    shuffled = list(data)
    random.shuffle(shuffled)
    train_end = int(len(shuffled) * 0.70)
    dev_end = int(len(shuffled) * 0.85)
    train_records = shuffled[:train_end]
    dev_records = shuffled[train_end:dev_end]
    test_records = shuffled[dev_end:]
    os.makedirs(model_output_dir, exist_ok=True)

    # 1. Prepare DocBins
    create_docbins(data, output_dir=os.path.dirname(model_output_dir))

    # 2. Check spaCy availability
    model_trained = False
    try:
        import spacy
        from spacy.training import Example
        from spacy.scorer import Scorer
        from spacy.util import minibatch, compounding

        print("\n--- Initializing spaCy NER Pipeline ---")
        try:
            nlp = spacy.load("en_core_web_sm")
            print("Loaded base model: en_core_web_sm")
        except Exception:
            nlp = spacy.blank("en")
            print("Initialized blank English pipeline")

        # Get or add NER pipeline component and remember whether it existed.
        had_ner = "ner" in nlp.pipe_names
        if not had_ner:
            ner = nlp.add_pipe("ner", last=True)
        else:
            ner = nlp.get_pipe("ner")

        ner.add_label("SKILL")

        # Prepare examples using ONLY the training partition.
        # Keep validation examples completely held out from optimizer updates.
        def to_examples(records):
            converted = []
            for item in records:
                doc = nlp.make_doc(item["text"])
                entities = []
                for start, end, label in item.get("entities", []):
                    entities.append((start, end, label))
                converted.append(Example.from_dict(doc, {"entities": entities}))
            return converted

        train_examples = to_examples(train_records)
        dev_examples = to_examples(dev_records)
        test_examples = to_examples(test_records)

        other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
        with nlp.disable_pipes(*other_pipes):
            optimizer = nlp.resume_training() if had_ner else nlp.initialize(get_examples=lambda: train_examples)
            print(f"Training records: {len(train_examples)} | Validation records: {len(dev_examples)} | Test records: {len(test_examples)}")
            print(f"Beginning training across {epochs} epochs...")
            for epoch in range(1, epochs + 1):
                random.shuffle(train_examples)
                losses = {}
                batches = minibatch(train_examples, size=compounding(4.0, 32.0, 1.001))
                for batch in batches:
                    nlp.update(batch, drop=dropout, losses=losses, sgd=optimizer)
                if epoch % 5 == 0 or epoch == epochs:
                    print(f"Epoch {epoch:02d}/{epochs:02d} | NER Loss: {losses.get('ner', 0.0):.4f}")

            # Evaluate predictions on held-out validation and test records.
            # Score freshly predicted docs against their gold reference docs;
            # scoring the original training Examples would score empty predictions.
            for split_name, split_examples in [("Validation", dev_examples), ("Test", test_examples)]:
                predicted_examples = []
                for example in split_examples:
                    predicted_doc = nlp(example.reference.text)
                    predicted_examples.append(Example(predicted_doc, example.reference))
                scores = Scorer().score(predicted_examples)
                print(f"{split_name} NER scores: P={scores.get('ents_p', 0):.4f}, R={scores.get('ents_r', 0):.4f}, F1={scores.get('ents_f', 0):.4f}")

        # Save model
        nlp.to_disk(model_output_dir)
        model_trained = True
        print(f"\nTrained spaCy NER model successfully saved to: {model_output_dir}")

    except ImportError:
        print("\nspaCy not installed in current environment.")
        print("Generating model configuration manifest and rule-based artifacts...")

    # Write model manifest and metadata
    manifest = {
        "model_name": "skill_ner",
        "version": "1.1.0",
        "labels": ["SKILL"],
        "architecture": "spaCy NER with taxonomy PhraseMatcher fallback",
        "model_status": "trained_and_serialized" if model_trained else "training_dependency_missing",
        "model_artifacts_present": model_trained,
        "training_examples": len(train_records),
        "validation_examples": len(dev_records),
        "test_examples": len(test_records),
        "total_annotated_records": len(data),
        "epochs": epochs,
        "base_model": "en_core_web_sm / blank:en",
    }
    with open(os.path.join(model_output_dir, "model_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Save skill extraction patterns
    taxonomy_file = "dataset/skills/skills_taxonomy.csv"
    if os.path.exists(taxonomy_file):
        import csv
        patterns = []
        with open(taxonomy_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = row.get("preferred_label", "")
                if label:
                    patterns.append({"label": "SKILL", "pattern": label})
                for alt in row.get("alt_labels", "").split("|"):
                    if alt.strip():
                        patterns.append({"label": "SKILL", "pattern": alt.strip()})
        with open(os.path.join(model_output_dir, "skill_patterns.json"), "w", encoding="utf-8") as f:
            json.dump(patterns, f, indent=2)

    print(f"Saved model artifacts to {model_output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train custom spaCy NER for skill extraction.")
    parser.add_argument("--annotations", default="dataset/annotations.json", help="Path to annotations JSON")
    parser.add_argument("--output", default="models/skill_ner", help="Path to save model")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    args = parser.parse_args()

    train_model(
        annotations_path=args.annotations,
        model_output_dir=args.output,
        epochs=args.epochs,
        dropout=args.dropout,
    )


if __name__ == "__main__":
    main()
