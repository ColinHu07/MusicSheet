from pathlib import Path
import argparse
import sys
import json
import hashlib
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from musicsheet.acoustic import acoustic_features, MelodyNetwork, predict
from musicsheet.engine import feature_pair

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("model-dir", "features-dir", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-export", action="store_true")
    args = parser.parse_args()
    mix, vocal = feature_pair(args.features_dir)
    weights = torch.load(args.model_dir / "checkpoint.pt", map_location="cpu", weights_only=True)
    model = MelodyNetwork(hidden=weights["input.weight"].shape[0]).to(args.device).eval()
    model.load_state_dict(weights)
    with np.load(args.model_dir / "normalization.npz") as norm:
        pitch, bounds = predict(model, acoustic_features(mix, vocal), norm["mean"], norm["scale"], args.device)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output_dir / "probabilities.npz", pitch=pitch, boundaries=bounds, times_s=mix["times_s"])
