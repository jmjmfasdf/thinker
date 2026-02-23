#!/usr/bin/env python
"""Train a simple MLP to predict a target feature from an input feature.

This script mirrors the pairing logic in embedding_analysis.ipynb:
use the planning step just before each real action (status == 0) as input,
and use the real-step target feature as label.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def _parse_specs(spec: str | Sequence[str]) -> List[str]:
    if isinstance(spec, (list, tuple)):
        parts: List[str] = []
        for item in spec:
            for sub in str(item).split(","):
                sub = sub.strip()
                if sub:
                    parts.append(sub)
        return parts
    parts = [s.strip() for s in str(spec).split(",")]
    return [p for p in parts if p]


def _flatten(arr: np.ndarray, max_len: int | None = None) -> np.ndarray:
    arr = np.asarray(arr)
    if max_len is not None:
        arr = arr[:max_len]
    if arr.ndim == 1:
        arr = arr.reshape(arr.shape[0], 1)
    else:
        arr = arr.reshape(arr.shape[0], -1)
    return arr.astype(np.float32)


def _concat_dict_values(d: dict) -> np.ndarray:
    parts = [np.asarray(v) for v in d.values()]
    if not parts:
        raise ValueError("dict feature에 값이 없습니다.")
    common_len = min(p.shape[0] for p in parts)
    flat_parts = [_flatten(p, common_len) for p in parts]
    return np.concatenate(flat_parts, axis=1)


def _feature_from_key(data: dict, key: str) -> np.ndarray:
    # tree_reps / tree_reps_vector는 동일 처리
    if key in ("tree_reps_vector", "tree_reps"):
        if "tree_reps" not in data or not isinstance(data["tree_reps"], dict):
            raise KeyError("'tree_reps' dict가 data에 없습니다.")
        return _concat_dict_values(data["tree_reps"])

    if key not in data:
        raise KeyError(f"'{key}'가 data에 없습니다.")

    value = data[key]
    if isinstance(value, dict):
        return _concat_dict_values(value)

    return _flatten(value)


def vectorize_by_specs(data: dict, specs: Sequence[str]) -> Tuple[np.ndarray, int]:
    entries = []
    for key in specs:
        try:
            entries.append(_feature_from_key(data, key))
        except KeyError:
            continue

    if not entries:
        raise ValueError("선택된 feature가 없습니다.")

    common_len = min(e.shape[0] for e in entries)
    entries = [e[:common_len] for e in entries]
    return np.concatenate(entries, axis=1), common_len


def pair_planning_before_real(status: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    status = np.asarray(status)
    real_idx = np.where(status == 0)[0]
    driver_idx = real_idx - 1
    mask = driver_idx >= 0
    return real_idx[mask], driver_idx[mask]


def _decode_label_array(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return np.argmax(arr, axis=1)
    if arr.ndim == 1 and np.issubdtype(arr.dtype, np.integer):
        return arr
    return arr.astype(object)


def _extract_labels(
    data: dict, label_specs: Sequence[str], max_len: int, real_idx: np.ndarray
) -> np.ndarray:
    parts = []
    for key in label_specs:
        if key not in data:
            raise KeyError(f"label key '{key}'가 data에 없습니다.")
        labels_all = _decode_label_array(data[key])[:max_len]
        parts.append(labels_all[real_idx])

    if len(parts) == 1:
        return parts[0]
    return np.array(
        [" | ".join(str(v) for v in x) for x in zip(*parts)], dtype=object
    )


def prepare_feature_label_pair(
    data: dict, input_specs: Sequence[str], target_specs: Sequence[str]
) -> Tuple[np.ndarray, np.ndarray]:
    features, max_len = vectorize_by_specs(data, input_specs)

    if "status" not in data:
        raise KeyError("'status'가 data에 없습니다. pairing을 위해 필요합니다.")
    status = np.asarray(data["status"])[:max_len]
    real_idx, driver_idx = pair_planning_before_real(status)
    if real_idx.size == 0:
        raise ValueError("pairing 결과가 비어 있습니다. status 배열을 확인하세요.")

    feature_subset = features[driver_idx]
    labels = _extract_labels(data, target_specs, max_len, real_idx)
    return feature_subset, labels


def _load_single_npy(path: Path) -> dict:
    obj = np.load(path, allow_pickle=True)

    # np.save(dict) 케이스
    if isinstance(obj, np.ndarray) and obj.dtype == object and obj.shape == ():
        data = obj.item()
        if isinstance(data, dict):
            return data

    # npz 케이스
    if hasattr(obj, "files"):
        return {k: obj[k] for k in obj.files}

    raise ValueError(f"{path}: dict 형태의 npy/npz를 해석할 수 없습니다.")


def _is_classification_target(y: np.ndarray) -> bool:
    if y.dtype == object or np.issubdtype(y.dtype, np.str_):
        return True
    if np.issubdtype(y.dtype, np.bool_) or np.issubdtype(y.dtype, np.integer):
        return True
    if np.issubdtype(y.dtype, np.floating):
        y_int = np.rint(y)
        if np.allclose(y, y_int) and len(np.unique(y_int)) <= 50:
            return True
    return False


def _encode_class_labels(y: np.ndarray) -> Tuple[np.ndarray, dict]:
    y = np.asarray(y)
    if np.issubdtype(y.dtype, np.floating):
        y = np.rint(y).astype(int)
    if np.issubdtype(y.dtype, np.integer):
        classes = np.unique(y)
        mapping = {int(c): int(c) for c in classes}
        return y.astype(int), mapping

    # object/string labels
    classes = np.unique(y.astype(str))
    mapping = {label: idx for idx, label in enumerate(classes)}
    encoded = np.array([mapping[str(v)] for v in y], dtype=int)
    return encoded, mapping


def _encode_with_mapping(y: np.ndarray, mapping: dict) -> np.ndarray:
    y = np.asarray(y)
    if np.issubdtype(y.dtype, np.floating):
        y = np.rint(y).astype(int)
    if np.issubdtype(y.dtype, np.integer):
        return np.array([mapping.get(int(v), -1) for v in y], dtype=int)
    return np.array([mapping.get(str(v), -1) for v in y], dtype=int)


def _to_int_labels(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if np.issubdtype(y.dtype, np.floating):
        y = np.rint(y).astype(int)
    if np.issubdtype(y.dtype, np.integer):
        return y.astype(int)
    raise ValueError("숫자형 label이 아닙니다.")


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: Sequence[int], out_dim: int) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _standardize_train_test(
    x_train: np.ndarray, x_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(x_train, axis=0, keepdims=True)
    std = np.nanstd(x_train, axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std
    return x_train, x_test


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train an MLP to predict target feature from input feature using paired planning/real steps."
    )
    parser.add_argument("directory", help="Root directory containing .npy files")
    parser.add_argument(
        "input_feature",
        nargs="?",
        help="Input feature key (comma-separated for multiple). Prefer --input-feature.",
    )
    parser.add_argument(
        "target_feature",
        nargs="?",
        help="Target feature key (comma-separated for multiple). Prefer --target-feature.",
    )
    parser.add_argument(
        "--input-feature",
        "--input_feature",
        dest="input_feature_flag",
        nargs="+",
        default=None,
        help="Input feature keys (space or comma separated).",
    )
    parser.add_argument(
        "--target-feature",
        "--target_feature",
        dest="target_feature_flag",
        nargs="+",
        default=None,
        help="Target feature keys (space or comma separated).",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden", type=str, default="256,128")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="Force number of classes for classification (e.g., 6).",
    )
    parser.add_argument(
        "--non-recursive",
        action="store_true",
        help="Only read .npy files in the top directory (default: recursive).",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    root = Path(args.directory)
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")

    pattern = "*.npy" if args.non_recursive else "**/*.npy"
    npy_files = sorted(root.glob(pattern))
    if not npy_files:
        raise FileNotFoundError(f"No .npy files found under {root}")

    if args.input_feature_flag is not None and args.input_feature is not None:
        print("[warn] both positional input_feature and --input-feature provided; using --input-feature")
    if args.target_feature_flag is not None and args.target_feature is not None:
        print("[warn] both positional target_feature and --target-feature provided; using --target-feature")

    if args.input_feature_flag is not None:
        input_specs = _parse_specs(args.input_feature_flag)
    elif args.input_feature is not None:
        input_specs = _parse_specs(args.input_feature)
    else:
        input_specs = []

    if args.target_feature_flag is not None:
        target_specs = _parse_specs(args.target_feature_flag)
    elif args.target_feature is not None:
        target_specs = _parse_specs(args.target_feature)
    else:
        target_specs = []

    # Simple validation
    if not input_specs:
        raise ValueError("input_feature가 필요합니다. --input-feature 또는 positional로 지정하세요.")
    if not target_specs:
        raise ValueError("target_feature가 필요합니다. --target-feature 또는 positional로 지정하세요.")

    per_file_data: List[Tuple[Path, np.ndarray, np.ndarray]] = []
    for path in npy_files:
        try:
            data = _load_single_npy(path)
            x, y = prepare_feature_label_pair(data, input_specs, target_specs)
        except Exception as exc:
            print(f"[skip] {path}: {exc}")
            continue
        if x.shape[0] == 0:
            print(f"[skip] {path}: no samples after pairing")
            continue
        per_file_data.append((path, x, y))
        print(f"[loaded] {path} samples={x.shape[0]}")

    if len(per_file_data) < 2:
        raise RuntimeError("유효한 npy 파일이 2개 이상 필요합니다. (train/test 분리)")

    test_idx = rng.randrange(len(per_file_data))
    test_path, x_test, y_test_raw = per_file_data[test_idx]
    train_parts = [item for i, item in enumerate(per_file_data) if i != test_idx]

    train_files = [p for p, _, _ in train_parts]
    print("\n[split]")
    print(f"test file: {test_path}")
    print("train files:")
    for p in train_files:
        print(f"  - {p}")

    x_train = np.concatenate([x for _, x, _ in train_parts], axis=0)
    y_train_raw = np.concatenate([y for _, _, y in train_parts], axis=0)

    # Handle NaN/inf
    x_train = np.nan_to_num(x_train, nan=0.0, posinf=0.0, neginf=0.0)
    x_test = np.nan_to_num(x_test, nan=0.0, posinf=0.0, neginf=0.0)

    x_train, x_test = _standardize_train_test(x_train, x_test)

    if _is_classification_target(y_train_raw):
        if args.num_classes is not None:
            try:
                y_train = _to_int_labels(y_train_raw)
                y_test = _to_int_labels(y_test_raw)
                label_map = {i: i for i in range(args.num_classes)}
                num_classes = args.num_classes
            except ValueError:
                y_train, label_map = _encode_class_labels(y_train_raw)
                y_test = _encode_with_mapping(y_test_raw, label_map)
                if len(label_map) > args.num_classes:
                    raise ValueError(
                        f"num_classes({args.num_classes}) < label count({len(label_map)})"
                    )
                num_classes = args.num_classes
        else:
            y_train, label_map = _encode_class_labels(y_train_raw)
            y_test = _encode_with_mapping(y_test_raw, label_map)
            num_classes = int(np.max(y_train)) + 1
        task = "classification"
    else:
        y_train = np.asarray(y_train_raw, dtype=np.float32)
        y_test = np.asarray(y_test_raw, dtype=np.float32)
        num_classes = None
        task = "regression"

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    hidden_dims = [int(x) for x in args.hidden.split(",") if x.strip()]
    if task == "classification":
        model = MLP(x_train.shape[1], hidden_dims, num_classes)
        criterion = nn.CrossEntropyLoss()
    else:
        out_dim = 1 if y_train.ndim == 1 else y_train.shape[1]
        model = MLP(x_train.shape[1], hidden_dims, out_dim)
        criterion = nn.MSELoss()

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if task == "classification":
        y_train_tensor = torch.from_numpy(y_train).long()
    else:
        y_train_tensor = torch.from_numpy(y_train).float()

    train_dataset = TensorDataset(torch.from_numpy(x_train).float(), y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    print(f"\n[train] task={task} device={device} samples={len(train_dataset)}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            if task == "classification":
                loss = criterion(preds, yb)
            else:
                loss = criterion(preds.squeeze(-1), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)

        avg_loss = total_loss / len(train_dataset)
        model.eval()
        with torch.no_grad():
            x_train_t = torch.from_numpy(x_train).float().to(device)
            if task == "classification":
                logits = model(x_train_t)
                pred_labels = torch.argmax(logits, dim=1)
                train_acc = (pred_labels.cpu() == y_train_tensor).float().mean().item()
                print(f"epoch {epoch:03d} loss={avg_loss:.4f} train_acc={train_acc:.4f}")
            else:
                preds = model(x_train_t).squeeze(-1).cpu().numpy()
                y_true = y_train_tensor.numpy()
                ss_res = float(np.sum((preds - y_true) ** 2))
                ss_tot = float(np.sum((y_true - y_true.mean()) ** 2)) + 1e-8
                r2 = 1.0 - ss_res / ss_tot
                print(f"epoch {epoch:03d} loss={avg_loss:.4f} train_r2={r2:.4f}")

    # Test evaluation
    model.eval()
    with torch.no_grad():
        x_test_t = torch.from_numpy(x_test).float().to(device)
        if task == "classification":
            y_test_tensor = torch.from_numpy(y_test).long()
            logits = model(x_test_t)
            pred_labels = torch.argmax(logits, dim=1).cpu()
            valid_mask = y_test_tensor >= 0
            if valid_mask.any():
                test_acc = (pred_labels[valid_mask] == y_test_tensor[valid_mask]).float().mean().item()
                print(f"\n[test] accuracy={test_acc:.4f} (file={test_path})")
            else:
                print(f"\n[test] accuracy=N/A (no known labels) (file={test_path})")
        else:
            y_test_tensor = torch.from_numpy(y_test).float()
            preds = model(x_test_t).squeeze(-1).cpu().numpy()
            y_true = y_test_tensor.numpy()
            ss_res = float(np.sum((preds - y_true) ** 2))
            ss_tot = float(np.sum((y_true - y_true.mean()) ** 2)) + 1e-8
            r2 = 1.0 - ss_res / ss_tot
            print(f"\n[test] r2={r2:.4f} (file={test_path})")


if __name__ == "__main__":
    main()
