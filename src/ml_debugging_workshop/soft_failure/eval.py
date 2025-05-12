import torch
from torch.utils.data import DataLoader
from train import CSVDataset, LitClassifier

CHECKPOINT_PATH = "models/trained-model.ckpt"
DATA_SPLITS = {
    "train": "splits/train.csv",
    "val": "splits/val.csv",
    "test": "splits/test.csv",
}
BATCH_SIZE = 64


def evaluate_split(name: str, path: str, model: LitClassifier) -> None:
    ds = CSVDataset(path)
    loader = DataLoader(ds, batch_size=BATCH_SIZE)
    correct = total = 0

    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            preds = (torch.sigmoid(logits) > 0.5).long()
            correct += (preds == y).sum().item()
            total += y.size(0)

    acc = correct / total
    print(f"{name.capitalize()} Accuracy: {acc:.3f}")


def main() -> None:
    sample_ds = CSVDataset(DATA_SPLITS["train"])
    input_dim = sample_ds.X.shape[1]
    model = LitClassifier.load_from_checkpoint(CHECKPOINT_PATH, input_dim=input_dim)
    model.eval()

    for name, path in DATA_SPLITS.items():
        evaluate_split(name, path, model)


if __name__ == "__main__":
    main()
