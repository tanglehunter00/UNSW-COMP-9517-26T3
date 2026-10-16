"""Build and execute a Lab2-style Lab3.ipynb."""
from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "Lab3.ipynb"
SUBMIT = ROOT / "Lab3_submit.ipynb"


def build() -> nbformat.NotebookNode:
    return new_notebook(
        cells=[
            new_code_cell(
                "%matplotlib inline\n"
                "from pathlib import Path\n"
                "import cv2\n"
                "import numpy as np\n"
                "import matplotlib.pyplot as plt\n"
                "from sklearn.neighbors import KNeighborsClassifier\n"
                "from sklearn.tree import DecisionTreeClassifier\n"
                "from sklearn.linear_model import SGDClassifier\n"
                "from sklearn.metrics import (\n"
                "    accuracy_score,\n"
                "    classification_report,\n"
                "    confusion_matrix,\n"
                "    ConfusionMatrixDisplay,\n"
                ")\n"
                "\n"
                "SEED = 42\n"
                'DATA_PATH = Path("BloodMNIST.npz")\n'
                "\n"
                "def show(image):\n"
                "    plt.imshow(image)\n"
                "    plt.show()\n"
                "\n"
                "def show_cm(cm):\n"
                "    ConfusionMatrixDisplay(cm).plot(colorbar=False)\n"
                "    plt.show()\n"
                "\n"
                "with np.load(DATA_PATH) as archive:\n"
                "    data = {key: archive[key] for key in archive.files}\n"
                "\n"
                'for split in ("train", "val", "test"):\n'
                '    images = data[f"{split}_images"]\n'
                '    labels = data[f"{split}_labels"].ravel()\n'
                '    print(split, images.shape, labels.shape)\n'
                "    print(\"counts:\", np.bincount(labels, minlength=8))\n"
                "\n"
                "# some images from each class\n"
                "train_images = data[\"train_images\"]\n"
                "train_labels = data[\"train_labels\"].ravel()\n"
                "for label in range(8):\n"
                "    idxs = np.flatnonzero(train_labels == label)[:3]\n"
                "    for i in idxs:\n"
                "        show(train_images[i])"
            ),
            new_code_cell(
                "def balanced_indices(labels, size, seed):\n"
                "    labels = np.asarray(labels).ravel()\n"
                "    quota = size // 8\n"
                "    rng = np.random.default_rng(seed)\n"
                "    idxs = []\n"
                "    for c in range(8):\n"
                "        group = np.flatnonzero(labels == c)\n"
                "        idxs.append(rng.choice(group, quota, replace=False))\n"
                "    idxs = np.concatenate(idxs)\n"
                "    rng.shuffle(idxs)\n"
                "    return idxs\n"
                "\n"
                "def flatten(images):\n"
                "    return images.reshape(len(images), -1)\n"
                "\n"
                "train_y = data[\"train_labels\"].ravel()\n"
                "test_y = data[\"test_labels\"].ravel()\n"
                "train_idx = balanced_indices(train_y, 5000, SEED)\n"
                "test_idx = balanced_indices(test_y, 1000, SEED + 1)\n"
                "\n"
                "full_train_x = flatten(data[\"train_images\"])\n"
                "full_test_x = flatten(data[\"test_images\"])\n"
                "\n"
                "experiments = {\n"
                '    "subset": (\n'
                "        full_train_x[train_idx], train_y[train_idx],\n"
                "        full_test_x[test_idx], test_y[test_idx],\n"
                "    ),\n"
                '    "full": (full_train_x, train_y, full_test_x, test_y),\n'
                "}\n"
                "\n"
                'print("subset train counts:", np.bincount(train_y[train_idx], minlength=8))\n'
                'print("subset test counts:", np.bincount(test_y[test_idx], minlength=8))\n'
                "for name, (xtr, ytr, xte, yte) in experiments.items():\n"
                '    print(name, "train", xtr.shape, "test", xte.shape)'
            ),
            new_code_cell(
                "def make_classifiers():\n"
                "    return {\n"
                '        "KNN": KNeighborsClassifier(n_neighbors=3),\n'
                '        "DT": DecisionTreeClassifier(),\n'
                '        "SGD": SGDClassifier(max_iter=250),\n'
                "    }\n"
                "\n"
                "for experiment, (xtr, ytr, xte, yte) in experiments.items():\n"
                "    for name, model in make_classifiers().items():\n"
                "        np.random.seed(SEED)\n"
                "        model.fit(xtr, ytr)\n"
                "        pred = model.predict(xte)\n"
                '        print(experiment, name)\n'
                '        print("accuracy:", accuracy_score(yte, pred))\n'
                "        print(classification_report(yte, pred, digits=4))\n"
                "        show_cm(confusion_matrix(yte, pred))"
            ),
            new_markdown_cell(
                "full set is a bit better than the balanced subset for all three\n"
                "\n"
                "knn did best here, then sgd, decision tree was weakest\n"
                "\n"
                "more training data helps a bit, but the full set is also unbalanced "
                "so the scores are not directly comparable to the subset"
            ),
        ],
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
    )


def main() -> None:
    nb = build()
    client = NotebookClient(
        nb,
        timeout=1800,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    )
    client.execute()
    tmp = OUT.with_suffix(".ipynb.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    tmp.replace(OUT)
    SUBMIT.write_bytes(OUT.read_bytes())
    print(f"Wrote {OUT.name} and {SUBMIT.name} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
