"""Build and execute Lab1.ipynb from this source of truth."""
from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "Lab1.ipynb"
SUBMIT = ROOT / "Lab1_submit.ipynb"


def build() -> nbformat.NotebookNode:
    return new_notebook(
        cells=[
            new_markdown_cell(
                "# COMP9517 26T3 — Lab 1\n\n"
                "Image averaging, Prewitt derivatives, and unsharp masking with OpenCV.\n\n"
                "Change `IMAGE_DIR` below if needed. "
                "`load_gray(filename, image_dir)` takes the image directory as an argument."
            ),
            new_code_cell(
                "%matplotlib inline\n"
                "from pathlib import Path\n"
                "import cv2\n"
                "import numpy as np\n"
                "import matplotlib.pyplot as plt\n"
                "\n"
                'IMAGE_DIR = Path("COMP9517_26T3_Lab1_Images")\n'
                "\n"
                "def load_gray(filename, image_dir):\n"
                "    path = Path(image_dir) / filename\n"
                "    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)\n"
                "    if image is None:\n"
                '        raise FileNotFoundError(f"Cannot read image: {path.resolve()}")\n'
                "    return image.astype(np.float64)"
            ),
            new_markdown_cell(
                "## Task 1 — Average ten noisy images\n\n"
                r"Pixelwise average: $\bar I=\frac{1}{N}\sum_{k=1}^N I_k$, computed in floating point."
            ),
            new_code_cell(
                'opera = np.stack([load_gray(f"Opera{i:02d}.jpg", IMAGE_DIR)\n'
                "                  for i in range(1, 11)], axis=0)\n"
                "average = np.mean(opera, axis=0)\n"
                "\n"
                "# Top-left sky ROI for noise measurement.\n"
                "x0, x1, y0, y1 = 20, 120, 20, 100\n"
                "std_single = np.std(opera[0, y0:y1, x0:x1])\n"
                "std_average = np.std(average[y0:y1, x0:x1])\n"
                "ratio = std_single / std_average\n"
                "\n"
                'plt.imshow(opera[0], cmap="gray", vmin=0, vmax=255)\n'
                'plt.axis("off")\n'
                "plt.show()\n"
                "\n"
                'plt.imshow(average, cmap="gray", vmin=0, vmax=255)\n'
                'plt.axis("off")\n'
                "plt.show()\n"
                "\n"
                'print(f"sd in Opera01: {std_single:.4f}")\n'
                'print(f"sd in average: {std_average:.4f}")\n'
                'print(f"ratio: {ratio:.4f}")\n'
                'print(f"theoretical: {np.sqrt(10):.4f}")'
            ),
            new_markdown_cell(
                "### Answers\n\n"
                "**Theoretical reduction:** With independent zero-mean noise of equal variance, "
                r"averaging $N$ images reduces the noise std by $\sqrt{N}$. "
                r"For $N=10$, that is $\sqrt{10}\approx 3.162$."
                "\n\n"
                "**Practical reduction:** See the printed values above (same top-left sky ROI). "
                r"The measured ratio is close to $\sqrt{10}$ but not exact, "
                "due to residual structure, JPEG, and finite sampling."
            ),
            new_markdown_cell(
                "## Task 2 — Derivatives and gradient magnitude\n\n"
                "Prewitt kernels from the specification. OpenCV `filter2D` correlates, "
                "so kernels are flipped for convolution. Float64 keeps signed values; "
                r"magnitude is $\sqrt{I_x^2+I_y^2}$."
            ),
            new_code_cell(
                "PX = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float64)\n"
                "PY = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float64)\n"
                "\n"
                "def prewitt_gradient(image):\n"
                "    image = np.asarray(image, dtype=np.float64)\n"
                "    dx = cv2.filter2D(image, cv2.CV_64F, cv2.flip(PX, -1),\n"
                "                      borderType=cv2.BORDER_REFLECT_101)\n"
                "    dy = cv2.filter2D(image, cv2.CV_64F, cv2.flip(PY, -1),\n"
                "                      borderType=cv2.BORDER_REFLECT_101)\n"
                "    magnitude = np.hypot(dx, dy)\n"
                "    return dx, dy, magnitude\n"
                "\n"
                "dx, dy, magnitude = prewitt_gradient(average)\n"
                "\n"
                'plt.imshow(dx, cmap="gray")\n'
                'plt.axis("off")\n'
                "plt.show()\n"
                "\n"
                'plt.imshow(dy, cmap="gray")\n'
                'plt.axis("off")\n'
                "plt.show()\n"
                "\n"
                'plt.imshow(magnitude, cmap="gray")\n'
                'plt.axis("off")\n'
                "plt.show()"
            ),
            new_markdown_cell(
                "**回答：** Prewitt 算子。最终图像是梯度幅值 "
                r"$\|\nabla I\|=\sqrt{I_x^2+I_y^2}$."
            ),
            new_markdown_cell(
                "## Task 3 — Unsharp masking\n\n"
                r"$L=I*G_\sigma$, $H=I-L$, $O=I+aH$. "
                "Parameters are set first; float64 computation, then clip to [0, 255] for display."
            ),
            new_code_cell(
                "SIGMA = 1.0\n"
                "AMOUNT = 2.0\n"
                "\n"
                "def unsharp_mask(image, sigma, amount):\n"
                "    image = np.asarray(image, dtype=np.float64)\n"
                "    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma,\n"
                "                               borderType=cv2.BORDER_REFLECT_101)\n"
                "    return np.clip(image + amount * (image - blurred), 0, 255)\n"
                "\n"
                'bridge = load_gray("Bridge.jpg", IMAGE_DIR)\n'
                "sharpened = unsharp_mask(bridge, SIGMA, AMOUNT)\n"
                "\n"
                'plt.imshow(bridge, cmap="gray", vmin=0, vmax=255)\n'
                'plt.axis("off")\n'
                "plt.show()\n"
                "\n"
                'plt.imshow(sharpened, cmap="gray", vmin=0, vmax=255)\n'
                'plt.axis("off")\n'
                "plt.show()\n"
                "\n"
                "detail = np.s_[250:450, 500:800]\n"
                'plt.imshow(bridge[detail], cmap="gray", vmin=0, vmax=255)\n'
                'plt.axis("off")\n'
                "plt.show()\n"
                "\n"
                'plt.imshow(sharpened[detail], cmap="gray", vmin=0, vmax=255)\n'
                'plt.axis("off")\n'
                "plt.show()"
            ),
            new_markdown_cell(
                "The sharpened image has higher local contrast around fine details. "
                "Clipping is applied only after the floating-point calculation."
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
        timeout=180,
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
