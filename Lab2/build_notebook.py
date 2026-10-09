"""Build and execute a simplified Lab2.ipynb from this source of truth."""
from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "Lab2.ipynb"
SUBMIT = ROOT / "Lab2_submit.ipynb"


def build() -> nbformat.NotebookNode:
    return new_notebook(
        cells=[
            new_markdown_cell(
                "# COMP9517 26T3 — Lab 2\n\n"
                "SIFT keypoints and image stitching. "
                "Change `IMAGE_DIR` / `PERSONAL_PATHS` below if needed. "
                "`read_image(path)` takes the image path as an argument."
            ),
            new_code_cell(
                "%matplotlib inline\n"
                "from pathlib import Path\n"
                "import cv2\n"
                "import numpy as np\n"
                "import matplotlib.pyplot as plt\n"
                "\n"
                "cv2.setRNGSeed(9517)\n"
                'IMAGE_DIR = Path("COMP9517_26T3_Lab2_Images")\n'
                "PERSONAL_PATHS = (IMAGE_DIR / \"p1.jpg\", IMAGE_DIR / \"p2.jpg\")\n"
                "\n"
                "def read_image(path, max_width=None):\n"
                "    image = cv2.imread(str(path))\n"
                "    if max_width and image.shape[1] > max_width:\n"
                "        scale = max_width / image.shape[1]\n"
                "        image = cv2.resize(\n"
                "            image, (max_width, round(image.shape[0] * scale)),\n"
                "            interpolation=cv2.INTER_AREA,\n"
                "        )\n"
                "    return image\n"
                "\n"
                "def show(image):\n"
                '    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))\n'
                "    plt.show()\n"
                "\n"
                "def sift_keypoints(image, nfeatures=0):\n"
                "    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)\n"
                "    return cv2.SIFT_create(nfeatures=nfeatures).detect(gray)\n"
                "\n"
                "def draw_kp(image, keypoints):\n"
                "    return cv2.drawKeypoints(\n"
                "        image, keypoints, None, color=(0, 255, 0),\n"
                "        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,\n"
                "    )\n"
                "\n"
                'images = [read_image(IMAGE_DIR / name) for name in ("Image1.jpg", "Image2.jpg")]\n'
                "for im in images:\n"
                "    show(im)"
            ),
            new_markdown_cell(
                "## Task 1 — SIFT keypoints\n\n"
                "**(a)** Default SIFT parameters. "
                "**(b)** `nfeatures=20` to keep about the 20 strongest features."
            ),
            new_code_cell(
                "default_kp = [sift_keypoints(im) for im in images]\n"
                "reduced_kp = [sift_keypoints(im, nfeatures=20) for im in images]\n"
                "\n"
                "for im, kp in zip(images, default_kp):\n"
                "    show(draw_kp(im, kp))\n"
                "\n"
                "for im, kp in zip(images, reduced_kp):\n"
                "    show(draw_kp(im, kp))"
            ),
            new_markdown_cell(
                "## Task 2 — Scale, rotation, salt-and-pepper\n\n"
                "Same reduced-keypoint settings as Task 1b "
                "(`nfeatures=20`) on each processed image."
            ),
            new_code_cell(
                "def rotate_clockwise(image, degrees=60):\n"
                "    h, w = image.shape[:2]\n"
                "    M = cv2.getRotationMatrix2D((w / 2, h / 2), -degrees, 1)\n"
                "    c, s = abs(M[0, 0]), abs(M[0, 1])\n"
                "    nw, nh = int(np.ceil(w * c + h * s)), int(np.ceil(h * c + w * s))\n"
                "    M[0, 2] += nw / 2 - w / 2\n"
                "    M[1, 2] += nh / 2 - h / 2\n"
                "    return cv2.warpAffine(image, M, (nw, nh))\n"
                "\n"
                "def salt_pepper(image, amount=0.05, seed=9517):\n"
                "    rng = np.random.default_rng(seed)\n"
                "    out = image.copy()\n"
                "    r = rng.random(image.shape[:2])\n"
                "    out[r < amount / 2] = 0\n"
                "    out[(r >= amount / 2) & (r < amount)] = 255\n"
                "    return out\n"
                "\n"
                "processed = [\n"
                "    [cv2.resize(im, None, fx=1.2, fy=1.2) for im in images],\n"
                "    [rotate_clockwise(im) for im in images],\n"
                "    [salt_pepper(im, seed=9517 + i) for i, im in enumerate(images)],\n"
                "]\n"
                "\n"
                "for pair in processed:\n"
                "    for im in pair:\n"
                "        show(draw_kp(im, sift_keypoints(im, nfeatures=20)))"
            ),
            new_markdown_cell(
                "### Answers\n\n"
                "Many keypoints stay on the same scene structures after scaling and rotation, "
                "because SIFT is designed to be scale- and rotation-invariant "
                "(characteristic scale + local orientation).\n\n"
                "Salt-and-pepper noise changes local contrast, so some keypoints move "
                "or disappear and new ones appear.\n\n"
                "For these images, SIFT is most robust to **scale** and **rotation**; "
                "it is least robust to **salt-and-pepper noise**."
            ),
            new_markdown_cell(
                "## Task 3 — Matching, RANSAC and stitching\n\n"
                "BFMatcher + `knnMatch(k=2)`, Lowe ratio 0.75, "
                "then `findHomography(..., cv2.RANSAC)`. "
                "Repeat on the personal pair via `PERSONAL_PATHS`."
            ),
            new_code_cell(
                "def stitch_pair(first, second):\n"
                "    sift = cv2.SIFT_create()\n"
                "    k1, d1 = sift.detectAndCompute(cv2.cvtColor(first, cv2.COLOR_BGR2GRAY), None)\n"
                "    k2, d2 = sift.detectAndCompute(cv2.cvtColor(second, cv2.COLOR_BGR2GRAY), None)\n"
                "\n"
                "    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d1, d2, k=2)\n"
                "    good = [m for m, n in pairs if m.distance < 0.75 * n.distance]\n"
                "\n"
                "    # Task 3a: selected correspondences\n"
                "    show(cv2.drawMatches(\n"
                "        first, k1, second, k2, good, None,\n"
                "        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,\n"
                "    ))\n"
                "\n"
                "    # Task 3b: RANSAC homography and stitch\n"
                "    src = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)\n"
                "    dst = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)\n"
                "    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)\n"
                "\n"
                "    h1, w1 = first.shape[:2]\n"
                "    h2, w2 = second.shape[:2]\n"
                "    corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)\n"
                "    warped = cv2.perspectiveTransform(corners, H).reshape(-1, 2)\n"
                "    both = np.vstack([warped, [[0, 0], [w2, 0], [w2, h2], [0, h2]]])\n"
                "    xmin, ymin = np.floor(both.min(axis=0)).astype(int)\n"
                "    xmax, ymax = np.ceil(both.max(axis=0)).astype(int)\n"
                "    T = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]], dtype=float)\n"
                "    size = (int(xmax - xmin), int(ymax - ymin))\n"
                "\n"
                "    a = cv2.warpPerspective(first, T @ H, size)\n"
                "    b = cv2.warpPerspective(second, T, size)\n"
                "    panorama = np.where(b > 0, b, a)\n"
                "\n"
                "    # Crop empty border.\n"
                "    gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)\n"
                "    ys, xs = np.where(gray > 0)\n"
                "    panorama = panorama[ys.min():ys.max() + 1, xs.min():xs.max() + 1]\n"
                "    show(panorama)\n"
                "\n"
                "stitch_pair(*images)"
            ),
            new_markdown_cell("### Personal image pair"),
            new_code_cell(
                "personal = [read_image(p, max_width=1000) for p in PERSONAL_PATHS]\n"
                "for im in personal:\n"
                "    show(im)\n"
                "stitch_pair(*personal)"
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
        timeout=300,
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
