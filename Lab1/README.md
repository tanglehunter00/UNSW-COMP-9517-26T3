# Lab 1

Submit the executed notebook. Prefer `Lab1_submit.ipynb` if Cursor keeps overwriting `Lab1.ipynb` when the file is open with outputs.

Rebuild from source (avoids editing a huge ipynb in the editor):

```bash
cd Lab1
python3 -m pip install --user --break-system-packages -r requirements.txt
python3 build_notebook.py
```

Select interpreter `/opt/homebrew/bin/python3`. Keep `COMP9517_26T3_Lab1_Images` beside the notebook. Change `IMAGE_DIR` in the first code cell if needed.
