"""Execute the feature-preparation notebook in a fresh Python process.

This lightweight runner is used because nbconvert/nbformat are not installed in
the project interpreter. It preserves execution counts and captured stdout in
the notebook JSON and fails immediately on any cell exception.
"""
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = (
    Path(sys.argv[1]).resolve()
    if len(sys.argv) > 1
    else ROOT / "notebooks" / "02_feature_engineering_data_preparation.ipynb"
)

notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
namespace = {"display": lambda value: None}
execution_count = 0
old_cwd = Path.cwd()

try:
    os.chdir(NOTEBOOK.parent)
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        execution_count += 1
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        stdout = StringIO()
        stderr = StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            exec(compile(source, f"cell_{index}", "exec"), namespace)
        outputs = []
        if stdout.getvalue():
            outputs.append({"name": "stdout", "output_type": "stream", "text": stdout.getvalue()})
        if stderr.getvalue():
            outputs.append({"name": "stderr", "output_type": "stream", "text": stderr.getvalue()})
        cell["execution_count"] = execution_count
        cell["outputs"] = outputs
finally:
    os.chdir(old_cwd)

NOTEBOOK.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Executed and saved {execution_count} code cells in {NOTEBOOK}")
