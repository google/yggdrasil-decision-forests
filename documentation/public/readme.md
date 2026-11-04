# Public documentation

## How to stage the documentation locally

```
# Create a virtual environment with Python >= 3.12 (ideally 3.14).
python3.14 -m venv /tmp/ydf_docs_venv
source /tmp/ydf_docs_venv/bin/activate

# Install dependencies
python -m pip install -r third_party/yggdrasil_decision_forests/documentation/public/requirements.txt

# Add the `__init__.py` files missing from the ydf 0.17.0 wheel (see the
# `post_install` job in `.readthedocs.yaml`).
python -c "import pathlib, ydf; [(d / '__init__.py').touch() for d in pathlib.Path(ydf.__file__).parent.rglob('*') if d.is_dir() and d.name != '__pycache__']"

# Start a http server with the documentation
(cd third_party/yggdrasil_decision_forests && mkdocs serve -a localhost:8889 -f documentation/public/mkdocs.yml)
```
