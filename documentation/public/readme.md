# Public documentation

## How to stage the documentation locally

```
# Install dependencies
python3 -m pip install -r third_party/yggdrasil_decision_forests/documentation/public/requirements.txt

# Start a http server with the documentation
(cd third_party/yggdrasil_decision_forests && mkdocs serve -a localhost:8889 -f documentation/public/mkdocs.yml)
```
