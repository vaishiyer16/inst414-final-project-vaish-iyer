import os

folders = [
    "data/extracted",
    "data/processed",
    "data/outputs",
    "data/reference-tables",
    "etl",
    "analysis",
    "vis"
]

files = [
    "main.py",
    "README.md",
    "requirements.txt",
    "etl/extract.py",
    "etl/transform.py",
    "etl/load.py",
    "analysis/model.py",
    "analysis/evaluate.py",
    "vis/visualizations.py"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

for file in files:
    with open(file, 'w') as f:
        pass  # create empty file
