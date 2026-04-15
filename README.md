# ImageAnalysis
This is a point cloud generation repository.
## Run
With python 3.12, run:
```bash
nix develop
poetry install
poetry run python3 main.py
```
or if you don't have `nix`, ensure many of the libraries from the `flake.nix` are installed and run:
```bash
poetry install
poetry run python3 main.py
```
