# ODE Lab v3

Includes:
- Python models + integrators + registry + presets: `python/ode_models.py`
- Tiny CLI: `python/ode_lab_cli.py`
- Julia mirror: `julia/ode_models.jl`
- Rust/WASM bindings: `rust-wasm/`
- Web UI: `web/` (live params, export, scenes, 3D attractor projection)

## Build & run UI

```bash
cd rust-wasm
wasm-pack build --target web --release
cd ..
cp -r rust-wasm/pkg web/
python -m http.server -d web 8080
# open http://localhost:8080
```

## Run CLI

```bash
cd python
python -m ode_lab_cli list-models
python -m ode_lab_cli list-presets --preset-file presets.json
python -m ode_lab_cli run --preset duffing_chaotic --preset-file presets.json --out-csv out.csv
```
