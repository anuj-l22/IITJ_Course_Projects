# Reproduction Guide (SISA Machine Unlearning)

## Environments
- CPU: `environment-cpu.yml`
- CUDA 11.8: `environment-cuda118.yml`
- CUDA 12.4: `environment-cuda124.yml`
- Or `pip install -r requirements.txt`

## Datasets
- **CIFAR-10**: auto-downloaded to `./data` by `torchvision` (default).
- **Purchase-100**: set `dataset: purchase` and `purchase_path` in `code/config.yaml`.

## SISA Defaults
- Small laptop: `K=10, S=2` (already in `code/config.yaml`).
- Workstation (e.g., RTX 5000): `K=20, S=3` via `code/config_s3.yaml`.

## One-liner to run everything
```bash
bash scripts/run_assignment.sh
```

## Verify deliverables
```bash
bash scripts/verify_assignment.sh
```
