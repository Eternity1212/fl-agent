# Data card — RFMiD (primary)

## Dataset

- **Name**: Retinal Fundus Multi-Disease Image Dataset (RFMiD) 1.0  
- **Task**: Multi-label fundus disease classification  
- **Pointers** (verify license on download page you use):  
  - Paper: https://www.mdpi.com/2306-5729/6/2/14  
  - IEEE DataPort: https://ieee-dataport.org/open-access/retinal-fundus-multi-disease-image-dataset-rfmid  
  - Hugging Face mirror (loader convenience): https://huggingface.co/datasets/ctmedtech/RFMID  

## Local layout

Place extracted files under:

```
data/raw/rfmid/
```

Do **not** commit raw images to git.

## Loader (v0.1 — local CSV + folder)

```python
from pathlib import Path

from fed_agent.data.rfmid import RFMiDLocalDataset

ds = RFMiDLocalDataset(
    labels_csv=Path("data/raw/rfmid/RFMiD_Training_Labels.csv"),
    images_dir=Path("data/raw/rfmid/RFMiD_Training_Data"),
)
sample = ds[0]  # keys: image_id, image (PIL RGB), label (np.float32 vector), label_names
```

Optional online smoke (small download, requires `pip install -e ".[data]"`):

```bash
python -m fed_agent.tools.download_rfmid --smoke-stream
```

## Federated splits (planned)

- `split_noniid_dirichlet.json` — label skew across virtual clients (**CLI implemented**)  
- `split_domain_shift.json` — if camera/device metadata exists in CSV, map that field; otherwise use **domain-hash** synthetic split (**CLI implemented**)  

Generate files:

```bash
python -m fed_agent.tools.build_splits \
  --labels_csv data/raw/rfmid/RFMiD_Training_Labels.csv \
  --out_dir configs/splits/generated \
  --n_clients 8 \
  --seed 0
```

See `configs/splits/README.md` for output layout.

## Optional

- **RFMiD 2.0** (smaller, long-tail): https://zenodo.org/records/7505822  
