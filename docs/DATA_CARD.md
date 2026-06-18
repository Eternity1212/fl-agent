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

## Federated splits (planned)

- `split_noniid_dirichlet.json` — label skew across virtual clients  
- `split_domain_shift.json` — group by camera / device metadata when available  

Scripts will live under `scripts/` (TODO).

## Optional

- **RFMiD 2.0** (smaller, long-tail): https://zenodo.org/records/7505822  
