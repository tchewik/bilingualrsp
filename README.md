# Bilingual Rhetorical Structure Parsing

This repository contains the official code and data for the ACL 2024 Findings paper [Bilingual Rhetorical Structure Parsing with Large Parallel Annotations](https://aclanthology.org/2024.findings-acl.577/).

## Trained Models

This repository focuses on data and experiments. For applying the trained parsers, visit the [IsaNLP RST repository](https://github.com/tchewik/isanlp_rst) for models and usage instructions.

## Data

The data directory structure should be as follows:

```
data/
├── gum_rs3/
│   ├── en/
│   │   └── *.rs3
│   └── ru/
│       └── *_RU.rs3
├── rstdt_rs3/
│   ├── TEST/
│   │   └── wsj_*.rs3
│   └── TRAINING/
│       └── wsj_*.rs3
└── rurstb_rs3/
    ├── train.*_part_*.rs3
    ├── dev.*_part_*.rs3
    └── test.*_part_*.rs3

```

- **gum_rs3/ru/** Contains the **RRG corpus** in Russian. `data/RRG.zip`
- **gum_rs3/en/** Place the GUM RST *.rs3 files here. [GUM dataset link](https://github.com/amir-zeldes/gum).
- **rstdt_rs3/** Place the RST-DT *.rs3 files here. [RST-DT dataset link](https://catalog.ldc.upenn.edu/LDC2002T07).
- **rurstb_rs3/** Contains the RRT corpus; one document = one tree. `data/rurstb_rs3.zip`

The train/dev/test splits for GUM/RRG are listed under `data/gum_file_lists` for GUM v9.1. If you are using a later extended version, you should update these file lists accordingly.

## Experiments

Set ``WANDB_KEY`` in ``dmrst_parser/keys.py`` for online wandb support.
 
### Monolingual Experiments

1. **Train:**
   ```bash  
   python dmrst_parser/multiple_runs.py --corpus "$CORPUS" --lang "$LANG" --model_type "$TYPE" --cuda_device 0 train
   ```

2. **Evaluate:**
   ```bash
   python dmrst_parser/multiple_runs.py --corpus "$CORPUS" --lang "$LANG" --model_type "$TYPE" --cuda_device 0 evaluate
   ```

### Bilingual Experiments

1. **Train:**
   ```bash
   python dmrst_parser/multiple_runs.py --corpus 'GUM' --lang "$LANG" --model_type "$TYPE" train_mixed --mixed 100
   ```

2. **Evaluate:**
   ```bash
   python utils/eval_dmrst_transfer.py --models_dir saves/path-with-models \
                                       --corpus 'GUM' --lang "$LANG2" --nfolds 5 evaluate
   ```

### Parameters

- `LANG`: `en`, `ru`
- `CORPUS`: `RST-DT`, `GUM` (RRG with `lang=ru`), `RuRSTB` (RRT)
- `TYPE`: `default`, `+tony`, `+tony+bilstm_edus`


