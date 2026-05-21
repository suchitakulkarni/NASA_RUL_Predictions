# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository. The following are the baseline rules. 
## Always use a logging facility you can use setup_logging defined below

def setup_logging(level=logging.INFO):
    """
    Call once from main.py. All modules use logging.getLogger(__name__)
    and inherit this configuration automatically.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(RESULTS_DIR, "run.log"), mode="w")
        ]
    )

## Project organisation vs scale

Match structure to project size. Do not over-engineer small projects, do not under-engineer large ones.

| Scale | Structure |
|---|---|
| ≤ 3 source files / single script | flat layout: `main.py`, `config.py`, `utils.py` at root — no `src/` |
| 4–10 source files | canonical layout below, but only create modules that are actually needed |
| > 10 source files or multiple entry points | full canonical layout; consider sub-packages inside `src/` |

Never create empty placeholder modules "for later." Add files when they have real content.

## Canonical directory structure (for medium/large projects)

```
project/
  src/
    config.py       # infrastructure: paths, RESULTS_DIR, logging levels
    hparams.py      # experiment hyperparameters as a dataclass
    data_io.py      # loading, saving, format conversions
    processing.py   # transformations, filtering, feature engineering
    plotting.py     # all visualization routines
    models.py       # core computation / ML / physics
    utils.py        # shared helpers
  tests/
  results/
  data/
    raw/
    processed/
  experiments/      # one YAML file per experiment run
  main.py
  agent.md
```

## Module separation

Keep module boundaries strict: plotting in `plotting.py`, data wrangling in `processing.py`, paths/constants in `config.py`. No plotting code in `main.py` or analysis modules; no I/O logic in plotting modules.
## All imports should be at the top of the file

## Plotting rules
1. Do not name files with Fig_1_*, Fig_2_*. Simply a descriptive filename would do.
2. Keep equal aspect ratio unless the situation demands otherwise
3. Use presentation.mpstyle file as long as possible 
4. unless absolutely necessary, do not use label/title fontsizes less than 12
5. Use colors such as red, green, blue, magenta, cyan

## Always have a unit test mechanism via pytest  and a CI/CD pipeline for git commit
## Keep the code FastAPI deployment friendly 
## Write the current state of the project to agent.md at the end session. The agent.md should contain current status, what works, what's broken, next steps. 
## Constants management

Split constants into two categories with different lifecycles:

**Infrastructure constants → `config.py`**
Paths, directory names, logging levels, file formats. These rarely change and do not need tracking.

**Experiment hyperparameters → `src/hparams.py` as a dataclass**
Anything you would ever want to compare across runs (hidden dims, layer counts, learning rate, batch size etc.) belongs here, not in `config.py`.

```python
# src/hparams.py
from dataclasses import dataclass, asdict

@dataclass
class HParams:
    hidden_dim: int = 256
    n_layers: int = 4
    learning_rate: float = 1e-3
    batch_size: int = 32
```

Load from a versioned YAML file in `main.py`:

```python
import yaml
from src.hparams import HParams

with open("experiments/run_01.yaml") as f:
    hparams = HParams(**yaml.safe_load(f))
```

Pass `hparams` explicitly to functions that need it. All callsites read `hparams.hidden_dim` etc. — transparent, no magic.

When adding an experiment tracker, log with `mlflow.log_params(asdict(hparams))` or equivalent. Nothing else changes.

Do not use Hydra or similar frameworks unless there is a concrete need — they obscure where values come from.
