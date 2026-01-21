# Multicalibration at Web Scale - Replication Materials for Benchmark Experiments

This repository contains replication materials for the benchmark experiments for the paper ["Multicalibration at Web Scale"](https://arxiv.org/pdf/2509.19884)

## How to run

The was tested with python 3.12, you need to install the dependencies in the `requirements.txt` file. You can do this by running the following command in the root directory of the repository:

```
pip install -r requirements.txt
```

To generate the results, run:

```bash
python run_experiments.py
```

By default, results will be store in `results/`.


## Acknowledgement

The code is is based on [Hansen et al.'s](https://arxiv.org/abs/2406.06487v1) code for their paper "When is Multicalibration Post-Processing necessary" (https://github.com/dutchhansen/empirical-multicalibration). We thank the authors for making their code available under the MIT License (see LICENSE file).
