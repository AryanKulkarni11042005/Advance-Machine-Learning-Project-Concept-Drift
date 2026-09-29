# Comparative Evaluation of Concept Drift Adaptation Strategies for Streaming Classification: A Benchmark Study Across Synthetic and Real-World Data

This repository studies online classification under concept drift. It compares classical and recurrent neural strategies on synthetic streams with controlled drift and real-world streams whose drift points are not externally labelled.

> Advanced Machine Learning project (CE29T/CE29P)

## Research question

Which concept-drift adaptation strategy restores predictive performance most effectively after drift, and how does effectiveness vary across drift types, model families, and computational constraints?

## Project status

| Phase | Status | Scope |
|---|---|---|
| 1 | Complete | Baseline streaming benchmark infrastructure |
| 2 | Complete | LSTM and GRU online adaptation integration |
| 3 | Complete | Multi-dataset benchmark: 330 runs across six datasets and eleven strategies |
| 4 | Next | Controlled ablation and sensitivity analysis |
| 5 | Planned | Statistical analysis |
| 6 | Planned | Visualization |
| 7 | Planned | Research paper and documentation |

Phase 4 and later work have not been run as part of the reported Phase 3 results.

## Repository structure

```text
.
├── configs/
│   └── phase3.yaml                 # Frozen Phase 3 matrix and model settings
├── data/                           # Local cached data, including Airlines parquet
├── docs/
│   └── findings.md                 # Earlier working notes; see final Phase 3 artifacts
├── notebooks/
│   ├── 01_sea.ipynb                # SEA stream exploration
│   ├── river_stock_market.ipynb    # Stock stream work
│   ├── stock_market.ipynb          # Stock data exploration
│   └── stock_market_cleaned.csv     # Prepared stock stream input
├── results/
│   └── phase3/
│       └── phase3_20260929_121546/ # Final Phase 3 artifacts
├── scripts/
│   ├── download_data.py            # Fetch/cache real-world data
│   ├── experiment_neural_drift.py  # Small neural drift experiment
│   ├── export_results.py           # Export legacy runs or rebuild a Phase 3 summary
│   ├── run_phase3.py               # Phase 3 benchmark with checkpoint/resume
│   └── smoke_neural.py             # Neural integration smoke run
├── src/
│   ├── experiments.py              # Earlier classical experiment grid and summaries
│   ├── pipeline.py                 # Streaming evaluation, adaptation, metrics, MLflow
│   ├── streams.py                  # Dataset loaders and synthetic stream generators
│   └── models/
│       ├── networks.py             # LSTM and GRU binary classifiers
│       ├── neural_adapter.py       # Online prediction/update/reset wrapper
│       └── sequence.py             # Causal rolling sequence builder
├── environment.yml                # Conda environment specification
├── requirements.txt                # Captured Python package environment
├── LICENSE.md
└── README.md
```

The Phase 3 deliverables are in `results/phase3/phase3_20260929_121546/`. The legacy exporter can create `results/raw_runs.csv` and `results/summary.csv`; those files are not present in the current checkout and are separate from the Phase 3 run outputs. The `data/` directory and local MLflow SQLite database are machine-local inputs/state.

## Implemented methodology

### Problem formulation and concept drift

The task is binary streaming classification: observations arrive in temporal order, a prediction is made for each eligible observation, and the model updates as labels become available. Concept drift is a change in the relationship between stream features and labels, which can reduce the usefulness of a model learned from earlier observations. The benchmark compares explicit adaptation policies with an online learner that has no explicit drift-triggered policy.

The Phase 3 matrix includes abrupt and gradual SEA transitions, continuously rotating Hyperplane, and temporal real-world streams. Synthetic streams provide known drift indices when the change is discrete. Hyperplane changes continuously and has no single event index. Elec2, Airlines, and S&P 500 have no externally verified drift ground truth in this project.

### Prequential evaluation and label timing

The pipeline follows test-then-train evaluation. For each observation it constructs the available causal representation and predicts before learning from that observation's label. A prediction is scored when its label is revealed; prediction errors are then passed to ADWIN when a detector is active, adaptation is applied if it triggers, and the learner is updated with the revealed label. Delayed labels remain queued until their configured reveal time. This ordering is intended to prevent target leakage.

The recurrent models use a rolling sequence containing the current and preceding feature rows. The first `sequence_length - 1` observations cannot produce a neural prediction because the sequence is not yet full. For the S&P 500 next-day direction task, the Phase 3 configuration sets `label_delay: 1`: a label is held for one subsequent stream step before scoring and model update. Other configured datasets use delay zero.

### Datasets

| Configuration key | Dataset | Stream characteristics | Drift reference in this benchmark |
|---|---|---|---|
| `sea_abrupt` | SEA Concepts | Synthetic abrupt concept transition; Phase 3 variants `[3, 2]`, 5,000 samples per concept | Known transition index/indices |
| `sea_gradual` | SEA Concepts | Synthetic probabilistic transition of 1,000 samples between concepts; 5,000 stationary samples per concept | Transition midpoint convention |
| `hyperplane` | Rotating Hyperplane | Synthetic continuous rotation; 10,000 samples, 10 features, 5 drifting features, `mag_change: 0.001` | No single drift index; recovery time is not defined |
| `elec2` | Electricity (Elec2) | Real-world market stream; 45,312 samples | No externally verified drift index |
| `airlines` | Airlines | First 100,000 rows, retained in temporal order | No externally verified drift index |
| `stock_sp500` | S&P 500 | Prepared daily direction stream, scale-free features, temporal order | No externally verified drift index |

The synthetic generators are implemented in `src/streams.py`. Elec2 is read through River; Airlines data is cached at `data/airlines.parquet` by `scripts/download_data.py`. The S&P 500 loader uses `notebooks/stock_market_cleaned.csv` (or the configured relative path). Download/build local real-world inputs before running the full benchmark.

### Compared strategies

The exact Phase 3 identifiers are:

| Family | Strategy | Implemented behavior |
|---|---|---|
| Classical | `baseline` | Online Hoeffding Tree updates without an explicit drift-triggered adaptation policy |
| Classical | `blind_periodic` | Periodically resets and refits the learner from its recent buffer, without a drift detector |
| Classical | `detector_full_retrain` | ADWIN-triggered learner reset followed by refitting from the recent buffer |
| Classical | `detector_incremental` | ADWIN observes prediction errors, while the learner continues its ordinary incremental updates without reset |
| Classical | `arf` | River Adaptive Random Forest classifier |
| Neural | `lstm`, `gru` | Causal sequence-based recurrent classifier with online updates |
| Neural | `lstm_adwin_reset`, `gru_adwin_reset` | Corresponding recurrent model is reinitialized when ADWIN detects drift |
| Neural | `lstm_adwin_finetune`, `gru_adwin_finetune` | On detection, the model receives additional updates from a bounded buffer of previously revealed sequences/labels |

`detector_incremental` is a useful control: the detector is active, but it does not cause a reset. It helps distinguish detector-triggered interventions from the base learner's normal online updates.

### Recurrent model details

Phase 3 uses a one-layer LSTM or GRU with hidden size 16 and a two-class linear classification head. Each input is a causal rolling sequence of length 10. The network predicts from the final recurrent output. The configured optimizer is Adam with learning rate 0.01 and one gradient update per revealed label. A fine-tuning policy can replay up to 32 previously revealed sequences after a detection; the regular online LSTM/GRU policies do not use that replay intervention. Reset policies reinitialize the model using that run's seed. All neural models in the implementation run on CPU.

### Drift detection and adaptation

ADWIN is configured with `delta: 0.002` and `clock: 32` in `configs/phase3.yaml`. The pipeline updates it with binary prediction error when a valid prediction's label becomes available. A detected change can trigger a full classical reset/refit, no explicit classical intervention (the incremental control), neural reinitialization, or neural fine-tuning. ADWIN is not enabled for the plain baseline, blind periodic policy, or ARF strategy in this benchmark.

### Airlines feature handling

The tree strategies receive categorical values natively. For neural Airlines runs, the pipeline applies a deterministic feature-wise BLAKE2b-64 hash into 32 buckets per categorical feature. Numeric features use a fixed signed `log1p` transform. No vocabulary or scaling statistics are fitted from the stream, so future rows cannot influence earlier feature representations. Hashing is fixed-width and can map distinct categories to the same bucket (collisions).

### Metrics and recovery definition

Run-level output records final prequential accuracy, macro F1, runtime, peak traced Python memory, valid prediction count, drift detection count, majority-class reference accuracy, and recovery fields. Windowed accuracy is also computed for recovery evaluation and logged to MLflow. The Phase 3 summary reports run counts, independent-unit counts, seed counts, and means/standard deviations for the configured metrics.

Recovery is measured only against supplied drift indices. The implementation takes the last windowed accuracy strictly before a drift as the baseline, sets the recovery threshold to that value minus the configured tolerance (0.02), and finds the first later evaluation point whose windowed accuracy meets that threshold. Recovery time is the number of stream samples from the drift index to that evaluation point. If there is no prior evaluation point, the record is `no_pre_drift_baseline`; if the threshold is never regained, it is `no_recovery`. Streams without discrete ground-truth indices receive `not_applicable` records. Recovery analysis is therefore most interpretable on synthetic streams with known drift structure. Real-world detector events are not substitutes for externally verified drift ground truth. The 200-sample evaluation interval also makes recovery estimates coarse, particularly around gradual transitions.

Peak memory is measured with Python `tracemalloc`; it does not capture all native allocations made by PyTorch or other libraries. Runtime and memory comparisons depend on hardware, software versions, and system load.

## Phase 3 benchmark and results

The completed benchmark is `results/phase3/phase3_20260929_121546/`. It contains the frozen `config.yaml` and the four CSV outputs below. The final files contain 330 run-level records and a summary of 66 dataset × strategy combinations (66 rows × 21 columns), corresponding to six datasets × eleven strategies × five recorded seeds (42, 43, 44, 45, 46).

| File | Contents |
|---|---|
| `config.yaml` | Resolved Phase 3 configuration, run identifier, and environment metadata |
| `raw_runs.csv` | One record per completed dataset × strategy × seed run |
| `summary.csv` | Aggregated metrics by dataset and strategy |
| `drift_events.csv` | ADWIN detection events and prediction/feedback indices |
| `recoveries.csv` | Per-run recovery records, including non-applicable cases |

### Descriptive observations

The results are specific to this configuration and data preparation; they do not establish a universally best concept-drift method.

- ARF had the highest mean accuracy among the evaluated strategies on both SEA abrupt (about 0.975) and SEA gradual (about 0.973).
- On rotating Hyperplane, the LSTM and LSTM fine-tuning means were both about 0.881. Blind periodic adaptation was higher in this benchmark (about 0.891), so the neural values should not be read as a dataset-wide lead.
- On Elec2, ARF was substantially stronger than the neural variants; its mean accuracy was about 0.886.
- On Airlines, LSTM and LSTM fine-tuning reached about 0.680 and 0.681, respectively, above ARF at about 0.654 in this benchmark.
- S&P 500 results were relatively weak (roughly 0.509–0.545 accuracy across strategies), and no ADWIN detections were recorded in the Phase 3 runs.
- Neural reset variants were substantially below their corresponding online or fine-tuning variants on several datasets, including SEA and Hyperplane.

Performance varies by dataset, drift structure, model family, and adaptation policy. These are descriptive benchmark observations, not general guarantees about the algorithms.

### Replication and independence caveat

Five seed values are recorded for every Phase 3 configuration, but five rows are not always five statistically independent replications. The summary includes `n_runs`, `n_independent_runs`, and `n_seeds`. For seeded synthetic streams, stream seeds change the generated stream. For fixed real-world streams, stochastic models such as ARF and neural networks vary by model seed, while deterministic classical strategies reuse the same fixed stream and can produce the same result for every seed. Their summary can therefore show `n_runs = 5`, `n_seeds = 5`, and `n_independent_runs = 1`. Do not treat the five seed entries as five independent observations in those cases.

## Reproducibility and execution

The frozen Phase 3 configuration is `configs/phase3.yaml`; the final run's resolved copy is stored beside its results. It specifies seeds 42–46, model settings, ADWIN settings, dataset arguments, label delays, and the MLflow experiment name. `environment.yml` declares Python 3.11.9 and the Conda environment `aml-project`. `requirements.txt` records a pip package snapshot (including River 0.26.1 and PyTorch 2.4.1).

Create the environment and install local real-world data as needed:

```bash
conda env create -f environment.yml
conda activate aml-project
python scripts/download_data.py
```

The project uses Python, River, PyTorch, pandas, scikit-learn, and MLflow; the environment file also includes plotting and notebook packages. To register the Jupyter kernel for the Conda environment:

```bash
python -m ipykernel install --user --name=aml-project --display-name="Python (aml-project)"
```

The full Phase 3 invocation is:

```bash
python scripts/run_phase3.py
```

The runner also supports bounded development modes and resuming an existing output directory:

```bash
python scripts/run_phase3.py --smoke
python scripts/run_phase3.py --pilot
python scripts/run_phase3.py --resume results/phase3/<run_id>
```

`--smoke` runs the SEA abrupt infrastructure smoke matrix; `--pilot` runs the configured matrix with seed 42 only. Normal execution creates a new run directory. The runner checkpoints CSV outputs after each completed dataset × strategy × seed run using temporary files and replacement. Resume reads that directory's `config.yaml`, skips keys already present in `raw_runs.csv`, and executes only unfinished combinations. Runs that completed computation but were interrupted before their raw result was checkpointed are rerun. The auxiliary event and recovery CSVs are reconstructed from checkpointed raw records to avoid duplicate rows and repair a partial multi-file checkpoint. Only newly executed combinations create MLflow runs when tracking is enabled.

Phase 3 MLflow logging uses the configured experiment name `concept-drift-phase3a` and the repository-local SQLite tracking URI defined in `src/pipeline.py`. Each run logs parameters, final metrics, and windowed accuracy points. To browse the store from the repository root:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

`scripts/export_results.py --phase3-run-dir results/phase3/phase3_20260929_121546` rebuilds the summary for that Phase 3 run directory without mixing it with the legacy top-level export. Running `python scripts/export_results.py` without that option exports the earlier MLflow experiment results to `results/raw_runs.csv` and `results/summary.csv`. The earlier classical grid can also be run with `python src/experiments.py`; it uses its own default matrix and MLflow experiment and is not the Phase 3 benchmark command.

## Experimental limitations

- Real-world datasets do not have externally verified drift ground truth in this benchmark; detection and recovery claims must reflect that.
- Recorded seeds do not make repeated deterministic fixed-stream runs independent.
- `tracemalloc` omits some native library and PyTorch allocations; the current neural implementation runs on CPU.
- Recovery time requires usable drift and recovery structure. It is not meaningful for all streams, and the fixed evaluation interval limits temporal resolution.
- Airlines feature hashing can introduce collisions.
- S&P 500 is a weak/noisy prediction setting in these results and should not be overinterpreted.
- Runtime and memory measurements depend on the execution environment.
- Phase 3 uses one configured ADWIN sensitivity (`delta: 0.002`); it is not a detector or hyperparameter sweep.

## Next: Phase 4

Phase 4 is planned controlled ablation and sensitivity analysis. The work will keep other factors fixed while varying the adaptation policy and model family. Planned comparisons include:

1. LSTM versus GRU.
2. Online neural adaptation versus ADWIN reset versus ADWIN fine-tuning.
3. ADWIN sensitivity at values such as `delta` = 0.001, 0.002, and 0.005.
4. Potentially, DDM as an additional detector if time and resources permit.

These are future experiments; no Phase 4 results are reported here.

## Selected references

There is no project bibliography file in the repository. The following foundational works are relevant to methods used here; citations identify background, not the implementation's authorship.

1. A. P. Dawid. “Statistical Theory: The Prequential Approach.” *Journal of the Royal Statistical Society: Series A*, 147(2), 278–292, 1984. [doi:10.2307/2981683](https://doi.org/10.2307/2981683).
2. A. Bifet and R. Gavaldà. “Learning from Time-Changing Data with Adaptive Windowing.” *Proceedings of the 2007 SIAM International Conference on Data Mining*, 443–448, 2007. [doi:10.1137/1.9781611972771.42](https://doi.org/10.1137/1.9781611972771.42).
3. J. Gama, I. Žliobaitė, A. Bifet, M. Pechenizkiy, and A. Bouchachia. “A Survey on Concept Drift Adaptation.” *ACM Computing Surveys*, 46(4), Article 44, 2014. [doi:10.1145/2523813](https://doi.org/10.1145/2523813).
4. H. M. Gomes et al. “Adaptive Random Forests for Evolving Data Stream Classification.” *Machine Learning*, 106, 1469–1495, 2017. [doi:10.1007/s10994-017-5642-8](https://doi.org/10.1007/s10994-017-5642-8).
5. S. Hochreiter and J. Schmidhuber. “Long Short-Term Memory.” *Neural Computation*, 9(8), 1735–1780, 1997. [doi:10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735).
6. K. Cho et al. “Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation.” *Proceedings of EMNLP 2014*, 1724–1734, 2014. [ACL Anthology D14-1179](https://aclanthology.org/D14-1179/).

## License

This project is licensed under the MIT License; see [LICENSE.md](LICENSE.md).
