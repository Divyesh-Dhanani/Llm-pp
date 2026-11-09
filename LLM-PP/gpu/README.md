# Re-running the LLM trainings

The analysis in `LLM-PP/` starts from the per-run test predictions committed under `LLM-PP/runs/`, so re-running this
folder is optional. It trains and tests all 66 LLM runs of the paper: about two weeks on one RTX 4090 (1.3–2.4 h per
run without source code, 7.3–7.8 h with code; Table S2 of the paper) and about 110 GB of disk.

Outputs of a re-run will be close to, but not bit-identical with, the committed ones: QLoRA training on a GPU is not
deterministic, and even the same configuration and seed rerun with a different data order moved the mean $R^2$ of one
configuration by 0.037 (Section VI-E of the paper).

## One command

```bash
cd LLM-PP
python3 -m venv .venv-gpu && . .venv-gpu/bin/activate
pip install -r gpu/requirements-gpu.txt

bash run_all.sh llm --dry-run    # list the 66 jobs and where each result goes; changes nothing
bash run_all.sh llm              # download the code, train and test every run, copy the results into runs/
bash run_all.sh                  # recompute every number, table and figure from them
```

`bash run_all.sh llm` overwrites the committed files under `runs/` once all runs have finished (`git checkout -- runs`
restores them). The base models are downloaded from Hugging Face on first use. Every job first waits until the GPU has
20 GB free, so the machine can be shared. Runs that already finished are skipped, so after an interruption just start
the same command again; failed runs are listed at the end and retried on the next start. Code checkouts, adapters and
logs go to `../work/` (set `WORK=/path` to move them).

To run only a part, give a pattern that matches lines of the `--dry-run` list. Nothing is copied into `runs/` then:

```bash
ONLY='main qwen3-8b e3np' bash gpu/run_llm_experiments.sh      # Qwen3-8B, 3 early epochs without proxies, three seeds
```

## What was tested

From a clean environment and a fresh download of the code, with this script: the install from `requirements-gpu.txt`
gives exactly the study's package versions; the predictor code is identical to the study's; the train, validation and
test prompts are byte-identical for six kinds of run (two and three epochs, no proxies, source code, the segmentation
and the transformer split). One complete run was retrained (Qwen3-8B, three early epochs + proxies, seed 42, 1 h 40 min):

| | First logged losses (steps 1, 20, 40) | Selected checkpoint | $R^2$ | MAE |
|---|---|---|---|---|
| Committed run | 4.0752, 3.4298, 1.3500 | `checkpoint-320` | 0.8268 | 0.0600 |
| Re-run | 4.0752, 3.4293, 1.3498 | `checkpoint-320` | 0.8195 | 0.0626 |

Its predictions correlate at 0.995 with the committed ones. The complete 66-run study has not been repeated with this
script.

The logs of a run contain `--- Logging error ---` tracebacks (`TypeError: not all arguments converted`). They come from
a warning inside the libraries, also appear in the logs of the study's runs, and do not stop or change training.

## The 66 runs

| Runs (folder under `runs/`) | Jobs | Code version |
|---|---|---|
| 2 ep. + proxies, 3 LLMs × 3 seeds (`acc_predict/`) | 9 | patches 00 + 01 |
| 3 ep. + proxies, 3 LLMs × 3 seeds (`acc_predict_3epoch/`) | 9 | patches 00 + 01 |
| Qwen3-8B, 3 ep. without proxies; 3 ep. + proxies with answer-only loss (`acc_predict_ablation/{e3np,e3p_ro}/`) | 6 | patches 00 + 01 |
| Qwen3-8B with source code, sequences truncated at 1,024 tokens (`acc_predict_ablation/*_trunc1024/`) | 9 | patches 00 + 01 |
| Qwen3-8B with source code (`acc_predict_ablation/{code2p,code3p,code3p_ro}/`) | 9 | patches 00 + 01 + 02 |
| Llama-3.1-8B with source code (`acc_predict_ablation/code3p/`) | 3 | patches 00 + 01 + 02 |
| Controls: batch size, checkpoint selection (`acc_predict_ablation/{e3p_ro_b1,e3p_all,e3p_ro_all}/`) | 9 | patches 00 + 01 + 02 |
| Further held-out families, seed 0 (`acc_predict_folds/<fold>/`): Gemma-2-9B `e2p`, `e3p` on `tf`, `rnn`, `det`, `seg`; Qwen3-8B `e3p_ro_b1`, `code3p_ro` on `tf`, `rnn` | 12 | patches 00 + 01 + 02 |

The two-epoch runs with Gemma-2-9B use exactly the code of its three-epoch runs, so that the third-epoch comparison is
matched. Not part of the 66, and not re-run by this script: the single-seed exploratory runs (`runs/paper/exploratory_56/`),
the earlier two-epoch Gemma-2-9B runs (`runs/superseded/`, made with torchao 0.14.1 instead of 0.16.0; $R^2$ 0.714 ± 0.018
against 0.709 ± 0.018), and the environment check (`runs/acc_predict_envcheck/`), which re-evaluates an adapter trained on
a second machine with `python run_envcheck.py <run_dir>`.

## Environment

| Component | Version used |
|---|---|
| GPU | NVIDIA RTX 4090, 24 GB |
| Python | 3.12.3 |
| PyTorch | 2.9.1 (CUDA 12.8) |
| Transformers | 5.2.0 |
| TRL | 0.24.0 |
| Unsloth | 2026.3.4 |
| PEFT | 0.21.1 (the two-epoch Qwen3-8B and Llama-3.1-8B runs: 0.21.0, second machine) |
| bitsandbytes | 0.50.2 |
| torchao | 0.16.0 |
| LEMUR | `nn-dataset` 2.2.13, commit `773342a` of [`ABrain-One/nn-dataset`](https://github.com/ABrain-One/nn-dataset) |

`requirements-gpu.txt` pins every package of that environment. Base models (4-bit, from Hugging Face):
`unsloth/Qwen3-8B-unsloth-bnb-4bit`, `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit`, `unsloth/gemma-2-9b-it-bnb-4bit`,
selected with `ACC_MODEL=qwen3-8b`, `llama3.1-8b` or `gemma2-9b`.

## Code

The predictor is the `ab/gpt/act/acc_predictor` package of
[`ABrain-One/nn-gpt`](https://github.com/ABrain-One/nn-gpt), pinned at commit **`e3269ce`**, plus these patches, which
`run_llm_experiments.sh` applies by itself in two checkouts under `work/`:

| Patch | Changes | Used by |
|---|---|---|
| `00-run-folders-model-presets-seed.patch` | One output folder per model and seed, model selection through `ACC_MODEL`, seed passed to the trainer. | every run |
| `01-gemma-system-role-and-answer-only-loss.patch` | Gemma-2 has no system role: fold the system prompt into the user turn. Adds the answer-only loss option (`TRAIN_ON_RESPONSES_ONLY`). Neither changes training for the other runs. | every run |
| `02-sequence-length-fix.patch` | Passes the trainer an explicit `SFTConfig` with `max_length=6144`. Without it, TRL/Unsloth silently truncate training sequences to 1,024 tokens. **Side effect:** Unsloth's config subclass fixes `data_seed=3407` and enables padding-free batching, so all seeds see the same data order. | runs with "fixed" data order (root `README.md`) |

To make the same checkout by hand (the stage `inputs` of `run_all.sh` needs one):

```bash
git clone https://github.com/ABrain-One/nn-gpt.git && cd nn-gpt && git checkout e3269ce
git apply /path/to/LLM-PP/gpu/00-*.patch /path/to/LLM-PP/gpu/01-*.patch /path/to/LLM-PP/gpu/02-*.patch
```

The later runs of the study were made after updating `nn-gpt` to `6b1fba8c`. Among the files these runs import, only
`infer.py` differs from the pinned version, and the function that changed is not used by training or testing.

Every runner passes the warm-up explicitly (64 steps, 2% of the 3,200 scheduled steps), because the package default
changed to 2 steps after the first runs.

## Frozen inputs

Two inputs of the runs are not reproducible from their sources and therefore ship with this repository:

- **The LEMUR export** (`runs/paper/lemur_export.jsonl.gz`, 6,349 runs with their early accuracies, targets and source
  code). nn-dataset builds a local database (`db/ab.nn.db`) from the current LEMUR records, so the same package
  version gives a different export today (we found 638 instead of 657 test runs). Every run of the study used this one
  export (identical checksum in all run folders). With `RAW` set, the runners copy it instead of rebuilding it.
- **The zero-cost proxy cache** (`runs/paper/proxy_cache.jsonl`, and `proxy_norm_stats.json` fitted on the training
  families). The proxies are computed at random initialisation on the GPU; recomputing them (about 5 minutes) gives
  identical depth and parameter counts but gradient-based scores (SynFlow, grad-norm, SNIP, Fisher, GraSP) that differ
  by up to 77% from the cached ones, because GPU gradient computations are not deterministic.

## Files

| File | Purpose |
|---|---|
| `run_llm_experiments.sh` | everything: code checkouts, all jobs, collection |
| `common.sh` | the list of the 66 jobs, and the wait for a free GPU |
| `run_ablation_seeds.sh`, `run_3epoch_seeds.sh` | train one model on one or more jobs, then test in a fresh process; log to `logs/*.status` |
| `run_ablation_phase.py`, `run_3epoch_phase.py` | one phase of one run: `prep` (build prompts only), `train`, `test`, or `ckpt` (evaluate one saved checkpoint) |
| `frozen_export.py` | use the frozen LEMUR export |
| `collect_runs.sh` | copy predictions, metrics and trainer logs into `../runs/`; adapters (~350 MB each) and prompt files are not copied |
| `run_envcheck.py` | environment check |

Arms with the suffix `_all` keep every epoch's checkpoint and evaluate each on validation and test. `FOLD=tf`, `rnn`,
`det` or `seg` selects a further held-out family (`../folds.py`).
