# ADoPT Replication

Re-implementation of **ADoPT: LiDAR Spoofing Attack Detection Based on Point-Level Temporal Consistency**
(Cho, Cao, Zhou, Mao, BMVC 2023, [arXiv:2310.14504](https://arxiv.org/abs/2310.14504)).
No official code was released, so everything here is built from the paper.

**Stage 1 (this version): core detector.** Scene flow with coherence loss, synthesis, cluster-based
detection, simulated attacks on nuScenes, Table 1 / Fig. 6 / Fig. 12 style outputs.
**Stage 2 (next):** voxel Scene Flow Approximation, ablations (Fig. 7, 8, 9), CARLO / 3D-TC2 comparison.

---

## 1. Project layout

```
adopt/
  config.py           all hyper-parameters, each marked [paper] or [assumed]
  geometry.py         crop, voxel downsample, RANSAC ground removal, transforms
  sfe.py              Sec. 5.1  MLP scene flow + Chamfer + coherence loss   (PyTorch)
  pipeline.py         Fig. 3    synthesis of history frames + check of the new frame
  detector.py         Sec. 5.2  cluster-based consistency measurement (DBSCAN)
  attack.py           Sec. 4    simulated dense / sparse point injection
  nuscenes_loader.py            sweeps at 10 Hz, ego-motion compensation, object bank
  metrics.py                    CD and DCD baselines (Fig. 6), FPR / TPR
  synthetic.py                  toy street scene with known motion (no dataset needed)
scripts/
  demo_synthetic.py   quick sanity check + plot
  run_nuscenes.py     main experiment -> CSV
  make_figures.py     CSV -> table.md, metric histograms (Fig. 6), ROC
  sweep_dbscan.py     DBSCAN threshold search (Fig. 12)
  rescore.py          recompute a finished run with new detection settings (no GPU, minutes)
  ablation.py         compare runs on exactly the same keyframes (Fig. 7, 8, 9)
  inspect_misses.py   why each fake was caught / missed, and the likely cause of each false alarm
  diag_sfe.py         scene-flow accuracy vs iterations / coherence / nn_every (known motion)
tests/
  test_core.py        no PyTorch needed
  test_sfe.py         scene-flow tests (PyTorch)
```

## 2. Setup (Windows / Linux, NVIDIA GPU)

```bash
conda create -n adopt python=3.10 -y
conda activate adopt
# PyTorch with CUDA: pick the command for your CUDA version at https://pytorch.org/get-started/locally/
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

In VS Code: open the folder, choose the `adopt` interpreter (Ctrl+Shift+P → *Python: Select Interpreter*).
Ready-made run configs are in the *Run and Debug* panel.

Check everything works:

```bash
python -m pytest tests -q                 # all tests should pass
python -c "import torch; print(torch.cuda.is_available())"   # should print True
```

## 3. Get nuScenes v1.0-mini (~4 GB)

1. Register at https://www.nuscenes.org/nuscenes#download and download **Full dataset (v1.0) → Mini**.
2. Extract so the folder looks like:
```
data/nuscenes/
  maps/  samples/  sweeps/  v1.0-mini/
```
`sweeps/` is required: the method uses 10 Hz history, keyframes are only 2 Hz.

## 4. Run, in this order

**Step A: sanity check (2 minutes)**
```bash
python scripts/demo_synthetic.py --oracle   # detector only, true motion: benign score 0, fake caught
python scripts/demo_synthetic.py            # same scene with the neural scene flow
```
Look at `results/demo_synthetic.png`. If the neural run gives a benign score far above 0,
increase `iters` (see *Tuning*).

Commands are one line each (PowerShell does not accept the `\` line breaks used on Linux).

**Step B: small nuScenes run (all three object types, ~10 min)**
```bash
python scripts/run_nuscenes.py --dataroot data/nuscenes --max-samples 10 --out results/test/dense.csv --save-clouds
```
Each line prints `benign=<score>` and one `<type>@<distance>=<score>` per object type,
e.g. `benign=0  ped@12m=31  cyc@24m=0  car@8m=57`. Benign should be 0, attacked > 0.

**Step C: full experiments (Table 1), best run overnight**
```bash
python scripts/run_nuscenes.py --dataroot data/nuscenes --attack dense  --out results/full/dense.csv  --save-clouds
python scripts/run_nuscenes.py --dataroot data/nuscenes --attack sparse --out results/full/sparse.csv --save-clouds
```

**Step D: report figures**
```bash
python scripts/make_figures.py results/full/dense.csv results/full/sparse.csv --outdir results/figures
python scripts/sweep_dbscan.py results/full/dense_clouds --name dense
python scripts/sweep_dbscan.py results/full/sparse_clouds --name sparse --eps 0.25 0.5 0.75 1.0 1.25
python scripts/inspect_misses.py results/full/dense_clouds
```
To change detection settings after a run (no scene flow needed):
`python scripts/rescore.py results/full/dense.csv --min-height 0.3`, then pass `dense_rescored.csv` to `make_figures.py`.
Outputs in `results/figures/`: `table.md` (Table 1, per object type), `metric_hist_*.png` (Fig. 6),
`range_*.png` (TPR vs distance), `roc_*.png`, `dbscan_sweep_*.png` (Fig. 12), `timing.txt`, `misses/`.

**Useful options of `run_nuscenes.py`**

| Option | Default | Meaning |
|---|---|---|
| `--kinds` | `pedestrian,cyclist,car` | Object types to inject; each gets its own attacked variant, all share one synthesis |
| `--range-min / --range-max` | 5 / 30 m | Distance of the fakes |
| `--range-mode` | `shift` dense, `natural` sparse | `shift`: dense object moved to a random distance, keeping its points. `natural`: objects seen at that distance, real density |
| `--natural-range` | off | With `natural`: take fakes as they come (mostly close) instead of balancing over 5 m bins |
| `--source-max-range` | 12 m | With `shift`: only copy objects seen this close (dense point spacing) |
| `--min-fake-points` | 100 shift, 20 natural | Only copy objects with at least this many LiDAR points |
| `--inject-anywhere` | off | Old placement: fake only avoids the new frame, not the history |
| `--max-samples` | all | Stop after N keyframes |
| `--sample-every` | 1 | Use every N-th keyframe of every scene (6 → ~64 keyframes over all 10 scenes, for ablations) |
| `--resume` | off | Continue an interrupted run; gives the same result as an uninterrupted one |
| `--set key=value` | | Override any setting in `config.py` |

## 5. How the code maps to the paper

| Paper | Code |
|---|---|
| Eq. L_coherence (Sec. 5.1) | `sfe.coherence_loss` (exact O(N) form, checked against the literal O(N²) formula in tests) |
| L = αCD + βL_coh, MLP 6×128, α=1, β=2 | `sfe.fit_flow`, `config.py` (iters / lr tuned, see below) |
| Synthesis of F₁…F_L (Fig. 3, 5) | `pipeline.build_synthesis` |
| Cluster consistency (Sec. 5.2, App. A) | `detector.cluster_consistency` |
| DBSCAN 17 pts / 0.25 (dense), 9 pts / 0.75 (sparse) | `ADoPTConfig.dense()` / `.sparse()` |
| Threshold search (App. B, Fig. 12) | `scripts/sweep_dbscan.py` |
| CD vs DCD vs ours (Fig. 6) | `metrics.py`, `make_figures.py` |

## 6. Assumptions (put these in your report)

The paper leaves these out. Each one is a setting in `config.py`.

| Topic | Our choice | Why |
|---|---|---|
| Attack data | Real object points copied from other scenes, rotated around the sensor, pasted into the newest frame | Paper's dense data was private; sparse followed CARLO |
| Fake placement | Free space in the new frame **and** in the history synthesis (`--inject-anywhere` = old behaviour) | Otherwise fakes land on recently-seen space and test the injector, not ADoPT |
| Dense attack | Bank objects seen within 12 m with ≥ 100 points; if > 200, the compact 200-point chunk around the centre is kept; then moved along the viewing ray to a random distance in 5–30 m, keeping all points | Paper: "visually recognizable", "up to 200 points". The attacker controls the laser, so a fake can be denser than a real object at that distance. Beyond ~12 m the 32-beam ring gap (> 0.25 m) is wider than the paper's DBSCAN radius, so far objects at real density cannot form a cluster (`--range-mode natural` gave 1/30 at 10 cm voxels) |
| Sparse attack | Cars seen at 5–30 m with their real density (balanced per 5 m bin), 20–64 points | Paper: "up to 64 points", cars only |
| Cyclists | Ridden bicycles **and** ridden motorcycles | nuScenes-mini has very few ridden bicycles |
| 10 Hz history | Every 2nd nuScenes sweep (20 Hz) | Paper uses 10 Hz |
| Ego motion | All frames moved into the newest frame's ego coordinates using nuScenes poses | Not stated; makes static scene align without flow |
| Ground | RANSAC plane removal, 0.25 m | Not stated; the ground would be one giant cluster |
| Chamfer | Mean (not sum), matches > 0.5 m ignored | Fig. 4 values (~0.11) imply a mean. With 2 m the final flow pulled history onto fakes (TPR 70% → 100% at 0.5 m on 10 frames) |
| Scene-flow optimisation | 300 iterations, lr 3e-3, neighbours recomputed every 10 iterations | Paper: 30 it., lr 8e-4, which did not converge here (`diag_sfe.py`) |
| Coherence mask DBSCAN | eps 0.4, min 10 | Not stated |
| MLP output init | Near zero | Most points are static after ego compensation |
| Synthesis | Chain the per-pair flow MLPs; pair flows cached and reused | Paper uses voxel approximation (Stage 2) |
| Detection resolution | New frame clustered at 5 cm voxels; scene flow and synthesis use 10 cm | With 10 cm voxels a flat surface (car side) has only ~13 points in a 0.25 m ball, below the paper's 17, so cars could never be detected |
| Flat clusters | History-free clusters flatter than 0.3 m are ignored | A laser ring on a curb / road bump re-samples at a different spot every sweep and looked history-free (the only false alarm in our 10-frame test, height 0.0 m). All faked objects are much taller |
| Near-car zone | History-free clusters centred < 4 m from the sensor are ignored | Each sweep drops points within 3 m of its own position (car body); as the car moves, the ring just outside has no history. 2 of the 4 remaining false alarms in the full run were at 3.0–3.2 m |
| Decision | Frame attacked if any point is left in a history-free cluster (`score > 0`) | Fig. 6 "distance" is not defined |
| ROI | 55 m for flow, 50 m for detection | Avoids false alarms at the range border |

## 7. Tuning tips

* The tuned scene-flow settings are now the defaults in `config.py`; no `--set` needed.
* Override any setting from the command line: `--set iters=100 det_eps=0.4 beta=0`.
* **Benign scores high (many false alarms)** → scene flow is not converging: try `--set iters=100 lr=0.002`,
  or look at which clusters remain (save clouds and plot them).
* **Attacked scores 0** → the fake object is too sparse for the DBSCAN thresholds: run
  `sweep_dbscan.py`, or raise `--min-fake-points`.
* `beta=0` turns off the coherence loss: this is the paper's ablation (a) in Fig. 7.
* Runtime is dominated by scene flow; bigger `input_voxel` = faster, slightly less accurate (paper Fig. 9).

## 8. Expected differences from the paper

Numbers will not match Table 1 exactly: different attack data (the paper's dense data, also on
v1.0-mini, came privately from the 3D-TC2 authors) and unstated preprocessing. The paper's sparse
attack used 355 frames from the v1.0-trainval validation set; ours uses v1.0-mini. What should replicate is the **pattern**:
low FPR, high TPR on small objects, and CD / DCD failing to separate benign from attacked frames.

## 9. Extension: adversarial attacks (FGSM / PGD) with and without defenses

ADoPT has no gradients (DBSCAN), so FGSM and PGD attack a **PointNet object classifier** trained on
the cars, pedestrians and cyclists cut out of nuScenes-mini. The attacker may move every point of an
object by at most `eps` metres per axis (L-infinity budget, in the object's own frame).

| Setting | Model | Defense |
|---|---|---|
| without defenses | `standard` | none |
| with defenses | `fgsm`, `pgd`, `both` | adversarial training on FGSM, PGD, or a 50/50 mix of both |
| with defenses | any model + ADoPT | does ADoPT's temporal check flag the perturbed object? |

Each model is attacked with FGSM alone, PGD alone, and both (an object counts as broken if either
attack fools the model).

```powershell
python -m pytest -q tests                                  # 19 tests, all should pass
python scripts/adv_train.py --smoke                        # 1-minute check on toy shapes
python scripts/adv_eval.py --smoke
python scripts/adv_build_dataset.py --dataroot data/nuscenes   # -> results/adv/objects.npz (~3 min)
python scripts/adv_train.py                                # 4 models -> results/adv/models (~15-30 min)
python scripts/adv_eval.py                                 # tables + results/adv/eval.csv
python scripts/adv_adopt.py --max-frames 3                 # quick check of the ADoPT part
python scripts/adv_adopt.py                                # all frames (~1-1.5 h), add --resume after a stop
```

Files: `adopt/objects.py` (boxes, object frame, dataset), `adopt/adversarial.py` (PointNet, FGSM, PGD,
adversarial training, evaluation), `adopt/adv_check.py` (puts perturbed points back into a saved frame
and reruns ADoPT's cluster check).

Assumptions: classes car / pedestrian / cyclist, where cyclist = any bicycle or motorcycle (too few
ridden ones in mini); objects 4–50 m away with at least 20 points after 5 cm voxels; train/test split by
object instance (about 1 in 5 instances is test); training budget eps = 0.05 m; adversarial training
loss = 0.5 clean + 0.5 adversarial; FGSM training uses a random start (Wong et al., 2020); PGD uses
7 steps in training and 20 in evaluation, step 2.5·eps/steps (Madry et al., 2018).
