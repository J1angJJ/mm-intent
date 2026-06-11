# VLA World Model Lab

Server-side scripts for derived multimodal, VLA-style, and small world-model experiments.

Private planning notes and runbooks are kept under `private/vla_world_model_lab/`.

Typical server entry points:

```bash
python vla_world_model_lab/scripts/build_sample_index.py
python vla_world_model_lab/scripts/build_episode_transitions.py
python vla_world_model_lab/scripts/visualize_dataset.py
python vla_world_model_lab/scripts/run_probe_baselines.py --modalities all
python vla_world_model_lab/scripts/visualize_results.py
```

Generated files are written to `vla_world_model_lab/artifacts/` and are ignored by Git.
