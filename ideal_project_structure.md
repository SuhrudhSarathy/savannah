savannah/                          # repo root
├── pyproject.toml                   # single source of truth for deps + metadata
├── README.md
├── .env.example                     # WANDB_API_KEY, HF_TOKEN etc
│
├── configs/                         # yamls / dataclasses for experiments
│   ├── train_act.yaml
│   ├── train_diffusion.yaml
│   └── train_flow.yaml
│
├── scripts/                         # entrypoints, not importable library code
│   ├── train.py
│   ├── eval.py
│   ├── collect_data.py
│   └── visualize.py
│
└── src/
    └── savannah/                  # the actual importable package
        ├── __init__.py
        │
        ├── nn/                      # pure neural network building blocks
        │   ├── __init__.py
        │   ├── attention.py         # MultiHeadAttention, CrossAttention
        │   ├── transformer.py       # TransformerEncoder, TransformerDecoder
        │   ├── unet.py              # UNet, ResBlock, DownBlock, UpBlock
        │   └── lora.py              # LoRALinear, inject_lora()
        │
        ├── models/                  # full assembled models
        │   ├── __init__.py
        │   ├── act.py               # ACT — your existing implementation
        │   ├── diffusion.py         # Diffusion Policy
        │   ├── flow.py              # Flow Matching Policy
        │   ├── encoders.py          # SigLIPEncoder, ResNetEncoder
        │   └── base.py              # BasePolicy ABC that all models inherit
        │
        ├── data/                    # everything data
        │   ├── __init__.py
        │   ├── dataset.py           # LeRobotDataset wrapper, __getitem__
        │   ├── collector.py         # MultiCameraObsWrapper, collect_dataset()
        │   └── transforms.py        # image augmentations, normalization
        │
        ├── envs/                    # simulation environment utils
        │   ├── __init__.py
        │   ├── metaworld.py         # env factory, camera wrapper
        │   └── eval.py              # run_eval(), EvalResult dataclass
        │
        ├── training/                # training infrastructure
        │   ├── __init__.py
        │   ├── trainer.py           # Trainer class, train loop
        │   ├── losses.py            # CFM loss, DDPM loss, ACT loss
        │   └── scheduler.py         # LR schedulers, noise schedules
        │
        └── utils/                   # genuinely shared utilities
            ├── __init__.py
            ├── logging.py           # WandB init, log_metrics(), log_video()
            ├── io.py                # checkpoint save/load
            └── misc.py              # seed_everything(), count_params() etc
