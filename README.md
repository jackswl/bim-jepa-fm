# Toward generalizable foundation models for 3D BIM geometry using a joint embedding predictive architecture

Created by [Jack Wei Lun Shi](https://jackswl.github.io/)\*, [Wawan Solihin](https://cde.nus.edu.sg/cee/staff/wawan-solihin/), Yufeng Weng, [Houhao Liang](https://scholar.google.com/citations?user=Fl5X79UAAAAJ&hl=en), [Yimin Zhao](https://ztony0712.github.io/), [Leong Hien Poh](https://scholar.google.com/citations?user=xZ3x56EAAAAJ&hl=en), [Justin Ker-Wei Yeoh](https://scholar.google.com/citations?user=m9LF49sAAAAJ&hl=en)

[[Automation in Construction]](https://doi.org/10.1016/j.autcon.2026.107169) [[Project Page]](https://jackswl.github.io/bim-jepa-fm/) [[Model Weights]](https://huggingface.co/collections/llama2thedog/bim-jepa-fm)

This repository contains BIM-JEPA implementation for __Toward generalizable foundation models for 3D BIM geometry using a joint embedding predictive architecture__, published in Automation in Construction.

BIM-JEPA is a point cloud-based foundation model for 3D Building Information Modeling (BIM) geometry, pre-trained via a Latent-Euclidean Joint Embedding Predictive Architecture on individual BIM objects. By enforcing alignment across multi-view BIM object representations within a regularized latent space, BIM-JEPA extracts robust semantic features while suppressing low-level geometric noise. The learned representations generalize across multiple downstream tasks, including standard and fine-grained object classification, semantic segmentation via transfer learning, out-of-distribution part segmentation of computer-aided design objects, and zero-shot tasks such as shape retrieval and anomaly detection.

## <a id="pretrained-models"></a>Pre-trained Model

This model is pre-trained on over 2.1 million BIM elements; our earlier [BIM-JEPA](https://github.com/jackswl/bim-jepa) release was pre-trained on roughly 900k.

|model| dataset | config | url|
| :---: | :---: | :---: |  :---: |
|BIM-JEPA-pretrained| BIMCompNet; IFC-884K; IFCNet; BIMGEOM | [`combined_pretrain_lejepa2.yaml`](BIM-JEPA-FM/configs/BIM-JEPA-FM/pretraining/combined_pretrain_lejepa2.yaml) | [🤗 llama2thedog/BIM-JEPA-FM](https://huggingface.co/llama2thedog/BIM-JEPA-FM) |

Download the checkpoint with:
```python
from huggingface_hub import hf_hub_download
hf_hub_download("llama2thedog/BIM-JEPA-FM", "last.ckpt", local_dir="models")
```
or from the command line:
```
hf download llama2thedog/BIM-JEPA-FM last.ckpt --local-dir models
```

## Repository structure

```
BIM-JEPA-FM/
├── pointjepa/                     # model, modules, datamodules, tasks
│   ├── models/point_lejepa.py     # LeJEPA pre-training model
│   ├── models/classification.py
│   └── models/part_segmentation.py
└── configs/BIM-JEPA-FM/
    ├── pretraining/               # combined_pretrain_lejepa2.yaml
    ├── classification/            # ifcnet / bimgeom / bimcompnet_100
    └── segmentation/              # shapenetpart (OOD)
```

## Usage

### Requirements
- PyTorch >= 2.4.1
- python == 3.11
- CUDA >= 12.1
- torchvision
- PyTorch3D

### Conda Installation
Option A (Recommended) -> You can create conda environment using:
```
conda env create -f environment.yml
conda activate bimjepafm
```

Option B -> Create environment:
```
conda create -n bimjepafm \
    python=3.11 \
    pytorch=2.4.1 \
    torchvision=0.19.1 \
    pytorch-cuda=12.1 \
    cudatoolkit \
    -c pytorch -c nvidia -y
```
After that, install PyTorch3D (https://github.com/facebookresearch/pytorch3d):
```
export FORCE_CUDA=1
conda install pytorch3d::pytorch3d
```
Finally, install the remaining miscellaneous/visualization packages:
```
pip install transformers accelerate "pytorch-lightning>=2.0" "jsonargparse[signatures]" \
    hydra-core omegaconf trimesh scikit-learn h5py matplotlib wandb timm \
    lightning-bolts lightning-utilities fvcore yacs pandas seaborn plotly \
    huggingface-hub tokenizers torchmetrics pydantic remotezip
```

### Dataset
We use multiple BIM/CAD datasets — see [DATASET.md](DATASET.md) for the full list (IFC-884K, IFCNet, IFCNetCore, BIMGEOM, BIMCompNet, BIMObject, BIMNet, ShapeNetPart) with download links and the expected directory layout. Place the raw data under `data/` (see `data/info.txt`), then run `data_convert.ipynb` to pre-process everything into `.npy` point clouds.

If you do not want the full BIMCompNet release (it is very large), `data/bimcompnet_extraction_scripts/` contains PBS jobs that stream-download just the `.obj` meshes you need (extract → quarantine → obj_to_npy).

### Training

Each config contains `/path/to/...` placeholders for the dataset directories and, for the
downstream tasks, `pretrained_ckpt_path` for the pre-trained checkpoint. Edit these to match
your setup before launching. All commands are run from the `BIM-JEPA-FM/` directory:

```
cd BIM-JEPA-FM
```

Pre-training:
```
python -m pointjepa fit -c configs/BIM-JEPA-FM/pretraining/combined_pretrain_lejepa2.yaml
```

Classification:
```
python -m pointjepa.tasks.classification fit -c configs/BIM-JEPA-FM/classification/ifcnet_classification.yaml
python -m pointjepa.tasks.classification fit -c configs/BIM-JEPA-FM/classification/bimgeom_classification.yaml
python -m pointjepa.tasks.classification fit -c configs/BIM-JEPA-FM/classification/bimcompnet_classification_100.yaml
```

Part segmentation (out-of-distribution):
```
python -m pointjepa.tasks.part_segmentation fit -c configs/BIM-JEPA-FM/segmentation/shapenetpart.yaml
```

Replace `fit` with `test` and pass `--ckpt_path <trained.ckpt>` to evaluate a trained model.

The pre-training config is set up for 4 GPUs (`trainer.devices: 4`, DDP, bf16-mixed); adjust
`trainer.devices` and `data.init_args.batch_size` for your hardware.


## License
MIT License

## Citation
If you find our work useful in your research, please consider citing: 
```
@article{shi2026toward,
  title={Toward generalizable foundation models for 3D BIM geometry using a joint embedding predictive architecture},
  author={Shi, Jack Wei Lun and Solihin, Wawan and Weng, Yufeng and Liang, Houhao and Zhao, Yimin and Poh, Leong Hien and Yeoh, Justin K.W.},
  journal={Automation in Construction},
  volume={191},
  pages={107169},
  year={2026},
  publisher={Elsevier}
}
```

## Acknowledgements
```
We sincerely thank the authors of LeJEPA, Point-JEPA, SpaRSE-BIM/IFCNet, BIMGEOM, BIMCompNet, BIMNet, for making their code/data and models publicly available, which served as the foundation for this work. If you use our work, please also consider citing these papers.
```