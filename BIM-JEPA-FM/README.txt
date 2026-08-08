This BIM-JEPA-FM repo is a direct modification of the Point-JEPA repo
(https://github.com/Ayumu-J-S/Point-JEPA), with the pre-training objective
replaced by LeJEPA (https://arxiv.org/abs/2511.08544).

The pre-training model is pointjepa/models/point_lejepa.py (class PointLeJepa):
multi-view global/local crops with a SIGReg isotropic-Gaussian regulariser, and
no EMA teacher. Downstream classification and part-segmentation models load the
resulting encoder via pointjepa/utils/checkpoint.py.
