# Language Conditioned Spatial Relation Reasoning for 3D Object Grounding

This repository is the official implementation of [Language Conditioned Spatial Relation Reasoning for 3D Object Grounding](https://arxiv.org/abs/2211.09646). 
Project webpage: https://cshizhe.github.io/projects/vil3dref.html

Localizing objects in 3D scenes based on natural language requires understanding and reasoning about spatial relations. In particular, it is often crucial to distinguish similar objects referred by the text, such as "the left most chair" and "a chair next to the window". In this work we propose a language-conditioned transformer model for grounding 3D objects and their spatial relations. To this end, we design a spatial self-attention layer that accounts for relative distances and orientations between objects in input 3D point clouds. Training such a layer with visual and language inputs enables to disambiguate spatial relations and to localize objects referred by the text. To facilitate the cross-modal learning of relations, we further propose a teacher-student approach where the teacher model is first trained using ground-truth object labels, and then helps to train a student model using point cloud inputs. We perform ablation studies showing advantages of our approach. We also demonstrate our model to significantly outperform the state of the art on the challenging Nr3D, Sr3D and ScanRefer 3D object grounding datasets.


## Installation
1. Follow the [instructions](https://github.com/zyang-ur/SAT#prerequisites) to build the environment.
```
conda create -n vil3dref python=3.8
conda activate vil3dref

conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt

# To use a PointNet++ visual-encoder you need to compile its CUDA layers for PointNet++: Note: To do this compilation also need: gcc5.4 or later.
cd model/external_tools/pointnet2
python setup.py install
```

2. Download [preprocessed data](https://www.dropbox.com/s/n0m5bpfvea1fg7w/referit3d.tar.gz?dl=0) and [pretrained models](https://www.dropbox.com/s/5zh8wgt7x2iqq76/exprs_neurips22.tar.gz?dl=0). Put the data in `datasets' directory.
(Some preprocessing codes are in the preprocess/ directory).


## Training
1. train the teacher model with groundtruth object labels
```
cd og3d_src
# change the configfile for other datasets, e.g., scanrefer_gtlabel_model.yaml
configfile=configs/nr3d_gtlabel_model.yaml
python train.py --config $configfile --output_dir ../datasets/exprs_neurips22/gtlabels/nr3d
```

2. train the pointnet encoder
```
python train_pcd_backbone.py --config configs/pcd_classifier.yaml \
    --output_dir ../datasets/exprs_neurips22/pcd_clf_pre
```

3. train the student model with 3d point clouds
```
# change the configfile for other datasets
configfile=configs/nr3d_gtlabelpcd_mix_model.yaml
python train_mix.py --config $configfile \
    --output_dir ../datasets/exprs_neurips22/gtlabelpcd_mix/nr3d \
    --resume_files ../datasets/exprs_neurips22/pcd_clf_pre/ckpts/model_epoch_100.pt \
    ../datasets/exprs_neurips22/gtlabels/nr3d/ckpts/model_epoch_49.pt
```

## Inference
```
configfile=configs/nr3d_gtlabelpcd_mix_model.yaml
python train_mix.py --config $configfile \
    --output_dir ../datasets/exprs_neurips22/gtlabelpcd_mix/nr3d \
    --resume_files ../datasets/exprs_neurips22/gtlabelpcd_mix/nr3d/ckpts/model_epoch_96.pt \
    --test
```

## Citation
If you find this work useful, please consider citing:
```
@InProceedings{chen2022vil3dref,
author       = {Chen, Shizhe and Tapaswi, Makarand and Guhur, Pierre-Louis and Schmid, Cordelia and Laptev, Ivan},
title        = {Language Conditioned Spatial Relation Reasoning for 3D Object Grounding},
booktitle    = {NeurIPS},
year         = {2022},
}
```

## Acknowledgement
Some of the codes are built upon [ReferIt3D](https://github.com/referit3d/referit3d) and [SAT](https://github.com/zyang-ur/SAT).
Thanks them for their great works!
