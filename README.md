# USC SAIL - MediaEval 2020
MediaEval 2020: Emotions and Themes in Music

This repo is based on the USC SAIL submission for [MediaEval 2020](https://multimediaeval.github.io/editions/2020/tasks/music/), but it is designed to be easy to setup and evaluate for general music tagging problems. Our ensemble model won the [MediaEval 2020 challenge](https://multimediaeval.github.io/2020-Emotion-and-Theme-Recognition-in-Music-Task/results).

## Requirements

Python 3.7 was used for this implementation.

```
pip install -r requirements.txt
```

## Usage

Given a directory `audio_dir` of audio files (e.g., the mood/theme [split](https://github.com/MTG/mtg-jamendo-dataset) from MTG-Jamendo) and an output folder, the following gives a brief usage example for training and evaluating a music tagging model:

1. Downsample and convert all audio files to npy:
```
python -u resample2npy.py --data_path audio_dir \
			  --output_path /path/to/npy_audios/
```

2. Create a .tsv file with music tag labels, as specified in *data_loader.py*. Such tsv examples can be found under `example_splits/`. Dataloader bahavior can be modified in the `read_file` function. Additionally, the tags label set (currently set to mood/theme tags for the challenge) can be modified via the *TAGS* list in *data_loader.py*.

3. Run *train.py*. The following example uses BCE loss and [mixup](https://arxiv.org/pdf/1710.09412). A pretrained model can be loaded by setting `model_load_path` to point to a pytorch state_dict, such as *best_model.py*, pretrained on [MSD](https://arxiv.org/abs/2006.00751). Note that only weights from layers with matching names will be loaded.

```
python -u main.py --data_path /path/to/npy_audios/ \
		  --splits_path /path/to/splits_tsv/ \
		  --model_save_path /path/to/outputs/ \
		  --use_mixup 1 \
		  --loss_function bce \
		  --sampling_type standard
```

4. Run *eval.py* to evaluate a trained model on given test split:

```
python -u eval.py --data_path /path/to/npy_data/ \
		  --splits_path /path/to/splits_tsvs/ \
		  --model_load_path /path/to/model/best_model.pth \
		  --use_val_split 0 \
		  --save_predictions 1 \
		  --save_path /output/path/
```

## Loss functions

We provide multiclass, multilabel implementations for the following loss functions (see *losses.py*). They can be specified via the `loss_function` argument in *main.py*.

- [BCE loss](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html) - argument: `bce`
- [Focal loss](https://arxiv.org/abs/1708.02002) - argument: `focal_loss`
- [Class-balanced loss](https://arxiv.org/abs/1901.05555) - argument: `cb_focal_loss`
- [Distribution-balanced loss](https://arxiv.org/abs/2007.09654) - argument: `db_focal_loss`

Additionally, we provide implementations for *mixup* (set `use_mixup` flag to 1), as well as class-aware resampling, specified by setting `sampling_type` to *class_aware*. 

## Model

Our model is a modified version of the [Short-Chunk CNN with Residual Connections](https://arxiv.org/abs/2006.00751) by Won et al. The low-level CNN layers were pretrained on MSD. We used the training, validation, and test splits as specified by the challenge. We combined our training set with instances with matching tags from the [Music4All](https://ieeexplore.ieee.org/document/9145170) dataset.

## Citation

If you find our approach interesting for your research, please consider citing our [paper](http://ceur-ws.org/Vol-2882/paper67.pdf):
```
@inproceedings{knox2020mediaeval,
  title={MediaEval 2020 Emotion and Theme Recognition in Music task: Loss Function Approaches for Multi-label Music Tagging},
  author={Knox, Dillon and Greer, Timothy and Ma, Benjamin and Kuo, Emily and Somandepalli, Krishna and Narayanan, Shrikanth},
  booktitle={CEUR Workshop Proceedings},
  year={2020}
}
```