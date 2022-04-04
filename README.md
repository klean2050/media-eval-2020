# USC SAIL - MediaEval 2020
MediaEval 2020: Emotions and Themes in Music

This repo is based on the USC SAIL submission for [MediaEval 2020](https://multimediaeval.github.io/editions/2020/tasks/music/), but it is designed to be easy to setup and evaluate for general music tagging problems. Our ensemble model won the [MediaEval 2020 challenge](https://multimediaeval.github.io/2020-Emotion-and-Theme-Recognition-in-Music-Task/results).

## Requirements

Python >= 3.7

```
pip install -r requirements.txt
```

## Usage

Given a directory of .mp3 files (such as the mood/theme split from MTG-Jamendo; https://github.com/MTG/mtg-jamendo-dataset), the following gives a brief usage example for training and evaluating a music tagging model:

1. First run *resample2npy.py* to resample all mp3 files in the given directory to 16 kHz.
	* `python -u resample2npy.py run /path/to/mp3s/`
2. Create a .tsv file with music tag labels, as specified in *data_loader.py*. 
	- See *example_splits* for examples of the structure of tsv files that *data_loader.py* is currently set up to parse. This bahavior can be modified in `read_file` function.
	- Additionally, the tags label set (currently set to mood/theme tags for the challenge) can be modified via the *TAGS* list in *data_loader.py*.
3. Run *train.py*. The following example uses binary cross-entropy as the loss function, and also uses mixup (https://arxiv.org/pdf/1710.09412).
	* `python -u main.py --data_path /path/to/npy_data/ --splits_path /path/to/splits_tsvs/ --model_save_path /output/path/ --use_mixup 1 --loss_function bce --sampling_type standard` 
		- A pretrained model can also be loaded by setting `model_load_path` to point to a pytorch state dict, such as *best_model.py* (pretrained on Million Song Dataset, please see Won et al. https://arxiv.org/abs/2006.00751 for further details). Note that only weights from layers with matching names will be loaded.
4. Run *eval.py*. The following script will evaluate a trained model on the given test split.
	* `python -u eval.py --data_path /path/to/npy_data/ --splits_path /path/to/splits_tsvs/ --model_load_path /path/to/model/best_model.pth --use_val_split 0 --save_predictions 1 --save_path /output/path/`
		- Setting `use_val_split` to 1 instead evaluates the model on the validation set in the `get_dataset` function.

## Loss functions

In this repository, we provide multiclass, multilabel implementations for the following loss functions (see *losses.py*):

- Focal loss (https://arxiv.org/abs/1708.02002)
- Class-balanced loss (https://arxiv.org/abs/1901.05555)
- Distribution-balanced loss (https://arxiv.org/abs/2007.09654)

Loss functions are defined in *losses.py*. They can be specified as follows via the `loss_function` argument in *main.py*.
- Binary cross-entropy: `bce`
- Focal loss: `focal_loss`
- Class-balanced focal loss: `cb_focal_loss`
- Distribution-balanced focal loss: `db_focal_loss`

Additionally, we provide implementations for *mixup*, which can be used by setting the `use_mixup` flag to 1, as well as class-aware resampling, specified by setting `sampling_type` to *class_aware*. 

## Model

Our model is modified from the Short-Chunk CNN with Residual Connections presented by Won et al. (https://arxiv.org/abs/2006.00751)

## Dataset

The low-level feature extract layers for the CNN were pretrained on the Million Song Dataset (http://millionsongdataset.com/). See our Working Notes paper for implementation details.

We used the training, validation, and test splits as specified by the challenge. We combined our training set with instances with matching tags from the Music4All dataset (https://ieeexplore.ieee.org/document/9145170).
