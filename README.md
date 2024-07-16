# RSVQA: Visual question answering for remote sensing data

This is a fork of the original repository, which can be found here: [RSVQA](https://github.com/syvlo/RSVQA)

Adjustments have been made to import the HR dataset and start training.

To set-up the repo for training, follow the steps below:

Download the HR dataset from [zenodo](https://zenodo.org/record/6344367) and place the JSON files in the `data/text` folder.
Extract and move the images to the `data/images` folder.

Install pytorch with the following command (venv recommended):

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 # or cu118
```

Install the remaining dependencies:

```bash
pip3 install -r requirements.txt
```
    

# RSVQA Code

## Preprocessing
We include several jupyter notebooks to precompute the embeddings. These can be used to precompute the embeddings for the HR dataset. The notebooks are located in the `perprocessing` folder. For the models using BERT and ViT, `VQA_model\preprocessing\text_preprocessing-bert-attention.ipynb` and `VQA_model\preprocessing\text_preprocessing-vit-attention.ipynb` can be used respectively. Other variations included are for CLS only embeddings.

For more information on the skip-thoughts RNN model used previously check out the upstream repository.

## Training and architecture
The architecture is defined in models/model.py, and you can use train.py to launch a training.
For the multitask model, you can use the train_multitask.py script. The variation with attention is defined in models/multitask_attention.py.
A different variation of the multitask model is defined in models/multitask.py, where everything is shared up to the fusion layer.

When instantianting the model, parameters can be passed to specify the output dimensionality of the feature extractors (e.g. if using a bigger feature extractor), as well as the number of neurons in the prediction layers. Defaults are set for vit-base and bert-base-uncased. The type of fusion used can be set directly in the model file.

We use wandb for logging training runs and save a checkpoint of the model at the end of each epoch. The checkpoints are saved in the `outputs` folder.
We include preliminary code for also training the feature extractors, which is available on the branch `ft-multitask`.

## Evaluation

The evaluation is done using the `computeStats.py` script. You can specify the model to use and the checkpoint to load. The script will load the checkpoint and evaluate the model on the test set. It includes accuracy metrics as well as confusion matrices and a distribution plot for the counting task.


# References
If you use this code, please cite the original paper, which it is based on.

[1] Lobry, Sylvain, et al. "RSVQA: Visual question answering for remote sensing data." IEEE Transactions on Geoscience and Remote Sensing 58.12 (2020): 8555-8566.
