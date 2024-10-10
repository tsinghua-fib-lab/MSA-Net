# MSA-Net
A python implementation of **MSA-Net: A multi-scale information diffusion model awaring user activity level**
## Data
We provide two processed datasets of Higgs and Weibo together with their processing codes. The relevant data and code can be found in floader ``data``
## Our Model
An Pytorch implementation of ``MSA-Net`` can be found in the folder ``train``.
**Hyper_params**
`history_window`: length of the history input.
`pred_window`: prediction length.
`slide_step`: length of slide window when constructing the dataset.
`input_size`: input dim of the vector.
`hidden_size`: hidden dim of LSTM and GCN.
## How to run our model
You can simply use ``python train/train.py`` to run our model.
## Requirements
You can run our model using torch==1.13.1. Other details can be found in `requirements.txt`