# Bert_with_additional_SelfAttention
## Transformer-based classification model with fine-tuning a pretrained model and additional self attention layers

### (This repository is still under development!)

The finetuning for the pretrained model and upgating its gradients only continues until a specific epoch, then weights get frozen and only the embeddings are being extracted, and passed to the next layers. 

In case of using a Twitter data, use tweet_preprocessing.py first to clean and normalize the inputs.

The code will get updated soon to get the variables as input, such as at which epoch do you want to freeze the weights in finetuning, and what pretrained model do you want to use (bert, Roberta, etc.). Right now, just set the variables at the beginning of main function. 

The lines for the self-attention layers in model.py are adopted from [**this repository**](https://github.com/SamLynnEvans/Transformer).
