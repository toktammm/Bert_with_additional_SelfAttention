# Bert_with_additional_SelfAttention
# Transformer-based classification model with fine-tuning a pretrained model and additional self attention layers

## The finetuning for the pretrained model and upgating its gradients only continues until a specific empoch, then weights get frozen and only the embeddings are being passed to the next layers. 
## The code will get updated soon to get some variables as input, such as which epoch do you want to freeze the weights in finetuning. Right now, just set the variables at the beginning of main function. 

The lines for the self-attention layers in model.py are adopted from [**this repository**](https://github.com/SamLynnEvans/Transformer).
