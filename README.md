# handSignalPrediction
hand signal prediction project for Dacon

## Classification Approach
|Model         |log_loss|option|
|:------------:|:------:|:----:|
|EfficientNetB2|0.5087  |      |
|EfficientNetB7| 0.1006 |    None augmentor, acc = 0.9658  |
|EfficientNetB7|0.5087  |    None augmentor, valid_set  |
|Resnet50v2    |0.2281  |      |
|Slow-Fast     |1.09    |      |
