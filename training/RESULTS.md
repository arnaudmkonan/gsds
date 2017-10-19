# Model Training Results 

Explore models based on data with original labels and with the data and corrected labels.


1. Original Labels

80:20 training-validation split

| Model | Hyperparameters                | Epoch | Train Accuracy | Validation Accuracy | Comments |
|:------|:-------------------------------|:------|:---------------|:--------------------|:-------|
| Resnet18 ft 2 layers | bs=256, lr=0.001 lrs=2 | 5/12  |  76.8          |   79.1    | 1 |
| Resnet18 ft 2 layers | bs=256, lr=0.005 lrs=2 | 3/12  |  66.7          |   73.8    | 1 |
| Resnet18 ft 2 layers | bs=256, lr=0.001 lrs=3 | 5/12  |  78.4          |   80.1    | 1 |
| Resnet18 ft 2 layers | bs=256, lr=0.001 lrs=4 | 3/12  |  77.9          |   80.3    | 1 |
| Resnet18 retrain 1 ly| bs=32, lr=0.001 lrs=4 | 5/12  |  92.6          |   80.4    | 2 |
| Resnet18 retrain 2 ly| bs=32, lr=0.001 lrs=4 l2=0.001 | 5/12  |  89.9    |   82.4    | 3 |
| Resnet18 retrain 2 ly| bs=32, lr=0.001 lrs=4 l2=0.01  | 5/12  |  84.5    |   82.8    | 4  |
| Resnet18 retrain 2 ly| bs=32, lr=0.001 lrs=4 l2=0.01  | 5/12  |  84.5    |   82.8    | 4  |
| 100  Resnet18 retrain 2 ly| bs=32, lr=0.001 lrs=4 l2=0.01  | NA  |  NA   |   <81.8>  | 5  |


Comments

1. underfitting 
2. overfitting, need to reduce batch size to fit in memory
3. still overfitting but less due to L2
4. pretty close, but accuracy is low..need to clean labels
5. Run this model 100 times with different seeds, mean accuracy in <>


| Abbreviation | Definition|
|:-------------|:----------|
| bs           | batch size |
| lr           | learning rate |
| lrs    | learning rate decrease step frequency |
| ly  | layer|
| l2  | L2 regularization |


1. Corrected Labels

80:20 dataset 100 model averages

    `Data set sizes: {'train': 2131, 'val': 531, 'test': 100}`

90:10 dataset 100 model averages

    `Data set sizes: {'train': 2397, 'val': 265, 'test': 100}`

| Model | Hyperparameters                | Epoch | Test Accuracy | Validation Accuracy | Comments |
|:------|:-------------------------------|:------|:---------------|:--------------------|:-------|
| 100 80-20  Resnet18 retrain 2 ly| bs=32, lr=0.001 lrs=4 l2=0.01  | NA  | <90.6>  |   <87.8>  | 6  |
| 100 90-10 Resnet18 retrain 2 ly| bs=32, lr=0.001 lrs=4 l2=0.01  | NA  |  <89.0>  |   <90.0>  | 7  |

Comments

6. Test set size is only 100 images. validation accuracy may be lower than 90-10 result simply since more data in this validation set. 
7. The exact same test set is used. 80-20 split thus seems to be a better strategy as you pick models based on more validation experience.





