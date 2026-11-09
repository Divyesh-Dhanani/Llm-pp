## split_by_family

| family       |   train |   val |   test |
|:-------------|--------:|------:|-------:|
| cnn          |    3736 |     0 |      0 |
| detector     |       0 |   145 |      0 |
| other        |       0 |     0 |    657 |
| rnn          |       0 |   201 |      0 |
| segmentation |    1356 |     0 |      0 |
| transformer  |       0 |   254 |      0 |

## split_by_dataset

| dataset       |   train |   val |   test |
|:--------------|--------:|------:|-------:|
| celeba-gender |     347 |     5 |     55 |
| cifar-10      |    1385 |    53 |     73 |
| cifar-100     |     328 |    13 |     53 |
| coco          |    1506 |   366 |    133 |
| denoise       |       1 |     0 |      0 |
| imagenette    |     533 |    30 |     44 |
| mnist         |     368 |    24 |     71 |
| places365     |      70 |     3 |     15 |
| svhn          |     554 |     7 |     70 |
| wikitext      |       0 |    99 |    143 |

## split_by_task

| task               |   train |   val |   test |
|:-------------------|--------:|------:|-------:|
| img-captioning     |     198 |   217 |    133 |
| img-classification |    3585 |   135 |    381 |
| img-denoising      |       1 |     0 |      0 |
| img-segmentation   |    1206 |     0 |      0 |
| obj-detection      |       0 |   139 |      0 |
| txt-generation     |       0 |    99 |    143 |
| txt-image          |     102 |    10 |      0 |

## unique_architectures

| split   |   unique nn |
|:--------|------------:|
| train   |         976 |
| val     |          53 |
| test    |          15 |

## proxy_coverage_pct

| split   |   proxy_synflow |   proxy_nwot |   proxy_grad_norm |   proxy_log_params |   proxy_depth |
|:--------|----------------:|-------------:|------------------:|-------------------:|--------------:|
| train   |            92.2 |         97.8 |              67.8 |               99.3 |          99.3 |
| val     |            24.3 |         59.3 |              22.5 |              100   |         100   |
| test    |            51.8 |         42.6 |              39.3 |              100   |         100   |

## labels

| split   |   ('best_accuracy', 'mean') |   ('best_accuracy', 'median') |   ('best_accuracy', 'std') |   ('epochs_to_best', 'mean') |   ('epochs_to_best', 'median') |   ('epochs_to_best', 'std') |   ('total_epochs', 'mean') |   ('total_epochs', 'median') |   ('total_epochs', 'std') |
|:--------|----------------------------:|------------------------------:|---------------------------:|-----------------------------:|-------------------------------:|----------------------------:|---------------------------:|-----------------------------:|--------------------------:|
| train   |                       0.548 |                         0.57  |                      0.292 |                       40.044 |                             45 |                      16.574 |                     51.718 |                           50 |                    12.766 |
| val     |                       0.325 |                         0.208 |                      0.3   |                       31.983 |                             41 |                      19.83  |                     50.667 |                           50 |                     5.74  |
| test    |                       0.546 |                         0.584 |                      0.309 |                       39.957 |                             48 |                      14.011 |                     50     |                           50 |                     0     |

