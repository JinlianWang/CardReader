CardReader Data Processor. This repo contains a few python scripts to create training and testing dataset for card reader. All the images (cards and backgrounds) are downloaded from internet; their copyright belongs to the original author and shall be used for demo only. 

* Process the card images: 

```
python data_utils/preprocess.py cardsOrig/ cardsProcessed/ --create-label-file=1
```

* Process the background images: 

```
python data_utils/preprocess.py backsOrig/ backsProcessed/ --is-background=1
```

* Generate the training and testing data: 

```
python generate_data.py cardsProcessed/ backsProcessed/
```
