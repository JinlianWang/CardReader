CardReader Data Processor

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
