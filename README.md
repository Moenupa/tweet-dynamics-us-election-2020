# Sentiment Analysis

## Data Structure

1. Download the data from [kaggle us 2020 election tweets](https://www.kaggle.com/datasets/manchunhui/us-election-2020-tweets)
2. Structure your data like this:
    ```sh
    .
    ├── README.md
    └── data
        ├── predict.ipynb
        ├── emotion             # prediction of tweet's emotion
        ├── language            # prediction of tweet's language
        ├── sentiment           # prediction of tweet's sentiment
        ├── stance_trump        # prediction of tweet's stance for trump
        ├── stance_biden        # prediction of tweet's stance for biden
        └── src                 # <- kaggle original data
            ├── hashtag_donaldtrump.csv
            └── hashtag_joebiden.csv
    ```

## How to run

```py
from util import load_data
data = load_data("trump") # eithor "trump" or "biden"
```
   
