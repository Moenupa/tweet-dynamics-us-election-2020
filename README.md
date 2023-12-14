# Exploring Public Opinion Dynamics From 2020 U.S. Election Tweets

## Abstract 

This study delves into the dynamics of public opinion during the 2020 U.S. Presidential Election through sentiment analysis, emotion analysis, and stance detection applied to around 1.72M tweets. We explore the predictive power of sentiment analysis in political science, detailing the transition from traditional to advanced NLP models, notably XLM-RoBERTa, for enhanced Twitter data anal- ysis. Additionally, we examine stance detection's role in reflecting public attitude and potential voting behavior. The study investigates how sentiment and stance correlate with major election events, presenting a temporal and geographical analysis of the data. The goal is to ascertain the impact of social media on election outcomes and public opinion.

## Inference

1. Download the data from [kaggle us 2020 election tweets](https://www.kaggle.com/datasets/manchunhui/us-election-2020-tweets)
2. Structure your data like this, or run `inference.ipynb` to generate the inferences:
    ```sh
    .
    ├── README.md
    └── data
        ├── inference.ipynb
        ├── emotion             # inference of tweet's emotion
        ├── language            # inference of tweet's language
        ├── sentiment           # inference of tweet's sentiment
        ├── stance_trump        # inference of tweet's stance for trump
        ├── stance_biden        # inference of tweet's stance for biden
        └── src                 # <- kaggle original data
            ├── hashtag_donaldtrump.csv
            └── hashtag_joebiden.csv
    ```

## Figures

See [figure_generator.py](figure_generator.py). You can obtain a compiled version from [github@Moenupa/tweet-dynamics-us-election-2020](https://github.com/Moenupa/tweet-dynamics-us-election-2020)
   
