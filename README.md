# Review Sentiment Analysis

## Motivation
This is an End-to-End ML project for Review Sentiment Analysis.

In forum environments, review classification can be used to automatically filter and rate
comments and thus providing feedback to the company.

For the pretrained Model here a domain-specific dataset was used (Amazon Phone Reviews), 
but the model can be fine-tuned or retrained on another dataset. The steps and evaluation 
for that are provided in the Jupyter Notebook.

## Setup

All requirements are in *requirements.txt* file, so just perform: 

```
pip install -r requirements.txt
```

Using conda also works for most packages, but the *symspellpy* dependency needs pip install.

- Run the *app.py* file inside the *api* package to test the model through a simple RESTApi.

- Inside the *tests* package are two test script to test the model with unittests.



