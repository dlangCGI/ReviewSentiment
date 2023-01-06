# Review Sentiment Analysis

## Motivation
This is an End-to-End ML project for Review Sentiment Analysis.

In forum environments, review classification can be used to automatically filter and rate
comments and thus providing feedback to the company.

For the pretrained Model here a domain-specific dataset was used (Amazon Phone Reviews), 
but the model can be fine-tuned or retrained on another dataset. The steps and evaluation 
for that are provided in the Jupyter Notebook.

## Setup
Python 3.9 is used for the project.

All other requirements are in *requirements.txt* file, so just perform: 

```
pip install -r requirements.txt
```

Using conda also works for most packages, but the *symspellpy* dependency needs pip install.
symspellpi also needs buildtools like gcc to be successfully installed.
On linux you could use:

```apt-get update && apt-get install -y build-essential```

- Run the *app.py* file inside the *api* package to test the model through a simple RESTApi.

- Inside the *tests* package are two test script to test the model with unittests.

## Docker
You can also use the Dockerfile to directly build an image with all dependencies and use the web app.

To build the image:

```docker build -t <imagename>:<tag> <rootdirectory>```

After that just run your image as a container:

```docker run -d -p <localport>:5000 <imagename>:<tag>```

Then you can connect with the web app via ```localhost:<localport>``` in your browser.


