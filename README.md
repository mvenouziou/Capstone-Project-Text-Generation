### Nonlinear RNN for Text Generation

This program constructs a character-level sequence model to generate text according to a character distribution learned from the dataset. It implements two different model architectures: "linear" and "nonlinear." The linear model uses character-level embeddings to form the model. The nonlinear model adds a parallel word level embedding network, which is merged with the character embedding model.

- Try my web app implementation at www.communicatemission.com/ml-projects#text_generation (linear model only)
- model files available at https://raw.githubusercontent.com/mvenouziou/text_generator.
- See credits /attributions below


### What's New? 
*(Although very likely that others have created similar models, I personally have not seen them and these can be considered independent constructions. Citations for other content are below:)*

- Option to implement either the standard linear model architecture (see credits below) or nonlinear architectures.
- Nonlinear model architecture uses parallel RNN's for word-level embeddings and character-level embeddings.
- Manage RNN statefulness for independent data sources. The linear models credited below use a single continuous work, which necessarily implies a dependence relation between samples / batches. This model implements the ability to treat independent works (individual poems, books, authors, etc.) as truly independent samples by resetting RNN states and shuffling independent data sources.
- Load and prepare data from multiple CSV and text files. Each rows from a CSV and each complete TXT file are treated as independent data sources. (CSV data prep accepts titles and content.)
    
    ---
### Credits / Citations / Attributions:

Linear Model and Shared Code

- Other than items noted in previous sections, this python code and linear model structure is based heavily on code found in Imperial College London's Coursera course "Customising your models with Tensorflow 2" *(https://www.coursera.org/learn/customising-models-tensorflow2)* and the Tensorflow RNN text generation documentation *(https://www.tensorflow.org/tutorials/text/text_generation?hl=en).*

Nonlinear Model:

- This utilizes the pretrained Small BERT word embeddings from Tensorflow Hub, which they credit to Iulia Turc, Ming-Wei Chang, Kenton Lee, Kristina Toutanova's paper "Well-Read Students Learn Better: On the Importance of Pre-training Compact Models." *(See https://tfhub.dev/google/collections/bert/1)*


Web App

- The web app is built on the Anvil platform and (at the time of this writing) is hosted on Google Cloud server (CPU).


---
About
Find me on LinkedIn: https://www.linkedin.com/in/movenouziou/ or GitHub: https://github.com/mvenouziou
---
