### Nonlinear RNN for Text Generation

This program constructs a character-level sequence model to generate text according to a character distribution learned from the dataset. It implements linear, nonlinear and distributional different model architectures. The linear model uses character-level embeddings to form the model. The nonlinear model adds a parallel word level embedding network, which is merged with the character embedding model.

- Try my web app implementation at www.communicatemission.com/ml-projects#text_generation (linear model only)
- model files available at https://github.com/mvenouziou/Project-Text-Generation.
- See credits /attributions below

##### Python / Google Colab
This repository includes an ipnyb file to run the entire program through Google Colab. For clarity, I have also included the model paramaters and architecture as python files. (Their code is already included in the full ipnyb file.)


##### What's New?
*(These items are original in the sense that I personally have not seen them at the original time of coding. Citations are below for content I have seen elsewhere.)*

Visualization:

- Plotting confidence intervals for character predictions using utilizing Tensorflow Probability distribution layers.

Model Architectures:

- Experiments with: Nonlinear model architecture uses parallel RNN's with both word-level embeddings and character-level embeddings. 

- Experiments with: Tensorflow Probability layers to create a more interpretable probability distribution model. (Character-model only). The standard text generation algorithm outputs logits, which we view as a distribution from which to generate the next character. Here, we formalize this as outputing our model as a TF Probability Distribution, using probablistic weights in the Dense layer (instead of scalars) and trained via maximum likelihood. 

- Proper handling of GRU states for multiple stateful layers

- Easily switch between model architectures through 'Paramaters' class object. Includes file management for organizing each architecture's checkpoints.


Data Processing / Preparation:

*These ideas are not all new, but I have not seen them implemented in text generation systems:*

- Random crops and with random lengths and start locations. 

- Standard (character level) generation separates inputs and targets by one letter (generating one char at a time). This is adjusted for word-embeddings to avoid leaking target data, and multiple characters are generated at a time.

- Load and prepare data from multiple CSV and text files. Each rows from a CSV and each complete TXT file are treated as independent data sources. (CSV data prep accepts titles and content.) 


Generation:

- Add perturbations to learned probabilties in final generation function, to add extra variety to generated text.  (Included in addition to the 'temperature' control described in TF's documentation)        
    

    ---
### Credits / Citations / Attributions:

Linear Model and Shared Code

- Other than items noted in previous sections, this python code and linear model structure is based heavily on code found in Imperial College London's Coursera course "Customising your models with Tensorflow 2" *(https://www.coursera.org/learn/customising-models-tensorflow2)* and the Tensorflow RNN text generation documentation *(https://www.tensorflow.org/tutorials/text/text_generation?hl=en).*

Nonlinear Model:

This utilizes the pretrained embeddings:

- Small BERT word embeddings from Tensorflow Hub, (credited to Iulia Turc, Ming-Wei Chang, Kenton Lee, Kristina Toutanova's paper "Well-Read Students Learn Better: On the Importance of Pre-training Compact Models." https://tfhub.dev/google/collections/bert/1)*
- ELECTRA-Small++ from Tensorflow Hub, (credited to Kevin Clark and Minh-Thang Luong and Quoc V. Le and Christopher D. Manning's paper "ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators." https://hub.tensorflow.google.cn/google/electra_small/2)*

Datasets:

- 'robert_frost_collection.csv' is a Kaggle dataset available at https://www.kaggle.com/archanghosh/robert-frost-collection. Any other datasets used are public domain works available from Project Gutenberg https://www.gutenberg.org.



Web App

- The web app is built on the Anvil platform and (at the time of this writing) is hosted on Google Cloud server (CPU).


---
### About

Find me on LinkedIn: https://www.linkedin.com/in/movenouziou/ or GitHub: https://github.com/mvenouziou

---
