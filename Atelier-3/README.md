# Lab 3: NLP with Sequence Models and Transformers

**Course:** Deep Learning
**Department:** Département Génie Informatique
**University:** Université Abdelmalek Essaadi, Tangier
**Instructor:** Pr. ELAACHAK LOTFI

## Objective

The main purpose of this lab is to gain familiarity with PyTorch and build deep neural network architectures for Natural Language Processing tasks using Sequence Models (RNNs, GRUs, LSTMs) and Transformers (GPT-2).

## Project Structure

This project is divided into two main parts:

1.  **Part 1: Classification Task** - Building sequence models (specifically an LSTM in the provided notebook) for a text relevance classification (regression) task on Arabic text.
2.  **Part 2: Transformer (Text Generation)** - Fine-tuning a pre-trained GPT-2 model on a custom dataset for Arabic text generation.

The project code is contained within the `Atelier_3.ipynb` Jupyter notebook.

## Part 1: Classification Task

### Task Description

The goal is to classify Arabic text based on a relevance score (between 0 and 10). This involves collecting text data, preprocessing it, training various sequence models (RNN, Bi-RNN, GRU, LSTM), and evaluating their performance.

### Implementation Details

1.  **Data Collection (Step 1 in Notebook):**
    *   Used `requests` and `BeautifulSoup` to scrape article titles from the Al Jazeera news website (`https://www.aljazeera.net/news/`).
    *   Created a Pandas DataFrame with 'Text' (scraped titles) and a 'Score' column assigned with *random* values between 0 and 10 using `random.uniform`.
    *   *Note: The lab instructions requested data on a single topic; the current implementation scrapes recent news headlines, which cover various topics. The relevance scores are random, not based on actual content analysis.*

2.  **Preprocessing Pipeline (Step 2 in Notebook):**
    *   Used `nltk` for tokenization (`word_tokenize`).
    *   Removed non-Arabic characters and punctuation using regular expressions.
    *   Removed common Arabic stop words using `nltk.corpus.stopwords`.
    *   *Note: The lab instructions mentioned stemming, lemmatization, and discretization, which were not explicitly implemented in the provided notebook's `preprocess` function.*
    *   Encoded tokens into numerical IDs based on a vocabulary built from the collected data.
    *   Padded sequences to a fixed maximum length (20).

3.  **Model Training (Step 3 in Notebook):**
    *   Implemented an LSTM model (`LSTMModel`) using PyTorch (`torch.nn`).
    *   The model consists of an embedding layer, an LSTM layer, and a linear output layer for the regression score.
    *   Used `torch.utils.data.Dataset` and `DataLoader` to handle the data during training.
    *   Split the data into training and testing sets using `sklearn.model_selection.train_test_split`.
    *   Trained the LSTM model using Mean Squared Error (MSE) loss and the Adam optimizer for 10 epochs.
    *   *Note: The lab instructions requested training RNN, Bi-RNN, and GRU models as well, but only the LSTM model implementation and training loop are present in the notebook.*

4.  **Model Evaluation (Step 4 in Notebook):**
    *   Evaluated the trained LSTM model on the test set.
    *   Used `sklearn.metrics` to calculate Mean Squared Error (MSE) and R² score (`r2_score`).
    *   *Note: The lab instructions mentioned using metrics like the Blue score. Blue score is typically used for text generation evaluation (measuring overlap with reference translations/texts), not for regression/classification tasks like this relevance scoring. MSE and R² are appropriate for regression.*

### Results (Part 1)

The notebook output shows the training loss decreasing over epochs. The final evaluation metrics on the test set are printed:

*   **MSE:** [Output value from notebook]
*   **R² Score:** [Output value from notebook]
    *   *Note: A negative R² score indicates that the model performs worse than simply predicting the mean of the target variable.*

## Part 2: Transformer (Text Generation)

### Task Description

This part focuses on leveraging the power of the Transformer architecture, specifically by fine-tuning a pre-trained GPT-2 model for text generation in Arabic.

### Implementation Details

1.  **Load Pretrained GPT-2 (Step 1 in Notebook):**
    *   Installed the `transformers` library from Hugging Face.
    *   Loaded the `gpt2-medium` model and its corresponding tokenizer using `GPT2Tokenizer.from_pretrained('gpt2-medium')` and `GPT2LMHeadModel.from_pretrained('gpt2-medium')`.
    *   Moved the model to the available device (GPU if available, otherwise CPU).

2.  **Custom Dataset (Step 2 in Notebook):**
    *   Defined a `CustomTextDataset` class to load text data from a CSV file (`/content/arabic_dataset.csv`).
    *   Each line read from the CSV is treated as a sample, with the `<|endoftext|>` token appended.
    *   A `DataLoader` is used to batch the data (batch size 1 in the current implementation, but the training loop handles manual batching).
    *   *Note: This step requires a CSV file named `arabic_dataset.csv` to be present in the `/content/` directory when running the notebook.*

3.  **Hyperparameters (Step 3 in Notebook):**
    *   Defined key hyperparameters for fine-tuning: `BATCH_SIZE`, `EPOCHS`, `LEARNING_RATE`, `WARMUP_STEPS`, `MAX_SEQ_LEN`.

4.  **Training (Fine-tuning) (Step 4 in Notebook):**
    *   Used `AdamW` optimizer and `get_linear_schedule_with_warmup` scheduler from `transformers.optimization`.
    *   Implemented a manual training loop that accumulates samples into batches up to `MAX_SEQ_LEN`.
    *   Calculated the language modeling loss using the model's built-in loss computation (`outputs.loss`).
    *   Performed backpropagation and optimizer steps after accumulating `BATCH_SIZE` sequences.
    *   Saved the fine-tuned model's state dictionary after each epoch to the `./trained_models/` directory.
    *   *Note: The manual batching logic building `tmp_tensor` is unconventional. The scheduler initialization `num_training_steps=-1` might also need review depending on the specific training setup intended.*

5.  **Text Generation (Step 5 in Notebook):**
    *   Loaded the model state dictionary from a specified epoch (`gpt2_custom_epoch{MODEL_EPOCH}.pt`).
    *   Defined a `choose_from_top` function for sampling the next token using top-k sampling.
    *   Generated 100 text samples.
    *   Each generation starts with the fixed prompt "SENTENCE:".
    *   Text is generated token by token until an end-of-text token is predicted or a maximum length (100 tokens) is reached.
    *   The generated text is decoded and saved to a text file (`generated_text_epoch{MODEL_EPOCH}.txt`).
    *   *Note: The lab instructions asked to generate text according to a *given* sentence, whereas the current implementation uses a fixed prompt "SENTENCE:".*

### Generated Text

The generated text samples are saved in `generated_text_epoch{MODEL_EPOCH}.txt`. Due to the random scores in Part 1 and the potentially limited custom dataset in Part 2 (the CSV content is not provided), the quality and relevance of the generated text are highly dependent on the contents of `arabic_dataset.csv`.

## Prerequisites

*   Python 3.10+
*   Jupyter Notebook or Google Colab
*   Required Python packages:
    *   `requests`
    *   `beautifulsoup4`
    *   `pandas`
    *   `nltk`
    *   `torch` (with CUDA support recommended for Part 2)
    *   `transformers`
    *   `sklearn`

You can install the necessary libraries using pip:

```bash
pip install requests beautifulsoup4 pandas nltk torch transformers scikit-learn
```
Additionally, download NLTK data:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

You also need to prepare the `arabic_dataset.csv` file in the `/content/` directory for Part 2. The content should be lines of Arabic text.

## How to Run

1.  Clone this repository.
2.  Ensure you have the prerequisites installed.
3.  Place your Arabic text data in a file named `arabic_dataset.csv` in the `/content/` directory (or modify the `CustomTextDataset` class path).
4.  Open the `Atelier_3.ipynb` notebook in Jupyter or Google Colab.
5.  Run the cells sequentially.

## Files

*   `Atelier_3.ipynb`: The main Jupyter notebook containing all the code for data collection, preprocessing, model training (Part 1 and Part 2), and text generation.
*   `README.md`: This file.
*   `data/`: (Optional) Directory for storing the scraped data or `arabic_dataset.csv`.
*   `trained_models/`: (Will be created) Directory to save the fine-tuned GPT-2 model checkpoints.
*   `generated_text_epochX.txt`: (Will be created) File containing the generated text samples.

## Your Synthesis / Brief Report

*Replace this section with your personal synthesis as required by the lab instructions.*

Write a brief synthesis about what you have learned during this lab. Consider reflecting on:
*   The process of web scraping for text data.
*   The standard NLP preprocessing steps and their importance.
*   Working with sequence models (RNNs, LSTMs) in PyTorch.
*   Training and evaluating regression models for NLP tasks.
*   Using the Hugging Face `transformers` library for pre-trained models.
*   Fine-tuning a large language model like GPT-2.
*   Text generation techniques (like sampling).
*   Any challenges encountered and how you addressed them.

## Author

**Mohamed Amine BAHASSOU**
