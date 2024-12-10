


 BERT Integration for Question Pair Classification

This repository contains a Python implementation for finetuning a BERT model to classify question pairs as duplicates or not, using the Quora Question Pairs dataset.

 Features
 Preprocessing and encoding question pairs using the BERT tokenizer.
 Finetuning the BERT model for binary classification.
 Training and evaluation loops.
 Saving and loading the finetuned model.

 Requirements
 Python 3.8+
 PyTorch
 Transformers (Hugging Face)
 pandas
 scikitlearn

 Files
 bert.py: Main script for training and evaluation.
 quora_question_pairs.csv: Dataset file containing question pairs and labels.

 Setup
1. Clone this repository:
   ```bash
   git clone https://github.com/HarshithaTentu/bertintegration.git
   cd bertintegration
   ```

2. Install dependencies:
   ```bash
   pip install r requirements.txt
   ```

3. Place the `quora_question_pairs.csv` file in the same directory as `bert.py`.

 Usage
Run the training script:
```bash
python bert.py
```

The script will:
1. Load and preprocess the dataset.
2. Finetune the BERT model on the question pairs.
3. Save the trained model to `quora_bert_model`.

 Outputs
 Training logs with loss values.
 Validation accuracy after training.
 A finetuned BERT model saved in `quora_bert_model`.

 Notes
 Ensure the dataset file is formatted with columns: `question1`, `question2`, `is_duplicate`.
 Modify paths in the code if necessary to point to your dataset file.

 Contributing
Feel free to fork and submit pull requests to improve the project.
And this is our test data:[https://www.kaggle.com/c/quora-question-pairs/data?select=test.csv.zip
](url)


References:

[https://www.kaggle.com/c/quora-question-pairs](url)

[https://www.kaggle.com/c/quora-question-pairs/discussion](url)

[https://github.com/seatgeek/fuzzywuzzy#usage](url)

[https://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/](url)

[http://proceedings.mlr.press/v37/kusnerb15.pdf](url)

