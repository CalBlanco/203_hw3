# Text Summarization with Seq2Seq and Attention

This repository contains an implementation of an abstractive text summarization system using a sequence-to-sequence model with Bahdanau attention. The model is designed to generate concise summaries from news articles.

## Installation

To set up the environment, install the required dependencies using:

```bash
pip install -r requirements.txt
```

This will install all necessary packages including PyTorch, CUDA support libraries, and evaluation metrics like ROUGE.

## Project Structure

The main components of the project are:

- `src/train.py`: Unified script for training the model and generating predictions
- `src/summarizer.py`: Core model architecture implementation
- `src/encoder.py`: Encoder component of the seq2seq model
- `src/decoder.py`: Decoder component with attention mechanism
- `src/attention.py`: Implementation of Bahdanau attention
- `src/tokenizer.py`: Custom tokenizer for text processing

## Usage

### Command Line Arguments

The train.py script supports various command line arguments for flexible configuration:

```bash
python src/train.py [OPTIONS]
```

#### Common Arguments:
- `--model_name`: Name of the model file to save/load (default: 'model_weights')
#### Training Arguments:
- `--epochs`: Number of training epochs (default: 5)
- `--batch_size`: Training batch size (default: 64)
- `--train_size`: Fraction of training data to use (default: 0.010)
- `--test_size`: Fraction of test data to use

#### Prediction Arguments:
- `--test_file`: Test file path (default: './data/test.txt.src')
- `--output_file`: Output file for predictions (default: 'predictions.txt')

### Training

To train the summarization model:

```bash
python src/train.py --mode train
```

Key features of the training process:
- Mixed precision training with gradient scaling
- Gradient accumulation (4 steps) to effectively increase batch size
- Automatic checkpointing of the best model based on validation loss

### Prediction

To generate summaries using a trained model:

```bash
python src/train.py --mode predict --model_path your_model.pt
```

### Training and Prediction in One Run

To train a model and immediately generate predictions:

```bash
python src/train.py --mode both --model_name my_model
```

This will:
1. Train the model and save it as 'my_model.pt'
2. Use the trained model to generate predictions on the test set

## Model Architecture

The `AbstractiveSummarizer` class in `summarizer.py` implements the core sequence-to-sequence architecture:

```python
class AbstractiveSummarizer(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.encoder = Encoder(vocab_size, embed_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.decoder = Decoder(vocab_size, embed_size, hidden_size)
```

The model consists of three main components:

1. **Encoder**: Processes the input article and creates a hidden representation
2. **Attention**: Implements Bahdanau attention to focus on relevant parts of the input
3. **Decoder**: Generates the summary text with attention context

During forward propagation, the model:
1. Encodes the source text
2. Applies attention at each decoding step
3. Generates output tokens sequentially

The model supports teacher forcing during training, where the probability of using ground truth vs. predicted tokens can be controlled with `teacher_forcing_ratio`.

## Hyperparameters

The default hyperparameters are:
- Embedding dimension: 256
- Hidden size: 512
- Epochs: 5
- Batch size: 64
- Optimizer: Adam

## Data Format

The model expects data in the following format:
- Source articles: `./data/train.txt.src`, `./data/val.txt.src`, `./data/test.txt.src`
- Target summaries: `./data/train.txt.tgt`, `./data/val.txt.tgt`

Each file should contain one article/summary per line.

