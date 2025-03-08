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

- `src/train.py`: Script for training the summarization model
- `src/predict.py`: Script for generating summaries using a trained model
- `src/summarizer.py`: Core model architecture implementation
- `src/encoder.py`: Encoder component of the seq2seq model
- `src/decoder.py`: Decoder component with attention mechanism
- `src/attention.py`: Implementation of Bahdanau attention
- `src/tokenizer.py`: Custom tokenizer for text processing

## Usage

### Training

To train the summarization model, run:

```bash
python src/train.py
```

The training script:
1. Sets up GPU devices (defaults to using GPUs 1, 2, and 3)
2. Builds or loads a vocabulary from the training data
3. Creates a seq2seq model with attention
4. Trains the model for 5 epochs with a batch size of 64
5. Saves the best model based on validation loss

Key features of the training process:
- Mixed precision training with gradient scaling
- Gradient accumulation (4 steps) to effectively increase batch size
- Parallel data loading with multiple workers
- Automatic checkpointing of the best model

### Prediction

To generate summaries using a trained model:

```bash
python src/predict.py
```

The prediction script:
1. Loads the trained model and vocabulary
2. Processes test articles in batches
3. Generates summaries using beam search
4. Saves the predictions to `predictions.txt`

The batch inference approach allows for efficient processing of large test sets.

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

