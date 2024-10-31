"""
Generation of news titles using LSTM
Data taken from: https://www.kaggle.com/datasets/rmisra/news-category-dataset/
Student: Krunal Rathod
"""
# Packages

import pandas as pd
import pickle
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.data import Dataset
import numpy as np
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence
from torch import Tensor
import math

from rich import print
from rich.console import Console
from rich.table import Table
from art import text2art

import pickle
import nltk

nltk.download("punkt")
from nltk.tokenize import word_tokenize
import os

from collections import Counter


class TextDataset(Dataset):
    def __init__(self, tokenized_sequences, word_to_int):
        self.tokenized_sequences = tokenized_sequences
        self.word_to_int = word_to_int

    def __len__(self):
        return len(self.tokenized_sequences)

    def __getitem__(self, idx):
        # Get the tokenized sequence
        tokenized_sequence = self.tokenized_sequences[idx]

        # Convert the tokenized sequence to a list of integers
        sequence = [self.word_to_int[word] for word in tokenized_sequence.split()]

        # Convert the list of integers to a tensor
        sequence = torch.tensor(sequence).to(device)

        # Return the sequence
        return sequence[:-1].clone().detach().to(device), sequence[
            1:
        ].clone().detach().to(device)


# Implementing the LSTM model
class LSTMModel(nn.Module):
    def __init__(
        self,
        word_to_int,
        vocab_size,
        embedding_dim,
        hidden_dim,
        output_dim,
        n_layers,
        drop_prob=0.5,
    ):
        super(LSTMModel, self).__init__()

        # Initialize attributes
        self.word_to_int = word_to_int
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.drop_prob = drop_prob

        # Initialize layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
        )
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden):
        # Embedding layer
        x = self.embedding(x)

        # LSTM layer
        x, hidden = self.lstm(x, hidden)

        # Dropout layer
        x = self.dropout(x)

        # Fully connected layer
        x = self.fc(x)

        return x, hidden

    def init_state(self, batch_size):
        return (
            torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device),
            torch.  zeros(self.n_layers, batch_size, self.hidden_dim).to(device),
        )

    def init_hidden(self, batch_size):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return (
            torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device),
            torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device),
        )


def collate_fn(batch, pad_value):
    data, targets = zip(*batch)

    padded_data = pad_sequence(data, batch_first=True, padding_value=pad_value)
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=pad_value)

    return padded_data, padded_targets


def random_sampler_next(probabilities):
    # Convert probabilities to numpy array
    probabilities = probabilities.detach().cpu().numpy()

    # Sample from the distribution
    sampled_token_index = np.random.choice(len(probabilities), p=probabilities)

    return sampled_token_index


def sampler_argmax(probabilities):
    # Get the index of the maximum probability
    max_prob_index = torch.argmax(probabilities).item()

    return max_prob_index


def sample(model, sampler_function, word_to_int, int_to_word, device, prompt):
    # Convert the prompt to a list of tokens
    prompt_tokens = [word_to_int[word] for word in prompt.split()]

    # Convert the list of tokens to a tensor and add a batch dimension
    prompt_tokens = torch.tensor(prompt_tokens).unsqueeze(0).to(device)

    # Initialize the hidden state
    hidden = model.init_state(1)

    # Initialize the generated sentence with the prompt
    generated_sentence = prompt

    while True:
        # Forward pass through the model
        output, hidden = model(prompt_tokens, hidden)

        # Get the probabilities of the next word by applying softmax to the output
        probabilities = F.softmax(output[0, -1], dim=0)

        # Use the sampler function to get the next word
        next_word_index = sampler_function(probabilities)

        # Break the loop if the end of sentence token is generated
        if next_word_index == word_to_int["<EOS>"]:
            break

        # Add the generated word to the sentence
        generated_sentence += " " + int_to_word[next_word_index]

        # Add the generated word to the prompt tokens
        prompt_tokens = torch.cat(
            [prompt_tokens, torch.tensor([[next_word_index]]).to(device)], dim=1
        )

    return generated_sentence


def training_loop(
    model,
    dataloader,
    loss_function,
    optimizer,
    device,
    word_to_int,
    int_to_word,
    epochs,
    k,
    clip=None,
):
    model.train()

    loses = []
    perplexities = []
    running_loss = 0
    epoch_losses = []

    epoch = 0
    print(len(dataloader))
    while epoch < epochs:
        epoch += 1
        i = 0
        for x, y in dataloader:
            optimizer.zero_grad()

            prev_state = model.init_state(x.shape[0])
            out, state = model(x, prev_state)

            loss = loss_function(out.transpose(1, 2), y)

            loses.append(loss.item())
            running_loss += loss.item()

            loss.backward()
            if clip:
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            # Print the loss
            # print("Epoch: {}/{}...".format(epoch, epochs),
            # (100 * i / len(dataloader)), "% complete",
            # "Loss: {:.6f}...".format(loss.item()))
            i += 1

        average_loss = running_loss / len(dataloader)
        epoch_losses.append(average_loss)
        perplexity = np.exp(running_loss / len(dataloader))
        print(
            "Epoch: {}/{}...".format(epoch, epochs),
            "Loss: {:.6f}...".format(average_loss),
            "Perplexity: {:.6f}...".format(perplexity),
            "Time: {}".format(time.strftime("%H:%M:%S", time.localtime())),
        )

        if epoch == 1 or epoch == epochs / 2 or epoch == epochs:
            # Generate a sample
            generated_sentence = sample(
                model, sampler_argmax, word_to_int, int_to_word, device, "earth"
            )
            print(
                "Generated Sentence for Epoch {}: {}".format(epoch, generated_sentence)
            )

        perplexities.append(perplexity)

        running_loss = 0

        if average_loss < 1.5:
            print("Average Loss less than 1. Breaking the loop")
            break

    # Plot the Losses
    plt.figure()
    plt.plot(epoch_losses)
    plt.axhline(y=1.5, color="r", linestyle="--")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(["Training Loss (LSTM)"])
    plt.title("Loss vs Epochs (LSTM)")
    plt.savefig("./plots/LSTM-training-loop-losses.png")

    # Plot the Perplexities
    plt.figure()
    plt.plot(perplexities)
    plt.xlabel("Epochs")
    plt.ylabel("Perplexity")
    plt.legend(["Training Perplexity"])
    plt.title("Perplexity vs Epochs")
    plt.savefig("./plots/LSTM-training-loop-perplexities.png")

    return model, loses, perplexities


def train_with_TBBTT(
    max_epochs, model, dataloader, criterion, optimizer, chunk_size, device, clip=None
):
    losses = []
    perplexities = []
    epoch = 0
    while epoch < max_epochs:
        epoch += 1
        model.train()
        loss = None
        total_loss = 0
        num_batches = 0
        for input, output in dataloader:
            # Get the number of chunks
            n_chunks = input.shape[1] // chunk_size

            # Loop on chunks
            for j in range(n_chunks):
                # Switch between the chunks
                if j < n_chunks - 1:
                    input_chunk = (
                        input[:, j * chunk_size : (j + 1) * chunk_size]
                        .to(device)
                        .to(torch.int64)
                    )
                    output_chunk = (
                        output[:, j * chunk_size : (j + 1) * chunk_size]
                        .to(device)
                        .to(torch.int64)
                    )
                else:
                    input_chunk = input[:, j * chunk_size :].to(device).to(torch.int64)
                    output_chunk = (
                        output[:, j * chunk_size :].to(device).to(torch.int64)
                    )
                # Initialise model's state and perform forward pass
                # If it is the first chunk, initialise the state to 0
                if j == 0:
                    h, c = model.init_hidden(input_chunk.size(0))
                else:  # Initialize the state to the previous state - detached!
                    h, c = h.detach(), c.detach()

                # Forward step
                output, (h, c) = model(input_chunk, (h, c))

                # Calculate loss
                loss = criterion(
                    output.view(-1, model.vocab_size), output_chunk.view(-1)
                )

                total_loss += loss.item()
                num_batches += 1

                # Calculate gradients and update parameters
                optimizer.zero_grad()
                loss.backward()

                # Clipping if needed
                if clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

                # Update parameters
                optimizer.step()

        average_loss = total_loss / num_batches
        # Print loss and perplexity every epoch
        print(
            f"Epoch: {epoch}, Loss: {average_loss}, Perplexity: {np.exp(average_loss)}",
            f'Time: {time.strftime("%H:%M:%S", time.localtime())}',
        )

        # Keep track of losses and perplexities
        losses.append(average_loss)
        perplexities.append(math.exp(average_loss))

        model.eval()
        # TODO prompt a sentence from the model
        generated_sentence = sample(
            model, sampler_argmax, word_to_int, int_to_word, device, "earth"
        )
        print("Generated Sentence for Epoch {}: {}".format(epoch, generated_sentence))

        if average_loss < 1:
            print("Average Loss less than 1. Breaking the loop")
            break

    # Plot the Losses
    plt.figure()
    plt.plot(losses)
    plt.axhline(y=1.5, color="r", linestyle="--")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(["Training Loss (TBBTT)"])
    plt.title("Loss vs Epochs")
    plt.savefig("./plots/TBBTT-training-loop-losses.png")

    # Plot the Perplexities
    plt.figure()
    plt.plot(perplexities)
    plt.xlabel("Epochs")
    plt.ylabel("Perplexity")
    plt.legend(["Training Perplexity (TBBTT)"])
    plt.title("Perplexity vs Epochs")
    plt.savefig("./plots/TBBTT-training-loop-perplexities.png")

    return model, losses, perplexities


if __name__ == "__main__":
    print(text2art("DLL: Assignment 3"))
    os.makedirs("./kaggle/working", exist_ok=True)
    print("-------------------------")
    print("[bold cyan]QUESTION 1.1: DATA (20 Points) [/bold cyan]")
    print("-------------------------")

    os.makedirs("./data/", exist_ok=True)
    os.makedirs("./model/", exist_ok=True)
    os.makedirs("./plots/", exist_ok=True)

    print("[bold green]Question 1.1.1 Download Data (2pts) [/bold green]")
    df = pd.read_json(
        "./data/News_Category_Dataset_v3.json", lines=True
    )
    df_politics = df[df["category"] == "POLITICS"]

    print(df_politics.head(3))

    print("\n[bold green]Question 1.1.2 Tokenization (3pts) [/bold green]")

    if os.path.exists("./data/working/tokenized_title.pkl"):
        with open("./data/tokenized_title.pkl", "rb") as f:
            tokenized_title = pickle.load(f)
    else:
        tokenized_title = []
        for i in range(len(df_politics["headline"])):
            # print(df_politics['headline'].iloc[i])
            title = df_politics["headline"].iloc[i].lower() + " <EOS>"
            tokenized_title.append(title)

        with open("./data/tokenized_title.pkl", "wb") as f:
            pickle.dump(tokenized_title, f)

    # Print first three tokenized titles
    print(tokenized_title[:3])

    print("\n[bold green]Question 1.1.3 Dictionaries (4pts) [/bold green]")

    if (
        os.path.exists("./data/word_to_int.pkl")
        and os.path.exists("./data/int_to_word.pkl")
        and os.path.exists("./data/all_words.pkl")
    ):
        with open("./data/word_to_int.pkl", "rb") as f:
            word_to_int = pickle.load(f)
        with open("./data/int_to_word.pkl", "rb") as f:
            int_to_word = pickle.load(f)
        with open("./data/all_words.pkl", "rb") as f:
            all_words = pickle.load(f)
    else:
        all_words = []
        for title in tokenized_title:
            for word in title.split():
                if word != "<EOS>" and word != "<PAD>" and word not in all_words:
                    all_words.append(word)

        all_words = ["<EOS>"] + all_words + ["<PAD>"]

        word_to_int = {word: i for i, word in enumerate(all_words)}
        int_to_word = {i: word for word, i in word_to_int.items()}

        print("Length of all_words: ", len(all_words))
        print("Length of word_to_int: ", len(word_to_int))
        print("Length of int_to_word: ", len(int_to_word))
        # print("int_to_word:", int_to_word)

        with open("./data/word_to_int.pkl", "wb") as f:
            pickle.dump(word_to_int, f)
        with open("./data/int_to_word.pkl", "wb") as f:
            pickle.dump(int_to_word, f)
        with open("./data/all_words.pkl", "wb") as f:
            pickle.dump(all_words, f)

    print("\n[bold green]Question 1.1.4 Dataset class (5pts) [/bold green]")
    batch_size = 64
    dataset = TextDataset(tokenized_title, word_to_int)

    if batch_size == 1:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda x: collate_fn(x, word_to_int["<PAD>"]),
        )

    print("\n[bold green]TRAINING LOOP [/bold green]")

    hidden_dim = 1024
    embedding_dim = 150
    vocab_size = len(all_words)
    output_dim = vocab_size
    n_layers = 1
    drop_prob = 0.3
    lr = 0.001
    epochs = 12
    clip = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    model = LSTMModel(
        word_to_int,
        vocab_size,
        embedding_dim,
        hidden_dim,
        output_dim,
        n_layers,
        drop_prob,
    )

    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.CrossEntropyLoss()

    model, losses, perplexities = training_loop(
        model,
        dataloader,
        loss_function,
        optimizer,
        device,
        word_to_int,
        int_to_word,
        epochs,
        2,
        clip,
    )

    # Save the model
    torch.save(model.state_dict(), "/model/lstm_model.pt")

    epoch = 10
    hidden_dim = 2048
    chunk_size = 15

    model_new = LSTMModel(
        word_to_int,
        vocab_size,
        embedding_dim,
        hidden_dim,
        output_dim,
        n_layers,
        drop_prob,
    )
    model_new.to(device)

    optimizer = optim.Adam(model_new.parameters(), lr=lr)
    loss_function = nn.CrossEntropyLoss(ignore_index=word_to_int["<PAD>"])

    model_new, losses, perplexities = train_with_TBBTT(
        epoch, model_new, dataloader, loss_function, optimizer, chunk_size, device, clip
    )

    torch.save(model_new.state_dict(), "/model/TBBTT_model.pt")

    # Import the model
    original_model = LSTMModel(
        word_to_int,
        vocab_size,
        embedding_dim,
        hidden_dim,
        output_dim,
        n_layers,
        drop_prob,
    )
    original_model.to(device)
    original_model.load_state_dict(torch.load("./model/lstm_model.pt"))

    epoch = 10
    hidden_dim = 2048
    chunk_size = 15

    # Import the new model
    new_model = LSTMModel(
        word_to_int,
        vocab_size,
        embedding_dim,
        hidden_dim,
        output_dim,
        n_layers,
        drop_prob,
    )
    new_model.to(device)
    new_model.load_state_dict(torch.load("./model/TBBTT_model.pt"))

    # Generating 3 sentences using the new model and the old model by using sampling strategies and greedy search

    for i in range(3):
        generated_sentence_original_mode = sample(
            original_model,
            random_sampler_next,
            word_to_int,
            int_to_word,
            device,
            "the president wants",
        )
        print(
            "Generated Sentence for Sampling Strategy with Original Model {}: {}".format(
                i + 1, generated_sentence_original_mode
            )
        )

        gererated_sentence_new_model = sample(
            new_model,
            random_sampler_next,
            word_to_int,
            int_to_word,
            device,
            "the president wants",
        )
        print(
            "Generated Sentence for Sampling Strategy with New Model {}: {}".format(
                i + 1, gererated_sentence_new_model
            )
        )

        generated_sentence_original_mode_greedy = sample(
            original_model,
            sampler_argmax,
            word_to_int,
            int_to_word,
            device,
            "the president wants",
        )
        print(
            "Generated Sentence for Greedy Search with Original Model {}: {}".format(
                i + 1, generated_sentence_original_mode_greedy
            )
        )

        gererated_sentence_new_model_greedy = sample(
            new_model,
            sampler_argmax,
            word_to_int,
            int_to_word,
            device,
            "the president wants",
        )
        print(
            "Generated Sentence for Greedy Search with New Model {}: {}".format(
                i + 1, gererated_sentence_new_model_greedy
            )
        )
