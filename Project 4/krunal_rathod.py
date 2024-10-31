"""
Code for the 4th assignment
Student: Krunal Rathod
"""
############################
# Packages
############################
import torch
import torch.nn as nn
import math
import regex as re
from torch import optim
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import random
import re
import os
from io import open
from rich import print
from art import text2art
from matplotlib import pyplot as plt
import pickle
import random
import warnings
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import random_split
import numpy as np
import itertools

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
from torch.optim.lr_scheduler import StepLR
import torch.nn.init as init

import sys


class DummyStream:
    def write(self, text):
        pass


############################
# Classes
############################
# Vocabulary class
class Vocabulary:
    """
    Class for dealing with our corpus
    """

    def __init__(self, name, pairs):
        """
        Args:
            name (str): name of the language
            pairs (list): list of pairs of sentences
        """
        self.name = name
        self.word2index = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2}
        self.index2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>"}
        self.pairs = pairs
        self.n_words = 3  # Count SOS and EOS

        self.add_vocabulary()

    def add_vocabulary(self):
        for pair in self.pairs:
            self.add_sentence(pair[0])
            self.add_sentence(pair[1])

    def add_sentence(self, sentence):
        for word in sentence:
            self.add_word(word)

    def add_word(self, word):
        """
        Add a word to the vocabulary
        :param word: a string
        """
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1

    def index2word(self, index):
        return self.index2word[index]


def clear_punctuation(text):
    """
    This function removes all the punctuation from a sentence and insert a blank between any letter and !?. EDITS: I have added more filters to make the language more uniform.
    :param s: a string
    :return: the "cleaned" string
    """
    re.sub(
        r"[^a-zA-Z.!?]+", r" ", text
    )  # Remove all the character that are not letters, puntuation or numbers
    # Insert a blank between any letter and !?. using regex
    text = re.sub(r"([a-zA-Z])([!?.])", r"\1 \2", text)
    return text


# Dataset class
class Dataset(torch.utils.data.Dataset):
    def __init__(self, vocabulary, pairs):
        self.vocabulary = vocabulary
        self.pair = pairs

    def __len__(self):
        return len(self.pair)

    def __getitem__(self, ix):
        pair = self.pair[ix]
        question, answer = pair

        question_tensor = torch.tensor(
            [self.vocabulary.word2index[word] for word in question]
        )

        answer_tensor = torch.tensor(
            [self.vocabulary.word2index[word] for word in answer]
        )

        return question_tensor, answer_tensor


def collate_fn(batch, pad_value):
    data, targets = zip(*batch)

    padded_data = pad_sequence(data, batch_first=True, padding_value=pad_value)
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=pad_value)

    return padded_data, padded_targets


class PositionalEncoding(nn.Module):
    """
    Adapted from
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, d_model, dropout=0.0, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        try:
            assert x.size(0) < self.max_len
        except:
            print(
                "The length of the sequence is bigger than the max_len of the positional encoding. Increase the max_len or provide a shorter sequence."
            )
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        encoder_layers,
        decoder_layers,
        dim_feedforward,
        num_heads,
        dropout_p,
        pad_id=0,
    ):
        super().__init__()

        # Stuff you may need
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.num_heads = num_heads

        # Add an embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)

        # Add a positional encoding layer
        self.pos_encoder = PositionalEncoding(d_model, dropout_p)

        # Add a transformer layer, you can use nn.Transformer. You can use the default values for the parameters, but what about batch_first?
        self.transformer = nn.Transformer(
            d_model,
            num_heads,
            encoder_layers,
            decoder_layers,
            dim_feedforward,
            dropout_p,
            batch_first=True,
        )

        # Add a linear layer. Note: output should be probability distribution over the vocabulary
        self.linear = nn.Linear(d_model, vocab_size)

    def create_padding_mask(self, x, pad_id=0):
        # Create a boolean mask for the <PAD> tokens
        mask = x == pad_id
        return mask

    def forward(self, src, tgt):
        # S is the source sequence length, T is the target sequence length, N is the batch size, E is the feature number
        # src: (N, S)
        # tgt: (N, T)
        # src_pad_mask: (N, S)
        # tgt_pad_mask: (N, T)
        # mask the future : (N * num_heads, T, T)

        src_pad_mask = self.create_padding_mask(src, self.pad_id)  # (N, S)
        tgt_pad_mask = self.create_padding_mask(tgt, self.pad_id)  # (N, T)

        src = self.embedding(src)
        tgt = self.embedding(tgt)

        src = self.pos_encoder(src)  # (N, S, E)
        tgt = self.pos_encoder(tgt)  # (N, T, E)

        # Mask the memory
        memory_key_padding_mask = src_pad_mask  # (N, S)

        # Mask the future
        tgt_mask = self.transformer.generate_square_subsequent_mask(
            tgt.size(1), dtype=torch.bool
        ).to(
            tgt.device
        )  # (T, T)
        # tgt_mask = tgt_mask.bool()

        # Expand to make it N * num_heads, T, T
        tgt_mask = tgt_mask.unsqueeze(0).repeat(
            tgt.size(0) * self.num_heads, 1, 1
        )  # (N, T, T

        # Transformer
        output = self.transformer(
            src,
            tgt,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )  # (N, T, E)

        # Apply layer normalization
        output = self.linear(output)
        return output


############################
# Methods
############################


def printLines(file, n=10):
    """
    Print the first n lines of a file
    :param file: a string
    :param n: an integer
    """
    with open(file, "rb") as datafile:
        lines = datafile.readlines()
    for line in lines[:n]:
        print(line)


def load_lines(filename):
    """
    Load movie_lines.txt into a dictionary
    """
    lines = {}
    with open(filename, "r", encoding="iso-8859-1") as f:
        for line in f:
            parts = line.split(" +++$+++ ")
            lines[parts[0]] = parts[-1].strip()
    return lines


def load_conversations(filename, lines):
    """
    Load movie_conversations.txt and form pairs of sentences.
    """
    pairs = []
    with open(filename, "r", encoding="iso-8859-1") as f:
        for line in f:
            parts = line.split(" +++$+++ ")
            conversation = [lines[line_id] for line_id in eval(parts[-1])]
            for i in range(len(conversation) - 1):
                pairs.append((conversation[i], conversation[i + 1]))
    return pairs


def id_to_line(line):
    """
    Create a dictionary that maps the id of a sentence to the sentence itself.
    """
    line = open(line, encoding="iso-8859-1").read().split("\n")
    id_to_line = {}
    for l in line:
        clean_line = l.split(" +++$+++ ")
        if len(clean_line) == 5:
            id_to_line[clean_line[0]] = clean_line[4]
    return id_to_line


def id_to_conversation(conversation):
    """
    Create a dictionary that maps the id of a conversation to the list of sentences in the conversation.
    """
    conversation = open(conversation, encoding="iso-8859-1").read().split("\n")
    id_to_conversation = []
    for c in conversation[:-1]:
        clean_line = c.split(" +++$+++ ")[-1][1:-1].replace("'", "").replace(" ", "")
        id_to_conversation.append(clean_line.split(","))
    return id_to_conversation


def tokenize_sentence(sentence):
    """
    Tokenize a sentence at the word level and append the <EOS> token.
    """
    sentence = clear_punctuation(sentence)
    tokens = sentence.split()
    tokens.append("<EOS>")
    return tokens


def tokenize_pairs(pairs):
    """
    Tokenize pairs of sentences.
    """
    tokenized_pairs = []
    for question, answer in pairs:
        tokenized_question = tokenize_sentence(question)
        tokenized_answer = ["<SOS>"] + tokenize_sentence(answer)
        tokenized_pairs.append((tokenized_question, tokenized_answer))
    return tokenized_pairs


def filter_pair_by_words(pair, filtered_words):
    """
    Filter out the words that are not in the filtered_words list.
    """
    filtered_pair = []
    for sentence in pair:
        filtered_sentence = []
        for word in sentence:
            if word in filtered_words:
                filtered_sentence.append(word)
        filtered_pair.append(filtered_sentence)
    return filtered_pair


def process_data(batch_size, random_sentence_count):
    # Download the data
    print("-------------------------")
    print("[bold cyan]QUESTION 1: DATA (40 Points) [/bold cyan]")
    print("-------------------------")

    print("\n[bold green]Question 1.1 Download Data (5 pts)[/bold green]")

    movie_conversation_lines = "movie_conversations.txt"
    movie_lines_lines = "movie_lines.txt"
    movie_conversation = os.path.join("data", movie_conversation_lines)
    movie_lines = os.path.join("data", movie_lines_lines)
    print("Data Downloaded")

    # printLines(movie_conversation)
    # printLines(movie_lines)

    # Create the pairs
    print("\n[bold green]Question 1.2 Create the Pair (5 pts)[/bold green]")
    lines = load_lines(movie_lines)
    pairs = load_conversations(movie_conversation, lines)
    id_to_lines = id_to_line(movie_lines)
    id_to_conversations = id_to_conversation(movie_conversation)
    questions = []
    answers = []
    # print(id_to_conversation[:5])
    for conv in id_to_conversations:
        for i in range(len(conv) - 1):
            questions.append(id_to_lines[conv[i]])
            answers.append(id_to_lines[conv[i + 1]])
    # for i in range (0, 5):
    #    print(questions[i])
    #    print(answers[i])
    #    print()
    print("Number of Questions:", len(questions))
    print("Number of Answers:", len(answers))
    # Clean the Data
    for i in range(len(questions)):
        questions[i] = clear_punctuation(questions[i])
        answers[i] = clear_punctuation(answers[i])
    # Create the pairs
    pairs = []
    for i in range(len(questions)):
        pairs.append((questions[i], answers[i]))
    # Tokenize the data
    print("\n[bold green]Question 1.3 Tokenize the Data (10 pts)[/bold green]")

    if os.path.exists("./data/tokenized_pairs_word_limit.pkl"):
        with open("./data/tokenized_pairs_word_limit.pkl", "rb") as f:
            tokenized_pair = pickle.load(f)
        print("Tokenized Pairs already created")

        print(
            "\n[bold green]Question 1.4 Filter ot the sentences that are too long (3 pts)[/bold green]"
        )
        print("Tokenized Pairs already filtered by length")
    else:
        tokenized_pair = tokenize_pairs(pairs)
        print("Tokenized Pairs created")
        # Filter out the sentences that are too long
        print(
            "\n[bold green]Question 1.4 Filter ot the sentences that are too long (3 pts)[/bold green]"
        )
        print("Length of the Tokenized Pairs before Filter:", len(tokenized_pair))
        min_length = 2
        max_length = 20
        for pair in tokenized_pair:
            if len(pair[0]) > max_length or len(pair[1]) > max_length:
                # print("Pair removed")
                tokenized_pair.remove(pair)
            elif len(pair[0]) < min_length or len(pair[1]) < min_length:
                tokenized_pair.remove(pair)
            else:
                continue
        print("Length of the Tokenized Pairs after Filter:", len(tokenized_pair))

        # Save the tokenized pairs as pkl file
        with open("./data/tokenized_pairs_word_limit.pkl", "wb") as f:
            pickle.dump(tokenized_pair, f)
    print(
        "\n[bold green]Question 1.6 Removing sentences which contains rare words (3 pts)[/bold green]"
    )
    frequency_vocab = {}
    for pair in tokenized_pair:
        for word in pair[0]:
            if word in frequency_vocab:
                frequency_vocab[word] += 1
            else:
                frequency_vocab[word] = 1
        for word in pair[1]:
            if word in frequency_vocab:
                frequency_vocab[word] += 1
            else:
                frequency_vocab[word] = 1
    # Plot the frequency of the words
    plt.title("Word Frequencies")
    plt.hist(list(frequency_vocab.values()), bins=50, log=True)
    plt.xlabel("Word Frequency")
    plt.ylabel("Number of Words")
    plt.title("Word Frequencies")
    plt.savefig("./data/word_frequencies.png")

    # Making a list of the words that are too rare
    print("Number of Words:", len(frequency_vocab))
    min_frequency = 10
    rare_words = []
    for word in frequency_vocab:
        if frequency_vocab[word] < min_frequency:
            rare_words.append(word)
    print("Number of Rare Words:", len(rare_words))
    print("Number of Words before Filter by frequency:", len(tokenized_pair))
    if os.path.exists("./data/tokenized_pairs_word_limit_filtered.pkl") == False:
        # Create a new list to store the pairs we want to keep
        filtered_pairs = []
        # Iterate over the tokenized pairs
        for pair in tokenized_pair:
            # Check if any of the words in the pair are in the list of rare words
            is_good_pair = True
            
            for words in itertools.chain(pair[0], pair[1]):
                if words in rare_words:
                    is_good_pair = False
                    break
            if is_good_pair:
                filtered_pairs.append(pair)

        tokenized_pair = filtered_pairs

        print("Number of Words after Filter by frequency:", len(tokenized_pair))

        with open("./data/tokenized_pairs_word_limit_filtered.pkl", "wb") as f:
            pickle.dump(tokenized_pair, f)
    else:
        print("Tokenized Pairs already filtered by frequency")
        with open("./data/tokenized_pairs_word_limit_filtered.pkl", "rb") as f:
            tokenized_pair = pickle.load(f)
        print("Number of Words after Filter by frequency:", len(tokenized_pair))

    print(
        "\n[bold green]Question 1.8 Randomly Sampling a subset of 10,000 sentences(3pts)[/bold green]"
    )
    random_sentences = random.sample(tokenized_pair, random_sentence_count)
    print("Number of Random Sentences:", len(random_sentences))
    # print(random_sentences[:5])

    print("\n[bold green]Question 1.10 Creating a Dataloader Object(5pts)[/bold green]")
    vocab_name = "myVocab"
    vocabulary = Vocabulary(vocab_name, random_sentences)
    for pair in random_sentences:
        vocabulary.add_sentence(pair[0])
        vocabulary.add_sentence(pair[1])

    print("Number of Words in Vocabulary:", vocabulary.n_words)
    dataset = Dataset(vocabulary, random_sentences)
    dataloader = 0
    return dataset, dataloader, vocabulary


def standard_train(
    model, train_loader, val_loader, num_epochs=20, lr=0.0001, step_size=0.1, gamma=0.1
):
    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move the model to the device
    model = model.to(device)

    # Initialize the optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr)

    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    # Define the loss function
    criterion = nn.CrossEntropyLoss(ignore_index=model.pad_id)

    training_loss = []
    validation_loss = []
    min_val_loss = float("inf")
    epochs_no_improve = 0
    n_epochs_stop = 5  # number of epochs to stop if no improvement

    # Loop over the epochs
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        # Loop over the training data
        for input_sentence, output_sentence in train_loader:
            # Move the data to the device
            input_sentence = input_sentence.to(device)
            output_sentence = output_sentence.to(device)

            # Prepare the input and target sentences for the decoder
            decoder_input = output_sentence[:, :-1]  # Remove <EOS> from the end
            decoder_target = output_sentence[:, 1:]  # Remove <SOS> from the start

            # print("Size of the input sentence:", decoder_input.size(0))
            # print("Size of the output sentence:", decoder_target.size(0))
            optimizer.zero_grad()
            # Forward pass
            output = model(input_sentence, decoder_input)

            # output = output[:, 1:].reshape(-1, output.shape[-1])
            # decoder_target = decoder_target[:, 1:].reshape(-1)

            # non_pad_elements = (decoder_target != model.pad_id).nonzero()
            # output = output[non_pad_elements.squeeze(), :]
            # decoder_target = decoder_target[non_pad_elements.squeeze()]

            # Calculate the loss
            loss = criterion(output.transpose(1, 2), decoder_target)

            # Backward pass and optimize
            loss.backward()

            # Clip the gradients
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Update the parameters
            optimizer.step()

            train_loss += loss.item()

        # Evaluate on the validation set
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for input_sentence, output_sentence in val_loader:
                # Move the data to the device
                input_sentence = input_sentence.to(device)
                output_sentence = output_sentence.to(device)

                # Prepare the input and target sentences for the decoder
                decoder_input = output_sentence[:, :-1]  # Remove <EOS> from the end
                decoder_target = output_sentence[:, 1:]  # Remove <SOS> from the start

                # Forward pass
                output = model(input_sentence, decoder_input)

                # Calculate the loss
                loss = criterion(output.transpose(1, 2), decoder_target)

                val_loss += loss.item()

        scheduler.step()

        # Print the losses
        print(
            f"Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader)}, Val Loss: {val_loss/len(val_loader)}"
        )
        training_loss.append(train_loss / len(train_loader))
        validation_loss.append(val_loss / len(val_loader))

        # Early stopping
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == n_epochs_stop:
                print("Early stopping!")
                break


        # # If the validation loss is below 1.5, stop the training
        # if val_loss / len(train_loader) < 1.5:
        #     print("Training complete.")
        #     break

    plt.figure(figsize=(10, 5))
    plt.title("Training Loss of the Standard Training Loop")
    plt.plot(training_loss, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    plt.savefig("./data/training_loss_simple_training_loop.png")

    plt.figure(figsize=(10, 5))
    plt.title("Validation Loss of the Standard Training Loop")
    plt.plot(validation_loss, label="Validation Loss")
    plt.axhline(y=1.5, color="r", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    plt.savefig("./data/validation_loss_simple_training_loop.png")

    return model, validation_loss, training_loss


def train_ga(
    model,
    train_loader,
    num_epochs=20,
    lr=0.0001,
    step_size=0.1,
    gamma=0.1,
    accumulation_steps=32,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    criterion = nn.CrossEntropyLoss(ignore_index=model.pad_id)
    
    training_loss = []
    validation_loss = []
    min_val_loss = float("inf")
    epochs_no_improve = 0
    n_epochs_stop = 5  # number of epochs to stop if no improvement
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        optimizer.zero_grad()
        for input_sentence, output_sentence in train_loader:
            input_sentence = input_sentence.to(device)
            output_sentence = output_sentence.to(device)
            decoder_input = output_sentence[:, :-1]
            decoder_target = output_sentence[:, 1:]
            
            optimizer.zero_grad()
            output = model(input_sentence, decoder_input)
            
            loss = criterion(output.transpose(1, 2), decoder_target)
            normalised_loss = loss / accumulation_steps
            normalised_loss.backward()
            
            if (epoch + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for input_sentence, output_sentence in val_loader:
                input_sentence = input_sentence.to(device)
                output_sentence = output_sentence.to(device)
                decoder_input = output_sentence[:, :-1]
                decoder_target = output_sentence[:, 1:]
                
                output = model(input_sentence, decoder_input)
                
                loss = criterion(output.transpose(1, 2), decoder_target)
                
                val_loss += loss.item()
                
        scheduler.step()
        print(
            f"Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader)}, Val Loss: {val_loss/len(val_loader)}"
        )
        
        training_loss.append(train_loss / len(train_loader))
        validation_loss.append(val_loss / len(val_loader))
        
        # Early stopping
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == n_epochs_stop:
                print("Early stopping!")
                break
            
                
    plt.figure(figsize=(10, 5))
    plt.title("Training Loss of the Gradient Accumulation Training Loop")
    plt.plot(training_loss, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    plt.savefig("./data/training_loss_gradient_accumulation.png")
    
    plt.figure(figsize=(10, 5))
    plt.title("Validation Loss of the Gradient Accumulation Training Loop")
    plt.plot(validation_loss, label="Validation Loss")
    plt.axhline(y=1.5, color="r", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    plt.savefig("./data/validation_loss_gradient_accumulation.png")  
            

    return model, validation_loss, training_loss


def greedy_decoding(transformer_model, source_seq, sequence_max_length, start_token):
    transformer_model = transformer_model.to(device)
    source_seq = source_seq.to(device)
    target_input = torch.ones(1, 1).fill_(start_token).type(torch.long).to(device)

    for i in range(sequence_max_length - 1):
        model_output = transformer_model(source_seq, target_input)
        next_token = model_output.argmax(dim=2)[:, -1].item()
        target_input = torch.cat(
            [target_input, torch.tensor([[next_token]], device=device)], dim=1
        )

        if next_token == vocabulary.word2index["<EOS>"]:
            break

    return target_input


def top_k_sampling(
    transformer_model, source_seq, sequence_max_length, start_token, top_k
):
    transformer_model = transformer_model.to(device)
    source_seq = source_seq.to(device)
    target_input = torch.ones(1, 1).fill_(start_token).type(torch.long).to(device)

    for i in range(sequence_max_length - 1):
        model_output = transformer_model(source_seq, target_input)
        probabilities = F.softmax(model_output[:, -1, :], dim=1)
        top_k_probabilities, top_k_indices = torch.topk(probabilities, top_k, dim=1)
        next_token = top_k_indices[0, torch.multinomial(top_k_probabilities, 1)].item()
        target_input = torch.cat(
            [target_input, torch.tensor([[next_token]], device=device)], dim=1
        )

        if next_token == vocabulary.word2index["<EOS>"]:
            break

    return target_input


def evaluation_model(
    model, input_sequence_1, input_sequence_2, input_sequence_3, model_name, vocabulary
):
    sys.stdout = original_stdout
    print("Evaluating of the model:", model_name)
    print("Greedy Decoding")

    eos_token_id = vocabulary.word2index["<EOS>"]
    max_length = 5

    output_sequence_1 = greedy_decoding(
        model, input_sequence_1, max_length, vocabulary.word2index["<SOS>"]
    )
    output_sequence_1 = output_sequence_1.squeeze(0)
    output_sequence_1 = [
        vocabulary.index2word[word.item()] for word in output_sequence_1
    ]
    output_sequence_1_string = " ".join(output_sequence_1)

    output_sequence_2 = greedy_decoding(
        model, input_sequence_2, max_length, vocabulary.word2index["<SOS>"]
    )
    output_sequence_2 = output_sequence_2.squeeze(0)
    output_sequence_2 = [
        vocabulary.index2word[word.item()] for word in output_sequence_2
    ]
    output_sequence_2_string = " ".join(output_sequence_2)

    output_sequence_3 = greedy_decoding(
        model, input_sequence_3, max_length, vocabulary.word2index["<SOS>"]
    )
    output_sequence_3 = output_sequence_3.squeeze(0)
    output_sequence_3 = [
        vocabulary.index2word[word.item()] for word in output_sequence_3
    ]
    output_sequence_3_string = " ".join(output_sequence_3)

    print("Input:", "Where are you?")
    print(
        "Output:",
        output_sequence_1_string,
    )

    print(
        "Input:",
        "how are you doing?",
    )
    print(
        "Output:",
        output_sequence_2_string,
    )

    print("Input:", "I am doing great how about you")
    print(
        "Output:",
        output_sequence_3_string,
    )

    print("Top-K Decoding")

    output_sequence_1 = top_k_sampling(
        model, input_sequence_1, max_length, vocabulary.word2index["<SOS>"], 5
    )
    output_sequence_1 = output_sequence_1.squeeze(0)
    output_sequence_1 = [
        vocabulary.index2word[word.item()] for word in output_sequence_1
    ]
    output_sequence_1_string = " ".join(output_sequence_1)

    output_sequence_2 = top_k_sampling(
        model, input_sequence_2, max_length, vocabulary.word2index["<SOS>"], 5
    )
    output_sequence_2 = output_sequence_2.squeeze(0)
    output_sequence_2 = [
        vocabulary.index2word[word.item()] for word in output_sequence_2
    ]
    output_sequence_2_string = " ".join(output_sequence_2)

    output_sequence_3 = top_k_sampling(
        model, input_sequence_3, max_length, vocabulary.word2index["<SOS>"], 5
    )
    output_sequence_3 = output_sequence_3.squeeze(0)
    output_sequence_3 = [
        vocabulary.index2word[word.item()] for word in output_sequence_3
    ]
    output_sequence_3_string = " ".join(output_sequence_3)

    print("Input:", "Where are you?")
    print(
        "Output:",
        output_sequence_1_string,
    )

    print(
        "Input:",
        "how are you doing?",
    )
    print(
        "Output:",
        output_sequence_2_string,
    )

    print("Input:", "I am doing great how about you")
    print(
        "Output:",
        output_sequence_3_string,
    )


###########################
# Methods
############################

if __name__ == "__main__":
    # Ignore warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    # Save a reference to the original stdout
    original_stdout = sys.stdout
    print(text2art("DLL: Assignment 4"))

    # Checking if GPU is available
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")
    print(f"Using device: {device}")

    # !!! Don't change the seed !!!
    torch.manual_seed(42)
    # !!!!!!

    # Model Parameters
    d_model = 512
    encoder_layers = 6
    decoder_layers = 6
    dim_feedforward = 2048
    num_heads = 8
    dropout_p = 0.2
    num_epochs = 30

    random_sentence_count = 20000
    batch_size = 32

    dataset, dataloader, vocabulary = process_data(batch_size, random_sentence_count)

    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(dataset))  # 80% for training
    val_size = len(dataset) - train_size  # 20% for validation
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create data loaders for training and validation sets
    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: collate_fn(x, 0),
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: collate_fn(x, 0),
    )

    model = TransformerModel(
        vocabulary.n_words,
        d_model,
        encoder_layers,
        decoder_layers,
        dim_feedforward,
        num_heads,
        dropout_p,
    )

    model = model.to(device)
    print("-------------------------")
    print("[bold cyan]QUESTION 3: Training (35 Points) [/bold cyan]")
    print("-------------------------")

    print(
        "\n[bold green]Question 3.1 Training Standard Training Loop (15pts)[/bold green]"
    )
    
    if os.path.exists("./data/standard_model.pt") == False:
        model, val_loss, train_loss = standard_train( model, dataloader, val_loader, num_epochs=num_epochs)
        # Save the Standard Model
        torch.save(model.state_dict(), "./data/standard_model.pt")
    else:
        print("Standard Model already trained")

    gradient_accumulation = 32
    batch_size = 32
    random_sentence_count = 30000
    d_model = 512
    dropout_p = 0.3

    dataset, dataloader, vocabulary = process_data(batch_size, random_sentence_count)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: collate_fn(x, vocabulary.word2index["<PAD>"]),
    )

    model = TransformerModel(
        vocabulary.n_words,
        d_model,
        encoder_layers,
        decoder_layers,
        dim_feedforward,
        num_heads,
        dropout_p,
    )

    print(
        "\n[bold green]Question 3.2 Training Gradient Accumulation Training Loop (15pts)[/bold green]"
    )
    
    if os.path.exists("./data/gradient_accumulation.pth") == False:
        
        model, val_loss, train_loss = train_ga(
            model,
            dataloader,
            num_epochs=num_epochs,
            accumulation_steps=gradient_accumulation,
        )

        torch.save(model.state_dict(), "./data/gradient_accumulation.pth")
    else:
        print("Gradient Accumulation Model already trained")

    if os.path.exists("./data/standard_model.pt") and os.path.exists("./data/gradient_accumulation.pth"):
        # Model Parameters
        d_model = 512
        encoder_layers = 6
        decoder_layers = 6
        dim_feedforward = 2048
        num_heads = 8
        dropout_p = 0.2
        num_epochs = 20

        random_sentence_count = 20000
        batch_size = 32

        dataset, dataloader, vocabulary = process_data(
            batch_size, random_sentence_count
        )

        # Load the state dict from the checkpoint
        checkpoint = torch.load("./data/standard_model.pt")

        # Create a new model with the correct vocab size
        correct_vocab_size = checkpoint["embedding.weight"].size(0)
        model = TransformerModel(
            correct_vocab_size,
            d_model,
            encoder_layers,
            decoder_layers,
            dim_feedforward,
            num_heads,
            dropout_p,
        )

        input_sentences = [
            "Where are you",
            "How are you doing",
            "I am doing great how about you",
        ]

        max_length = 5
        start_symbol = vocabulary.word2index["<SOS>"]
        k = 5

        tokenized_input_sentences = []

        # Preprocess the input sentences
        for sentence in input_sentences:
            sentence = clear_punctuation(sentence)
            tokens = sentence.split()
            tokens.append("<EOS>")
            tokenized_input_sentences.append(tokens)

        # Convert the input sentences to tensors
        input_sequence_1 = torch.tensor(
            [vocabulary.word2index[word] for word in tokenized_input_sentences[0]]
        ).unsqueeze(0)
        input_sequence_2 = torch.tensor(
            [vocabulary.word2index[word] for word in tokenized_input_sentences[1]]
        ).unsqueeze(0)
        input_sequence_3 = torch.tensor(
            [vocabulary.word2index[word] for word in tokenized_input_sentences[2]]
        ).unsqueeze(0)

        evaluation_model(
            model,
            input_sequence_1,
            input_sequence_2,
            input_sequence_3,
            "Standard Model",
            vocabulary,
        )

        gradient_accumulation = 32
        batch_size = 32
        random_sentence_count = 30000
        d_model = 512
        dropout_p = 0.3

        dataset, dataloader, vocabulary = process_data(
            batch_size, random_sentence_count
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda x: collate_fn(x, vocabulary.word2index["<PAD>"]),
        )

        # Load the state dict from the checkpoint
        checkpoint = torch.load("./data/gradient_accumulation.pth")

        # Create a new model with the correct vocab size
        correct_vocab_size = checkpoint["embedding.weight"].size(0)
        model = TransformerModel(
            correct_vocab_size,
            d_model,
            encoder_layers,
            decoder_layers,
            dim_feedforward,
            num_heads,
            dropout_p,
        )

        # Load the state dict into the new model
        model.load_state_dict(checkpoint)

        evaluation_model(
            model,
            input_sequence_1,
            input_sequence_2,
            input_sequence_3,
            "Gradient Accumulation Model",
            vocabulary,
        )
