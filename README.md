# TV Script Generation

In this project, we'll generate our own [Seinfeld](https://en.wikipedia.org/wiki/Seinfeld) TV scripts using RNNs.  

We'll be using part of the [Seinfeld dataset](https://www.kaggle.com/thec03u5/seinfeld-chronicles#scripts.csv) of scripts from 9 seasons.  The Neural Network will generate a new ,"fake" TV script, based on patterns it recognizes in this training data.

## Get the Data

The data is in `./data/Seinfeld_Scripts.txt`.


```python
# load in data
import helper
data_dir = './data/Seinfeld_Scripts.txt'
text = helper.load_data(data_dir)
```

## Explore the Data
Play around with `view_line_range` to view different parts of the data. This will give us a sense of the data. For example it is all lowercase text, and each new line of dialogue is separated by a newline character `\n`.


```python
view_line_range = (0, 10)
import numpy as np

print('Dataset Stats')
print('Roughly the number of unique words: {}'.format(
    len({word: None for word in text.split()})))

lines = text.split('\n')
print('Number of lines: {}'.format(len(lines)))
word_count_line = [len(line.split()) for line in lines]
print('Average number of words in each line: {}'.format(
    np.average(word_count_line)))

print()
print('The lines {} to {}:'.format(*view_line_range))
print('\n'.join(text.split('\n')[view_line_range[0]:view_line_range[1]]))
```

    Dataset Stats
    Roughly the number of unique words: 46367
    Number of lines: 109233
    Average number of words in each line: 5.544240293684143
    
    The lines 0 to 10:
    jerry: do you know what this is all about? do you know, why were here? to be out, this is out...and out is one of the single most enjoyable experiences of life. people...did you ever hear people talking about we should go out? this is what theyre talking about...this whole thing, were all out now, no one is home. not one person here is home, were all out! there are people trying to find us, they dont know where we are. (on an imaginary phone) did you ring?, i cant find him. where did he go? he didnt tell me where he was going. he must have gone out. you wanna go out you get ready, you pick out the clothes, right? you take the shower, you get all ready, get the cash, get your friends, the car, the spot, the reservation...then youre standing around, what do you do? you go we gotta be getting back. once youre out, you wanna get back! you wanna go to sleep, you wanna get up, you wanna go out again tomorrow, right? where ever you are in life, its my feeling, youve gotta go. 
    
    jerry: (pointing at georges shirt) see, to me, that button is in the worst possible spot. the second button literally makes or breaks the shirt, look at it. its too high! its in no-mans-land. you look like you live with your mother. 
    
    george: are you through? 
    
    jerry: you do of course try on, when you buy? 
    
    george: yes, it was purple, i liked it, i dont actually recall considering the buttons. 
    


---
## Implement Pre-processing Functions
The first thing to do to any dataset is pre-processing.  Implement the following pre-processing functions below:
- Lookup Table
- Tokenize Punctuation

### Lookup Table
To create a word embedding, we need to transform the words to ids.  
This function creates two dictionaries:
- Dictionary to go from the words to an id, we'll call `vocab_to_int`
- Dictionary to go from the id to word, we'll call `int_to_vocab`
and returns these dictionaries in the following **tuple** `(vocab_to_int, int_to_vocab)`


```python
import problem_unittests as tests
from collections import Counter, defaultdict
d = defaultdict(lambda: np.ndarray(0))

def create_lookup_tables(text):
    """
    Create lookup tables for vocabulary
    :param text: The text of tv scripts split into words
    :return: A tuple of dicts (vocab_to_int, int_to_vocab)
    """
    # sort by word frequency
    word_counts = Counter(text)
    sorted_words = sorted(word_counts, key=word_counts.get, reverse=True)
    # create lookup dicts
    vocab_to_int = {w: i for i, w in enumerate(sorted_words)}
    int_to_vocab = {i: w for w, i in vocab_to_int.items()}
    # return tuple
    return (vocab_to_int, int_to_vocab)


tests.test_create_lookup_tables(create_lookup_tables)
```

    Tests Passed


### Tokenize Punctuation
We'll be splitting the script into a word array using spaces as delimiters.  However, punctuations like periods and exclamation marks can create multiple ids for the same word. For example, "bye" and "bye!" would generate two different word ids.

Implement the function `token_lookup` to return a dict that will be used to tokenize symbols like "!" into "||Exclamation_Mark||".  Create a dictionary for the following symbols where the symbol is the key and value is the token:
- Period ( **.** )
- Comma ( **,** )
- Quotation Mark ( **"** )
- Semicolon ( **;** )
- Exclamation mark ( **!** )
- Question mark ( **?** )
- Left Parentheses ( **(** )
- Right Parentheses ( **)** )
- Dash ( **-** )
- Return ( **\n** )

This dictionary will be used to tokenize the symbols and add the delimiter (space) around it.  This separates each symbols as its own word, making it easier for the neural network to predict the next word. Make sure you don't use a value that could be confused as a word; for example, instead of using the value "dash", try using something like "||dash||".


```python
def token_lookup():
    """
    Generate a dict to turn punctuation into a token.
    :return: Tokenized dictionary where the key is the punctuation and the value is the token
    """
    tokens = {
        '.': '<period>',
        ',': '<comma>',
        '"': '<quotation_Mark>',
        ';': '<semicolon>',
        '!': '<exclamation_mark>',
        '?': '<question_mark>',
        '(': '<left_parentheses>',
        ')': '<right_parentheses>',
        '-': '<dash>',
        '\n': '<return>'
    }
    return tokens


tests.test_tokenize(token_lookup)
```

    Tests Passed


## Pre-process all the data and save it

Running the code cell below will pre-process all the data and save it to file. You're encouraged to lok at the code for `preprocess_and_save_data` in the `helpers.py` file to see what it's doing in detail, but you do not need to change this code.


```python
# pre-process training data
helper.preprocess_and_save_data(data_dir, token_lookup, create_lookup_tables)
```

# Check Point
This is your first checkpoint. If you ever decide to come back to this notebook or have to restart the notebook, you can start from here. The preprocessed data has been saved to disk.


```python
import helper
import problem_unittests as tests

int_text, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()
```

#
'/'
# Build the Neural Network
In this section, you'll build the components necessary to build an RNN by implementing the RNN Module and forward and backpropagation functions.

### Check Access to GPU


```python
import torch

# Check for a GPU
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('No GPU found. Please use a GPU to train your neural network.')
```

    No GPU found. Please use a GPU to train your neural network.


## Input
Let's start with the preprocessed input data. We'll use [TensorDataset](http://pytorch.org/docs/master/data.html#torch.utils.data.TensorDataset) to provide a known format to our dataset; in combination with [DataLoader](http://pytorch.org/docs/master/data.html#torch.utils.data.DataLoader), it will handle batching, shuffling, and other dataset iteration functions.

You can create data with TensorDataset by passing in feature and target tensors. Then create a DataLoader as usual.
```
data = TensorDataset(feature_tensors, target_tensors)
data_loader = torch.utils.data.DataLoader(data, 
                                          batch_size=batch_size)
```

### Batching
Implement the `batch_data` function to batch `words` data into chunks of size `batch_size` using the `TensorDataset` and `DataLoader` classes.

>You can batch words using the DataLoader, but it will be up to you to create `feature_tensors` and `target_tensors` of the correct size and content for a given `sequence_length`.

For example, say we have these as input:
```
words = [1, 2, 3, 4, 5, 6, 7]
sequence_length = 4
```

Your first `feature_tensor` should contain the values:
```
[1, 2, 3, 4]
```
And the corresponding `target_tensor` should just be the next "word"/tokenized word value:
```
5
```
This should continue with the second `feature_tensor`, `target_tensor` being:
```
[2, 3, 4, 5]  # features
6             # target
```


```python
from torch.utils.data import TensorDataset, DataLoader


def batch_data(words, sequence_length, batch_size):
    """
    Batch the neural network data using DataLoader
    :param words: The word ids of the TV scripts
    :param sequence_length: The sequence length of each batch
    :param batch_size: The size of each batch; the number of sequences in a batch
    :return: DataLoader with batched data
    """
    # TODO: Implement function
    features = [
        words[i:i + sequence_length]
        for i in range(len(words) - sequence_length)
    ]
    targets = [
        words[i + sequence_length]
        for i in range(len(words) - sequence_length)
    ]

    data = TensorDataset(torch.Tensor(features), torch.Tensor(targets))
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    # return a dataloader
    return data_loader


# there is no test for this function, but you are encouraged to create
# print statements and tests of your own
dl = batch_data([1, 2, 3, 4, 5, 6, 7, 8], 2, 3)
data_iter = iter(dl)

for data, target in data_iter:
    print(data, target)
```

    tensor([[ 4.,  5.],
            [ 1.,  2.],
            [ 6.,  7.]]) tensor([ 6.,  3.,  8.])
    tensor([[ 5.,  6.],
            [ 3.,  4.],
            [ 2.,  3.]]) tensor([ 7.,  5.,  4.])


### Test your dataloader 

You'll have to modify this code to test a batching function, but it should look fairly similar.

Below, we're generating some test text data and defining a dataloader using the function you defined, above. Then, we are getting some sample batch of inputs `sample_x` and targets `sample_y` from our dataloader.

Your code should return something like the following (likely in a different order, if you shuffled your data):

```
torch.Size([10, 5])
tensor([[ 28,  29,  30,  31,  32],
        [ 21,  22,  23,  24,  25],
        [ 17,  18,  19,  20,  21],
        [ 34,  35,  36,  37,  38],
        [ 11,  12,  13,  14,  15],
        [ 23,  24,  25,  26,  27],
        [  6,   7,   8,   9,  10],
        [ 38,  39,  40,  41,  42],
        [ 25,  26,  27,  28,  29],
        [  7,   8,   9,  10,  11]])

torch.Size([10])
tensor([ 33,  26,  22,  39,  16,  28,  11,  43,  30,  12])
```

### Sizes
Your sample_x should be of size `(batch_size, sequence_length)` or (10, 5) in this case and sample_y should just have one dimension: batch_size (10). 

### Values

You should also notice that the targets, sample_y, are the *next* value in the ordered test_text data. So, for an input sequence `[ 28,  29,  30,  31,  32]` that ends with the value `32`, the corresponding output should be `33`.


```python
# test dataloader

test_text = range(50)
t_loader = batch_data(test_text, sequence_length=5, batch_size=10)

data_iter = iter(t_loader)
sample_x, sample_y = data_iter.next()

print(sample_x.shape)
print(sample_x)
print()
print(sample_y.shape)
print(sample_y)
```

    torch.Size([10, 5])
    tensor([[ 40.,  41.,  42.,  43.,  44.],
            [ 16.,  17.,  18.,  19.,  20.],
            [ 33.,  34.,  35.,  36.,  37.],
            [ 11.,  12.,  13.,  14.,  15.],
            [ 34.,  35.,  36.,  37.,  38.],
            [ 28.,  29.,  30.,  31.,  32.],
            [  4.,   5.,   6.,   7.,   8.],
            [ 19.,  20.,  21.,  22.,  23.],
            [ 18.,  19.,  20.,  21.,  22.],
            [  7.,   8.,   9.,  10.,  11.]])
    
    torch.Size([10])
    tensor([ 45.,  21.,  38.,  16.,  39.,  33.,   9.,  24.,  23.,  12.])


---
## Build the Neural Network
Implement an RNN using PyTorch's [Module class](http://pytorch.org/docs/master/nn.html#torch.nn.Module). You may choose to use a GRU or an LSTM. To complete the RNN, you'll have to implement the following functions for the class:
 - `__init__` - The initialize function. 
 - `init_hidden` - The initialization function for an LSTM/GRU hidden state
 - `forward` - Forward propagation function.
 
The initialize function should create the layers of the neural network and save them to the class. The forward propagation function will use these layers to run forward propagation and generate an output and a hidden state.

**The output of this model should be the *last* batch of word scores** after a complete sequence has been processed. That is, for each input sequence of words, we only want to output the word scores for a single, most likely, next word.

### Hints

1. Make sure to stack the outputs of the lstm to pass to your fully-connected layer, you can do this with `lstm_output = lstm_output.contiguous().view(-1, self.hidden_dim)`
2. You can get the last batch of word scores by shaping the output of the final, fully-connected layer like so:

```
# reshape into (batch_size, seq_length, output_size)
output = output.view(batch_size, -1, self.output_size)
# get last batch
out = output[:, -1]
```


```python
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.5):
        """
        Initialize the PyTorch RNN Module
        :param vocab_size: The number of input dimensions of the neural network (the size of the vocabulary)
        :param output_size: The number of output dimensions of the neural network
        :param embedding_dim: The size of embeddings, should you choose to use them
        :param hidden_dim: The size of the hidden layer outputs
        :param dropout: dropout to add in between LSTM/GRU layers
        """
        super(RNN, self).__init__()
        # TODO: Implement function

        # set class variables
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        # define model layers

        # embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)

        # # an embedding layer for outputs
        # self.out_embedding = nn.Embedding(vocab_size, embedding_dim)

        # Initialize embedding tables with uniform distribution
        self.embedding.weight.data.uniform_(-1, 1)

        # dropout layer
        self.dropout = nn.Dropout(0.3)

        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_size)

    def forward(self, nn_input, hidden):
        """
        Forward propagation of the neural network
        :param nn_input: The input to the neural network
        :param hidden: The hidden state
        :return: Two Tensors, the output of the neural network and the latest hidden state
        """
        # TODO: Implement function
        batch_size = nn_input.size(0)

        # embeddings and lstm_out
        nn_input = nn_input.long()
        embeds = self.embedding(nn_input)
        lstm_out, hidden = self.lstm(embeds, hidden)

        # stack up lstm outputs
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        # dropout and fully-connected layer
        fc1_output = self.fc1(lstm_out)
        dropout_output = self.dropout(fc1_output)
        fc2_output = self.fc2(dropout_output)

        # reshape to be batch_size first
        output = fc2_output.view(batch_size, -1, self.output_size)
        output = output[:, -1]  # get last batch of labels

        # return one batch of output word scores and the hidden state
        return output, hidden

    def init_hidden(self, batch_size):
        '''
        Initialize the hidden state of an LSTM/GRU
        :param batch_size: The batch_size of the hidden state
        :return: hidden state of dims (n_layers, batch_size, hidden_dim)
        '''
        # Implement function

        # initialize hidden state with zero weights, and move to GPU if available
        weight = next(self.parameters()).data

        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())

        return hidden


tests.test_rnn(RNN, train_on_gpu)
```

    Tests Passed


### Define forward and backpropagation

Use the RNN class you implemented to apply forward and back propagation. This function will be called, iteratively, in the training loop as follows:
```
loss = forward_back_prop(decoder, decoder_optimizer, criterion, inp, target)
```

And it should return the average loss over a batch and the hidden state returned by a call to `RNN(inp, hidden)`. Recall that you can get this loss by computing it, as usual, and calling `loss.item()`.

**If a GPU is available, you should move your data to that GPU device, here.**


```python
def forward_back_prop(rnn, optimizer, criterion, inp, target, hidden):
    """
    Forward and backward propagation on the neural network
    :param decoder: The PyTorch Module that holds the neural network
    :param decoder_optimizer: The PyTorch optimizer for the neural network
    :param criterion: The PyTorch loss function
    :param inp: A batch of input to the neural network
    :param target: The target output for the batch of input
    :return: The loss and the latest hidden state Tensor
    """

    # move data to GPU, if available
    if (train_on_gpu):
        rnn, inp, target = rnn.cuda(), inp.cuda(), target.cuda()

    # perform backpropagation and optimization

    hidden = tuple([each.data for each in hidden])
    optimizer.zero_grad()

    output, hidden = rnn(inp, hidden)

    loss = criterion(output.squeeze(), target.long())
    loss.backward()

    nn.utils.clip_grad_norm_(rnn.parameters(), 5)

    optimizer.step()

    # return the loss over a batch and the hidden state produced by our model
    return loss.item(), hidden


# These tests aren't completely extensive.
# they are here to act as general checks on the expected outputs of your functions
tests.test_forward_back_prop(RNN, forward_back_prop, train_on_gpu)
```

    Tests Passed


## Neural Network Training

With the structure of the network complete and data ready to be fed in the neural network, it's time to train it.

### Train Loop

The training loop is implemented for you in the `train_decoder` function. This function will train the network over all the batches for the number of epochs given. The model progress will be shown every number of batches. This number is set with the `show_every_n_batches` parameter. You'll set this parameter along with other parameters in the next section.


```python
def train_rnn(rnn, batch_size, optimizer, criterion, n_epochs, show_every_n_batches=100):
    batch_losses = []

    rnn.train()

    print("Training for %d epoch(s)..." % n_epochs)
    for epoch_i in range(1, n_epochs + 1):

        # initialize hidden state
        hidden = rnn.init_hidden(batch_size)

        for batch_i, (inputs, labels) in enumerate(train_loader, 1):

            # make sure you iterate over completely full batches, only
            n_batches = len(train_loader.dataset) // batch_size
            if (batch_i > n_batches):
                break

            # forward, back prop
            loss, hidden = forward_back_prop(rnn, optimizer, criterion, inputs, labels, hidden)
            # record loss
            batch_losses.append(loss)

            # printing loss stats
            if batch_i % show_every_n_batches == 0:
                print('Epoch: {:>4}/{:<4}  Loss: {}\n'.format(
                    epoch_i, n_epochs, np.average(batch_losses)))
                batch_losses = []

    # returns a trained rnn
    return rnn
```

### Hyperparameters

Set and train the neural network with the following parameters:
- Set `sequence_length` to the length of a sequence.
- Set `batch_size` to the batch size.
- Set `num_epochs` to the number of epochs to train for.
- Set `learning_rate` to the learning rate for an Adam optimizer.
- Set `vocab_size` to the number of unique tokens in our vocabulary.
- Set `output_size` to the desired size of the output.
- Set `embedding_dim` to the embedding dimension; smaller than the vocab_size.
- Set `hidden_dim` to the hidden dimension of your RNN.
- Set `n_layers` to the number of layers/cells in your RNN.
- Set `show_every_n_batches` to the number of batches at which the neural network should print progress.

If the network isn't getting the desired results, tweak these parameters and/or the layers in the `RNN` class.


```python
# Data params
# Sequence Length
sequence_length = 10  # of words in a sequence
# Batch Size
batch_size = 200

# data loader - do not change
train_loader = batch_data(int_text, sequence_length, batch_size)
```


```python
# Training parameters
# Number of Epochs
num_epochs = 20
# Learning Rate
learning_rate = 0.001

# Model parameters
# Vocab size
vocab_size = len(vocab_to_int)
# Output size
output_size = len(vocab_to_int)
# Embedding Dimension
embedding_dim = 400
# Hidden Dimension
hidden_dim = 512
# Number of RNN Layers
n_layers = 2

# Show stats for every n number of batches
show_every_n_batches = 500
```

### Train
In the next cell, you'll train the neural network on the pre-processed data.  If you have a hard time getting a good loss, you may consider changing your hyperparameters. In general, you may get better results with larger hidden and n_layer dimensions, but larger models take a longer time to train. 
> **You should aim for a loss less than 3.5.** 

You should also experiment with different sequence lengths, which determine the size of the long range dependencies that a model can learn.


```python
# create model and move to gpu if available
rnn = RNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.5)
if train_on_gpu:
    rnn.cuda()

# defining loss and optimization functions for training
optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# training the model
trained_rnn = train_rnn(rnn, batch_size, optimizer, criterion, num_epochs, show_every_n_batches)

# saving the trained model
helper.save_model('./save/trained_rnn', trained_rnn)
print('Model Trained and Saved')
```

    Training for 20 epoch(s)...
    Epoch:    1/20    Loss: 5.2487041263580325
    
    Epoch:    1/20    Loss: 4.678163821697235
    
    Epoch:    1/20    Loss: 4.498290432453156
    
    Epoch:    1/20    Loss: 4.377445560455322
    
    Epoch:    1/20    Loss: 4.319503573417664
    
    Epoch:    1/20    Loss: 4.245671498775482
    
    Epoch:    1/20    Loss: 4.212081611633301
    
    Epoch:    1/20    Loss: 4.165680992603302
    
    Epoch:    2/20    Loss: 4.08266386911273
    
    Epoch:    2/20    Loss: 4.006602307319641
    
    Epoch:    2/20    Loss: 3.980281336784363
    
    Epoch:    2/20    Loss: 3.9741369547843934
    
    Epoch:    2/20    Loss: 3.9842675704956054
    
    Epoch:    2/20    Loss: 3.961428580760956
    
    Epoch:    2/20    Loss: 3.9555888991355896
    
    Epoch:    2/20    Loss: 3.930756108760834
    
    Epoch:    3/20    Loss: 3.8636351394156616
    
    Epoch:    3/20    Loss: 3.7959830813407898
    
    Epoch:    3/20    Loss: 3.7927556352615355
    
    Epoch:    3/20    Loss: 3.805548355579376
    
    Epoch:    3/20    Loss: 3.8070667872428894
    
    Epoch:    3/20    Loss: 3.7980847477912905
    
    Epoch:    3/20    Loss: 3.8010243072509766
    
    Epoch:    3/20    Loss: 3.7983804059028627
    
    Epoch:    4/20    Loss: 3.711127464224895
    
    Epoch:    4/20    Loss: 3.660415386199951
    
    Epoch:    4/20    Loss: 3.6758184885978697
    
    Epoch:    4/20    Loss: 3.665699333190918
    
    Epoch:    4/20    Loss: 3.688417768955231
    
    Epoch:    4/20    Loss: 3.7016181626319886
    
    Epoch:    4/20    Loss: 3.668921979904175
    
    Epoch:    4/20    Loss: 3.686590579509735
    
    Epoch:    5/20    Loss: 3.6137035434444744
    
    Epoch:    5/20    Loss: 3.555068018913269
    
    Epoch:    5/20    Loss: 3.5597557487487794
    
    Epoch:    5/20    Loss: 3.564852885723114
    
    Epoch:    5/20    Loss: 3.596783277988434
    
    Epoch:    5/20    Loss: 3.604326102256775
    
    Epoch:    5/20    Loss: 3.5805342526435853
    
    Epoch:    5/20    Loss: 3.6235752902030947
    
    Epoch:    6/20    Loss: 3.5324046194553373
    
    Epoch:    6/20    Loss: 3.4690963106155395
    
    Epoch:    6/20    Loss: 3.489684805870056
    
    Epoch:    6/20    Loss: 3.4923943128585817
    
    Epoch:    6/20    Loss: 3.513419545173645
    
    Epoch:    6/20    Loss: 3.5104973306655882
    
    Epoch:    6/20    Loss: 3.5478053193092345
    
    Epoch:    6/20    Loss: 3.5449086022377014
    
    Epoch:    7/20    Loss: 3.4574674611290295
    
    Epoch:    7/20    Loss: 3.405350127696991
    
    Epoch:    7/20    Loss: 3.4138274636268617
    
    Epoch:    7/20    Loss: 3.4436375613212586
    
    Epoch:    7/20    Loss: 3.459013870716095
    
    Epoch:    7/20    Loss: 3.4492683486938476
    
    Epoch:    7/20    Loss: 3.4684084191322326
    
    Epoch:    7/20    Loss: 3.4800872402191163
    
    Epoch:    8/20    Loss: 3.406979504227638
    
    Epoch:    8/20    Loss: 3.3523414883613585
    
    Epoch:    8/20    Loss: 3.3723497595787046
    
    Epoch:    8/20    Loss: 3.3927461705207826
    
    Epoch:    8/20    Loss: 3.387379436016083
    
    Epoch:    8/20    Loss: 3.4186446528434753
    
    Epoch:    8/20    Loss: 3.4056297087669374
    
    Epoch:    8/20    Loss: 3.4240229568481446
    
    Epoch:    9/20    Loss: 3.3463318196435767
    
    Epoch:    9/20    Loss: 3.312673158168793
    
    Epoch:    9/20    Loss: 3.32148820066452
    
    Epoch:    9/20    Loss: 3.3288730278015137
    
    Epoch:    9/20    Loss: 3.3523661551475525
    
    Epoch:    9/20    Loss: 3.3637926788330077
    
    Epoch:    9/20    Loss: 3.3486119751930237
    
    Epoch:    9/20    Loss: 3.386474793434143
    
    Epoch:   10/20    Loss: 3.319483475635449
    
    Epoch:   10/20    Loss: 3.242880956172943
    
    Epoch:   10/20    Loss: 3.2704734616279603
    
    Epoch:   10/20    Loss: 3.2899507269859316
    
    Epoch:   10/20    Loss: 3.319627736091614
    
    Epoch:   10/20    Loss: 3.323590055465698
    
    Epoch:   10/20    Loss: 3.333856041431427
    
    Epoch:   10/20    Loss: 3.3314898285865784
    
    Epoch:   11/20    Loss: 3.26687464689215
    
    Epoch:   11/20    Loss: 3.2166823372840883
    
    Epoch:   11/20    Loss: 3.2339503326416015
    
    Epoch:   11/20    Loss: 3.2548137822151184
    
    Epoch:   11/20    Loss: 3.269361214160919
    
    Epoch:   11/20    Loss: 3.28704562330246
    
    Epoch:   11/20    Loss: 3.3035574407577513
    
    Epoch:   11/20    Loss: 3.3101003398895266
    
    Epoch:   12/20    Loss: 3.2252686535318693
    
    Epoch:   12/20    Loss: 3.168487340450287
    
    Epoch:   12/20    Loss: 3.1992998509407045
    
    Epoch:   12/20    Loss: 3.2163495984077453
    
    Epoch:   12/20    Loss: 3.2367290816307066
    
    Epoch:   12/20    Loss: 3.2457709984779357
    
    Epoch:   12/20    Loss: 3.2584562187194823
    
    Epoch:   12/20    Loss: 3.292267857551575
    
    Epoch:   13/20    Loss: 3.201848326375087
    
    Epoch:   13/20    Loss: 3.146637210845947
    
    Epoch:   13/20    Loss: 3.1653626360893248
    
    Epoch:   13/20    Loss: 3.1802898864746094
    
    Epoch:   13/20    Loss: 3.20764964723587
    
    Epoch:   13/20    Loss: 3.214337046146393
    
    Epoch:   13/20    Loss: 3.2218326044082644
    
    Epoch:   13/20    Loss: 3.244129172325134
    
    Epoch:   14/20    Loss: 3.1728610006471474
    
    Epoch:   14/20    Loss: 3.119767719745636
    
    Epoch:   14/20    Loss: 3.1461305441856386
    
    Epoch:   14/20    Loss: 3.1563744139671326
    
    Epoch:   14/20    Loss: 3.166397168636322
    
    Epoch:   14/20    Loss: 3.179938488006592
    
    Epoch:   14/20    Loss: 3.217538221359253
    
    Epoch:   14/20    Loss: 3.2266262340545655
    
    Epoch:   15/20    Loss: 3.1429059331615767
    
    Epoch:   15/20    Loss: 3.088300848007202
    
    Epoch:   15/20    Loss: 3.1119890117645266
    
    Epoch:   15/20    Loss: 3.1439440813064574
    
    Epoch:   15/20    Loss: 3.152367917060852
    
    Epoch:   15/20    Loss: 3.1662757306098936
    
    Epoch:   15/20    Loss: 3.176982677459717
    
    Epoch:   15/20    Loss: 3.1893785581588747
    
    Epoch:   16/20    Loss: 3.1212863134841125
    
    Epoch:   16/20    Loss: 3.0701546068191528
    
    Epoch:   16/20    Loss: 3.0951798276901243
    
    Epoch:   16/20    Loss: 3.121554039478302
    
    Epoch:   16/20    Loss: 3.113096643447876
    
    Epoch:   16/20    Loss: 3.145941666126251
    
    Epoch:   16/20    Loss: 3.154481608390808
    
    Epoch:   16/20    Loss: 3.177802718162537
    
    Epoch:   17/20    Loss: 3.092607163886229
    
    Epoch:   17/20    Loss: 3.0546866788864135
    
    Epoch:   17/20    Loss: 3.0631416511535643
    
    Epoch:   17/20    Loss: 3.0813032870292663
    
    Epoch:   17/20    Loss: 3.118807766914368
    
    Epoch:   17/20    Loss: 3.112222282409668
    
    Epoch:   17/20    Loss: 3.1376626152992246
    
    Epoch:   17/20    Loss: 3.1521453471183776
    
    Epoch:   18/20    Loss: 3.08064982915918
    
    Epoch:   18/20    Loss: 3.022375586986542
    
    Epoch:   18/20    Loss: 3.045244505882263
    
    Epoch:   18/20    Loss: 3.066993263244629
    
    Epoch:   18/20    Loss: 3.092087601184845
    
    Epoch:   18/20    Loss: 3.0984520306587218
    
    Epoch:   18/20    Loss: 3.1192490029335023
    
    Epoch:   18/20    Loss: 3.1356327319145203
    
    Epoch:   19/20    Loss: 3.0610365892450013
    
    Epoch:   19/20    Loss: 3.000286376953125
    
    Epoch:   19/20    Loss: 3.0519462790489196
    
    Epoch:   19/20    Loss: 3.0544234991073607
    
    Epoch:   19/20    Loss: 3.064984381198883
    
    Epoch:   19/20    Loss: 3.0739823560714723
    
    Epoch:   19/20    Loss: 3.100578772544861
    
    Epoch:   19/20    Loss: 3.103016511440277
    
    Epoch:   20/20    Loss: 3.0437896219392617
    
    Epoch:   20/20    Loss: 2.987021891117096
    
    Epoch:   20/20    Loss: 3.015021628379822
    
    Epoch:   20/20    Loss: 3.0308129358291627
    
    Epoch:   20/20    Loss: 3.0468586773872377
    
    Epoch:   20/20    Loss: 3.067536421775818
    
    Epoch:   20/20    Loss: 3.081701693058014
    
    Epoch:   20/20    Loss: 3.0903398933410644
    


    /opt/conda/lib/python3.6/site-packages/torch/serialization.py:193: UserWarning: Couldn't retrieve source code for container of type RNN. It won't be checked for correctness upon loading.
      "type " + obj.__name__ + ". It won't be checked "


    Model Trained and Saved


### Deciding on the model hyperparameters

As a first step, I mimicked what we have done over excersise projects. Then I started fiddling with `sequence_length`, `hidden_dim` and `learning_rate`. 
My first aim was to make the network run faster and learn faster while keeping the parameters high enough for it to cover all complexities of natural language, specially a show script. So, I tried to keep sequence_length as small as possible but not too small. I worked around 4,6,7,8,10, and even 20. (Midway into parameter tuning, I settled on 6 as a good but small number)
To keep my experiences short, I kept period of showing loss as small as 20 for the start and later I used 100.
`batch_size` has been a parameter for me to train faster in the expense of learning quality at each step, so finding a sweet spot was rather difficult. I used either 128 or 256 in most cases.
Moreover, I kept my epochs rather short, for the time of tuning parameters. And I did not modify `embedding_dim` much in the first runs of tuning.
The following parameters were actually quite straight forward:
- `num_epochs` Need to go as long as the network diverges!
- `learning_rate` basically used 0.001 since 0.01 and 0.0001 did not perform same or any better
- `vocab_size` number of distinct int_words
- `output_size` same as vocab_size since it is the predicted word
- `n_layers` kept it to 2 and then upped it to 3.
- `show_every_n_batches` used 20 and 100

Midway, with the following tuning I got to 3.7~3.9 loss.
```python
sequence_length = 6
batch_size = 256
num_epochs = 10
learning_rate = 0.001
vocab_size = len(vocab_to_int)
output_size = len(vocab_to_int)
embedding_dim = 300
hidden_dim = 128
n_layers = 3
show_every_n_batches = 100
```
So, I had to add to the complexity of network in the expense of training speed!
```python
sequence_length = 10 # upped from 6
batch_size = 128 # halved from 256
num_epochs = 20 # increased with hope of good convergance in double epochs
learning_rate = 0.001 # kept the same
vocab_size = len(vocab_to_int) # kept the same
output_size = len(vocab_to_int) # kept the same
embedding_dim = 300 # kept the same
hidden_dim = 128 # kept the same
n_layers = 3 # kept the same
show_every_n_batches = 100 # kept the same
```
But, I did not get so much improvment to my previous network. So in another attempt I increased other parameters of network:
```python
sequence_length = 10 # kept the same
batch_size = 200 # increased from 128
num_epochs = 20 # kept the same
learning_rate = 0.001 # kept the same
vocab_size = len(vocab_to_int) # kept the same
output_size = len(vocab_to_int) # kept the same
embedding_dim = 400 # increased from 300
hidden_dim = 512 # increased from 128
n_layers = 2 # reduced since I increased hidden memory
show_every_n_batches = 500 # increase to lessen the printed information (need to observe the bigger picture)
```
And it looked promising. (3rd epoch: \~3.8, and then 4th epoch: \~3.7)
Which is why I let the network to train with these parameters for all 20 epochs.

---
# Checkpoint

After running the above training cell, your model will be saved by name, `trained_rnn`, and if you save your notebook progress, **you can pause here and come back to this code at another time**. You can resume your progress by running the next cell, which will load in our word:id dictionaries _and_ load in your saved model by name!


```python
import torch
import helper
import problem_unittests as tests

_, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()
trained_rnn = helper.load_model('./save/trained_rnn')
```

## Generate TV Script
With the network trained and saved, you'll use it to generate a new, "fake" Seinfeld TV script in this section.

### Generate Text
To generate the text, the network needs to start with a single word and repeat its predictions until it reaches a set length. You'll be using the `generate` function to do this. It takes a word id to start with, `prime_id`, and generates a set length of text, `predict_len`. Also note that it uses topk sampling to introduce some randomness in choosing the most likely next word, given an output set of word scores!


```python
import torch.nn.functional as F


def generate(rnn, prime_id, int_to_vocab, token_dict, pad_value, predict_len=100):
    """
    Generate text using the neural network
    :param decoder: The PyTorch Module that holds the trained neural network
    :param prime_id: The word id to start the first prediction
    :param int_to_vocab: Dict of word id keys to word values
    :param token_dict: Dict of puncuation tokens keys to puncuation values
    :param pad_value: The value used to pad a sequence
    :param predict_len: The length of text to generate
    :return: The generated text
    """
    rnn.eval()

    # create a sequence (batch_size=1) with the prime_id
    current_seq = np.full((1, sequence_length), pad_value)
    current_seq[-1][-1] = prime_id
    predicted = [int_to_vocab[prime_id]]

    for _ in range(predict_len):
        if train_on_gpu:
            current_seq = torch.LongTensor(current_seq).cuda()
        else:
            current_seq = torch.LongTensor(current_seq)

        # initialize the hidden state
        hidden = rnn.init_hidden(current_seq.size(0))

        # get the output of the rnn
        output, _ = rnn(current_seq, hidden)

        # get the next word probabilities
        p = F.softmax(output, dim=1).data
        if (train_on_gpu):
            p = p.cpu()  # move to cpu

        # use top_k sampling to get the index of the next word
        top_k = 5
        p, top_i = p.topk(top_k)
        top_i = top_i.numpy().squeeze()

        # select the likely next word index with some element of randomness
        p = p.numpy().squeeze()
        word_i = np.random.choice(top_i, p=p / p.sum())

        # retrieve that word from the dictionary
        word = int_to_vocab[word_i]
        predicted.append(word)

        if (train_on_gpu):
            current_seq = current_seq.cpu()  # move to cpu
        # the generated word becomes the next "current sequence" and the cycle can continue
        if train_on_gpu:
            current_seq = current_seq.cpu()
        current_seq = np.roll(current_seq, -1, 1)
        current_seq[-1][-1] = word_i

    gen_sentences = ' '.join(predicted)

    # Replace punctuation tokens
    for key, token in token_dict.items():
        gen_sentences = gen_sentences.replace(' ' + token.lower(), key)
    gen_sentences = gen_sentences.replace('\n ', '\n')
    gen_sentences = gen_sentences.replace('( ', '(')

    # return all the sentences
    return gen_sentences
```

### Generate a New Script
It's time to generate the text. Set `gen_length` to the length of TV script you want to generate and set `prime_word` to one of the following to start the prediction:
- "jerry"
- "elaine"
- "george"
- "kramer"

You can set the prime word to _any word_ in our dictionary, but it's best to start with a name for generating a TV script. (You can also start with any other names you find in the original text file!)


```python
# run the cell multiple times to get different results!
gen_length = 2000  # modify the length to your preference
prime_word = 'newman'  # name for starting the script

pad_word = helper.SPECIAL_WORDS['PADDING']
generated_script = generate(trained_rnn, vocab_to_int[prime_word + ':'],
                            int_to_vocab, token_dict, vocab_to_int[pad_word],
                            gen_length)
print(generated_script)
```

    /opt/conda/lib/python3.6/site-packages/ipykernel_launcher.py:53: UserWarning: RNN module weights are not part of single contiguous chunk of memory. This means they need to be compacted at every call, possibly greatly increasing memory usage. To compact weights again call flatten_parameters().


    newman: birthdays.
    
    elaine: hey!
    
    jerry: hey. i know what i think.
    
    jerry: you know, i really like it
    
    elaine: i don't know.
    
    george: well...
    
    elaine:(to george) you know, i was in the mood. i was trying to get a job interview.
    
    jerry: i don't think so.
    
    george: well, i'm sorry, but i don't know how you can do it. i mean, i don't know how it was.
    
    jerry: well, it's a long story for me, but i'm gonna call him back to my office. i just don't know what to say.
    
    kramer: you know i don't know what it is.
    
    jerry: i think you're getting married.
    
    jerry: oh!
    
    jerry:(to elaine) so, what is that?
    
    kramer: well, i don't know.
    
    kramer: oh, yeah.(to elaine) i don't know what i am, but i don't have to tell you this, i was wondering if i have a big deal cooking.
    
    elaine: what is that noise?
    
    newman: you can't do that. i don't know why they were talking about the whole thing.
    
    jerry: you know, you should be in the hospital to be able to do that.(to jerry) so you have to be the only way you want to go?
    
    jerry: yeah, that's right.
    
    jerry: you don't think so?
    
    george: i don't know, but, i know i was in my apartment, and i have a deal, i got it from me.
    
    jerry:(to elaine) hey, i know what you got here.
    
    elaine: oh, i know you don't know.
    
    jerry: you know, you know, i'm a little nervous about the whole life, and the other.
    
    jerry: i know, but you can't go with me.
    
    kramer: yeah, i think i should.
    
    george: oh!
    
    elaine:(to the drake) excuse me.. you got a big stain on it, you know, the guy is a little temperamental, nedda.
    
    jerry: i thought you were talking about what he did.
    
    jerry:(pointing) you know...
    
    elaine: i think i should.
    
    george:(laughs) i can't go anywhere, i don't know if i can get the receipt.
    
    kramer: oh, yeah...
    
    elaine: what do you mean, i don't know how to do it.
    
    elaine: well, i guess you should call her back with me.
    
    kramer: well i thought it was...
    
    elaine: what? what happened?
    
    george: you got a big meeting?
    
    george:(looking at his watch) oh, my god, you know, i don't even know what you said.
    
    george: i think i should get the machine.(looks around) hey. you know what? i'm a comedian.
    
    jerry: oh, no no no.... i got a date with my wife.
    
    jerry: what about that bavarian cream pie?
    
    jerry: oh, i was in the mood!
    
    elaine:(pause) you mean," i don't know if he was taller.
    
    george: yeah?
    
    george:(smiling, to elaine) i don't know, i'm gonna do it.(to george) what do you say?
    
    jerry: well, i was in a car accident, i was trying to get a job. i can't do this.
    
    elaine: oh, i got a new recliner on the other line...
    
    kramer: yeah, well i don't know if i am ready to do that.(he leaves)
    
    kramer: oh yeah! i got a great cafeteria downstairs.(to jerry) i don't want to see the name of the church.
    
    elaine:(to george) you know i think you could have the same way, jerry, you know, i have a very interesting idea for you.
    
    jerry: i don't understand, i don't have to tell you. but i have a little thing about the pilot.
    
    george: you don't want to know what it is...
    
    jerry: oh, no, no. i was just trying to get my finger in a lifetime of the air.
    
    jerry: oh.
    
    jerry: i don't know, i know, i don't know what it is. i mean, i think it's an emergency, it's the way it is.(jerry looks at the tag.)
    
    george: i don't know what to do. i think it's a little unusual.
    
    jerry: oh yeah.
    
    elaine: oh my god, i'm sorry, but i'm not going to have a little fun with this.
    
    morty: you know, i think you can get some air. i can't believe i was in my apartment.
    
    jerry: i know, i'm not gonna go to the bathroom with the other guy.
    
    elaine:(to george) i can't believe you don't know what i mean.
    
    jerry: well, i guess i should get a cab.
    
    george:(smiling) i don't think so. i don't think so.
    
    george: i don't understand, i can't.
    
    jerry:(to the phone) hey, what are ya' gonna do about the car?
    
    elaine:(smiling) oh, yeah...
    
    george:(interrupting) what is it?
    
    jerry: well i was just curious, i don't know.
    
    elaine:(to the intercom) oh.
    
    jerry:(answering the phone) hello?
    
    elaine: hi, george.
    
    kramer: oh yeah.
    
    elaine:(smiling. kramer)
    
    elaine: hi.
    
    jerry: hi.
    
    george:(to the man) hello, elaine!
    
    jerry: hi, hi.
    
    george: hi.
    
    george:(to jerry) so, you want to see her?
    
    jerry: oh, i don't know.
    
    kramer: oh, you have to do the show.
    
    jerry: oh, no.
    
    elaine: what are you doing?
    
    jerry: i'm a comedian.
    
    jerry:(to jerry) you know, it's a little unusual. it's not like anyone's a little...
    
    elaine:(to elaine) you know what i like to do? i mean, you know, i have to say," oh, i love it.")
    
    jerry: i got it all timed out.(jerry looks in his shoulder)...
    
    elaine:(laughs)
    
    jerry: i know.
    
    george: you got a date with that?
    
    jerry: i don't know. you know, the guy, the whole world has been delayed, the bad world, that was a wild color.
    
    elaine:(still trying to shut up) oh! oh, no- it's not. i'm going to be a member for me.
    
    jerry:(looking at the tv) yeah, yeah..
    
    jerry: you got the card?
    
    george: i can't believe you don't know.
    
    elaine: i don't know.
    
    jerry: well, i don't know.
    
    jerry:(to elaine) what is it?(jerry nods)
    
    jerry: oh, yeah, yeah, i am so nice.
    
    jerry: yeah.
    
    george: you know, i think i may have been a little more tactful.
    
    jerry:(to george) you sure i got this idea.
    
    kramer:(quietly) no no no no no, i'm just gonna get it...
    
    elaine: you know, you should be a little too much.
    
    jerry:(sarcastic) oh, no, no. no, i don't think so. i mean, i know.
    
    kramer: oh my god...(points to jerry)
    
    [setting: the costanza's house]
    
    jerry: i know, i know, you know, you don't have to be able to do it.
    
    jerry: i don't want it.
    
    george: what is she talking about? i was wondering if you were a man, you have to be on the way to get the theatre.
    
    morty: i can't. i can't believe it. i don't have any money.
    
    jerry: well i think you should get it back for me.
    
    kramer: oh! well, i'm sorry.
    
    george: oh, i don't know.(to jerry) i know. i know, i can't believe it...(george enters) i don't know if you could just go out to dinner.
    
    jerry:(to kramer) so what do you do? you know what, i'm gonna do it, and you can get a couple of seats in the middle of the night?
    
    jerry: no i don't have to do.
    
    jerry: oh!
    
    jerry: what are you doing?!
    
    george: you know you really should get a gardener.
    
    elaine: i don't know if you could do that.
    
    george: oh yeah..
    
    kramer:(looks at her watch) you think you're better than me?
    
    jerry: yeah, i know.
    
    kramer: oh, yeah.
    
    jerry: i mean i know. i just wanted to see someone in a situation like this.(laughs)
    
    jerry:(pause) you know, i think you should see that woman...
    
    elaine:(interrupting) i mean, i was hoping that he was going to do something like that.(elaine looks at george)
    
    kramer: hey, hey, you want to be a man, you don't think she was flirting with me? i don't know, i just got a big problem with you.
    
    george: oh, well, i guess i have a little doubt, i have to get


#### Save your favorite scripts

Once you have a script that you like (or find interesting), save it to a text file!


```python
# save script to a text file
f = open("generated_script_1.txt", "w")
f.write(generated_script)
f.close()
```

# The TV Script is Not Perfect
It's ok if the TV script doesn't make perfect sense. It should look like alternating lines of dialogue, here is one such example of a few generated lines.

### Example generated script

>jerry: what about me?
>
>jerry: i don't have to wait.
>
>kramer:(to the sales table)
>
>elaine:(to jerry) hey, look at this, i'm a good doctor.
>
>newman:(to elaine) you think i have no idea of this...
>
>elaine: oh, you better take the phone, and he was a little nervous.
>
>kramer:(to the phone) hey, hey, jerry, i don't want to be a little bit.(to kramer and jerry) you can't.
>
>jerry: oh, yeah. i don't even know, i know.
>
>jerry:(to the phone) oh, i know.
>
>kramer:(laughing) you know...(to jerry) you don't know.

You can see that there are multiple characters that say (somewhat) complete sentences, but it doesn't have to be perfect! It takes quite a while to get good results, and often, you'll have to use a smaller vocabulary (and discard uncommon words), or get more data.  The Seinfeld dataset is about 3.4 MB, which is big enough for our purposes; for script generation you'll want more than 1 MB of text, generally. 

