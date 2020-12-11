#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 23 18:37:40 2020

@author: https://www.tensorflow.org/tutorials/text/text_generation

"""

#%%
import tensorflow as tf
#%%
import numpy as np
import os
import time

#%%
path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

#%%
# Read, then decode for py2 compat.
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')

#%%
# The unique characters in the file
vocab = sorted(set(text))
char2idx = {u:i for i, u in enumerate(vocab)}   #字典推导式
idx2char = np.array(vocab)

text_as_int = np.array([char2idx[c] for c in text])
#%%
# The maximum length sentence we want for a single input in characters
seq_length = 100
examples_per_epoch = len(text)//(seq_length+1)

# Create training examples / targets
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

#%%
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)
def split_input_target(chunk):
    input_text = chunk[:-1]  #将前n-1个字符作为输入
    target_text = chunk[1:] #将后1-n为目标训练
    return input_text, target_text

dataset = sequences.map(split_input_target)

#%%
for input_example, target_example in  dataset.take(1):
    print ('Input data: ', repr(''.join(idx2char[input_example.numpy()])))
    print ('Target data:', repr(''.join(idx2char[target_example.numpy()])))

#%%
BATCH_SIZE = 64
BUFFER_SIZE = 10000
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

#%%
dataset
#%%
# Length of the vocabulary in chars
vocab_size = len(vocab)
# The embedding dimension
embedding_dim = 256
# Number of RNN units
rnn_units = 1024
#%%

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
    tf.keras.layers.GRU(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)
    ])
    return model
#%%
model = build_model(
  vocab_size = len(vocab),
  embedding_dim=embedding_dim,
  rnn_units=rnn_units,
  batch_size=BATCH_SIZE) 

#%%
model.summary()
#%%
for input_example_batch, target_example_batch in dataset.take(1):
    example_batch_predictions = model(input_example_batch)
    print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")
sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()
#%%
def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

example_batch_loss  = loss(target_example_batch, example_batch_predictions)
print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
print("scalar_loss:      ", example_batch_loss.numpy().mean())

#%%
model.compile(optimizer='adam', loss=loss)

#%%
# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)
#%%
EPOCHS=10
history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])
#%%

tf.train.latest_checkpoint(checkpoint_dir)
#%%

model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1, None]))
#%%
def generate_text(model, start_string):
    # Evaluation step (generating text using the learned model)

    # Number of characters to generate
    num_generate = 1000

    # Converting our start string to numbers (vectorizing)
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    # Empty string to store our results
    text_generated = []

    # Low temperatures results in more predictable text.
    # Higher temperatures results in more surprising text.
    # Experiment to find the best setting.
    temperature = 1.0

    # Here batch size == 1
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        # using a categorical distribution to predict the character returned by the model
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

        # We pass the predicted character as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(idx2char[predicted_id])

    return (start_string + ''.join(text_generated))
#%%
print(generate_text(model, start_string=u"ROMEO: "))
#%%


'''
输出样例：

ROMEO: I in health,
To Bastil turn of mine shant that?

Messenger:
Now the battle eat I come intend
The cried a thanking pleasure reck; you have no need giess,
Which, I am known,
Then have you ours to take ake of foot.

HENRY BOLINGBROKE:
I am his brother's, thinking
dysirms that comfort at quick forfett
The travellous angience of consprince,
The elder, and slingerous haste, they come.
We have been both, and we to ope. What carest thou, so:

KATHARINA:
Too great revenge! I mean, your sun can scape it.

MENENIUS:
Kath, I have bestain'd to your store to all the king's wife: my lord.

KING RICHARD II:
What will you stay believe you ought to hear.

QUEEN MARGARET:
Nay, no;
Forbed Henry of Lucentio.

BAPTISTA:
A leck my daughter, have as a suddended
For sweet Corioli, look you this,
You stable day.

DERBY:
Who is a better goods are gone?

CAPULET:
Coos therein my fortune pass adverside,
For thus I sa?

AING proboun: all my pity
I wannound my nature of your hands;
And 'tis right addisation to dysel
'''


