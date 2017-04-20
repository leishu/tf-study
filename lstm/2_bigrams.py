# We want to train a LSTM over bigrams, that is pairs of consecutive characters like 'ab' instead of single characters like 'a'.
# Since the number of possible bigrams is large, feeding them directly to the LSTM using 1-hot encodings will lead to a very sparse representation that is very wasteful computationally.
#
# a- Introduce an embedding lookup on the inputs, and feed the embeddings to the LSTM cell instead of the inputs themselves.
#
# b- Write a bigram-based LSTM, modeled on the character LSTM above.
#
# c- Introduce Dropout. For best practices on how to use Dropout in LSTMs, refer to this article.