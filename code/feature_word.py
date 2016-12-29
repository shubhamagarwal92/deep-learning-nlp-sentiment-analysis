

### Type 1 pre-processing
def sklearn_vectorizer(clean_train_reviews):
	"""
	Original taken from https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-1-for-beginners-bag-of-words
	"""
	from sklearn.feature_extraction.text import CountVectorizer

	# Initialize the "CountVectorizer" object, which is scikit-learn's
	# bag of words tool.  
	vectorizer = CountVectorizer(analyzer = "word",   \
	                             tokenizer = None,    \
	                             preprocessor = None, \
	                             stop_words = None,   \
	                             max_features = 5000) 

	# fit_transform() does two functions: First, it fits the model
	# and learns the vocabulary; second, it transforms our training data
	# into feature vectors. The input to fit_transform should be a list of 
	# strings.
	train_data_features = vectorizer.fit_transform(clean_train_reviews)

	# Numpy arrays are easy to work with, so convert the result to an 
	# array
	train_data_features = train_data_features.toarray()


##	See https://github.com/shubhamagarwal92/stockPredictionKaggle/blob/master/code/stockPredictions.py
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
def keras_vectorizer(clean_train_reviews):
	# Define max top words in vocabulary
	top_words = 10000
	tokenizer = Tokenizer(nb_words=top_words)
	tokenizer.fit_on_texts(joinedTrainX)
	# word_index = tokenizer.word_index
	# print('Found %s unique tokens.' % len(word_index))
	## Fit tokenizer
def generate_sequence(text,MAX_SEQUENCE_LENGTH):
	sequence= tokenizer.texts_to_sequences(text)
	sequence = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH)
	return sequence
	## Maximum sequence length

def keras_sequences(joinedTrainX,joinedTestX):
	max_words = 500
	sequencesTrainX = generate_sequence(joinedTrainX,max_words)
	sequencesTestX = generate_sequence(joinedTestX,max_words)

## Taken from 
## https://github.com/alexander-rakhlin/CNN-for-Sentence-Classification-in-Keras/blob/master/data_helpers.py

def pad_sentences(sentences, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]

