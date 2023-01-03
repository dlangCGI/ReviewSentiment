import pkg_resources
import re
from symspellpy import SymSpell, Verbosity

from nltk.tokenize import sent_tokenize

from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

VOCAB_SIZE = 15000
MAX_LEN = 100

# initialize Spellchecker
dictionary_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_dictionary_en_82_765.txt"
)
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

# delete all duplicates at once -> works most of the time and faster
repeat_pattern = re.compile(re.compile(r'(\w)\1*'))
match_substitution = r'\1'


def expand_contractions(phrase):
    """expand contractions like can't to can not"""
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)

    return phrase


def spell_correction(word_arr):
    for i, word in enumerate(word_arr):
        # lookup suggestions
        # include_unknown=False -> don't include tested word if nothing found
        # just include TOP -> closest with the highest frequency
        suggestions = sym_spell.lookup(word, Verbosity.TOP,
                                       max_edit_distance=2,
                                       include_unknown=False)
        # append correction if found
        if suggestions:
            word_arr[i] = suggestions[0].term
        # if not try to delete duplicate letters first (like 'greeeat')
        else:
            del_duplicate = repeat_pattern.sub(match_substitution, word)
            suggestions = sym_spell.lookup(del_duplicate, Verbosity.TOP,
                                           max_edit_distance=2,
                                           include_unknown=False)
            # if word exists after removal -> append; otherwise skip
            if suggestions:
                word_arr[i] = suggestions[0].term

    return word_arr


def clean_text(raw_text):
    """Do basic cleaning before tokenizing for keras model.
    This also includes sentence tokenization for removing punctuation and
    perform spelling correction.
    Sentences are seperated by a dot after cleaning."""
    # split attached words ->
    # capital + at last one lower case +
    # arbitrary other characters till the first capital again
    temp = " ".join(re.split(r'([A-Z][a-z]+[^A-Z]*)', raw_text))
    # lowercase
    temp = temp.lower()
    # remove links
    # (first www to https, then everything with http until the next whitespace)
    temp = re.sub(r'www', 'https', temp)
    temp = re.sub(r'http\S+', '', temp)
    # combine words like "wi-fi"
    temp = re.sub(r'(\w+)(\s-\s|-)(\w+)', r'\1\3', temp)
    # remove numbers
    temp = re.sub(r'\d+', '', temp)
    # expanding apostrophes
    pre_cleaned = expand_contractions(temp)

    # perform rest of cleaning for each sentence to keep sentences seperated
    sentences = sent_tokenize(pre_cleaned)
    for i, sentence in enumerate(sentences):
        # remove punctuation
        sentence = re.sub(r'\W+', ' ', sentence).strip()
        # correct spellings
        sentence = " ".join(spell_correction(sentence.split(" ")))
        sentences[i] = sentence

    return " . ".join(sentences)


def get_sequences(tokenizer, cleaned_text):
    """convert cleaned text to word index sequences.

    tokenizer       -- fitted keras tokenizer on training data
    cleaned_text    -- array of phrases
    """

    sequences = tokenizer.texts_to_sequences(cleaned_text)
    # pad sequences to get uniform size
    padded = pad_sequences(sequences, maxlen=MAX_LEN,
                           truncating='post', padding='post')
    return padded
