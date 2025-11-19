import re

import nltk
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)


def preprocess_text(text, remove_stopwords_flag=True, use_stemming=False):
    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Lowercase
    text = text.lower()

    # Remove special characters
    text = re.sub(r"[^a-z0-9\s]", "", text)

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords (optional)
    if remove_stopwords_flag:
        stop_words = set(stopwords.words("english"))
        tokens = [word for word in tokens if word not in stop_words]

    # Apply stemming or lemmatization
    if use_stemming:
        stemmer = PorterStemmer()
        processed_tokens = [stemmer.stem(word) for word in tokens]
    else:
        lemmatizer = WordNetLemmatizer()
        pos_tags = pos_tag(tokens)
        processed_tokens = [
            lemmatizer.lemmatize(word, get_wordnet_pos(pos)) for word, pos in pos_tags
        ]

    return " ".join(processed_tokens)


def get_wordnet_pos(nltk_pos_tag):
    """Convert NLTK POS tags to WordNet POS tags for lemmatization."""
    if nltk_pos_tag.startswith("J"):
        return wordnet.ADJ
    elif nltk_pos_tag.startswith("V"):
        return wordnet.VERB
    elif nltk_pos_tag.startswith("N"):
        return wordnet.NOUN
    elif nltk_pos_tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN
