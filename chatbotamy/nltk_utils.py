import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')  # Gerekli olan NLTK veri setini indiriyoruz
stemmer = PorterStemmer()  # PorterStemmer algoritmasını kullanarak bir nesne oluşturuyoruz


def tokenize(sentence):
    """
    Cümleyi kelime/sembol dizisine böler
    Bir belirteç bir kelime, noktalama işareti veya sayı olabilir
    """
    return nltk.word_tokenize(sentence)  # NLTK kütüphanesini kullanarak cümleyi belirteçlere ayırıyoruz


def stem(word):
    """
    kelimenin kök halini bulma işlemi
    örnekler:
    words = ["organize", "organizes", "organizing"]
    words = [stem(w) for w in words]
    -> ["organ", "organ", "organ"]
    """
    return stemmer.stem(word.lower())  # PorterStemmer algoritması ile kelimenin kökünü buluyoruz


def bag_of_words(tokenized_sentence, words):
    """
    kelime torbası dizisini döndürür:
    Cümlede var olan her bilinen kelime için 1, aksi halde 0
    örnek:
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bog   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """
    # Her kelimeyi kök haline getiriyoruz
    sentence_words = [stem(word) for word in tokenized_sentence]
    # Her kelime için 0 ile başlayan bir torba oluşturuyoruz
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1

    return bag
