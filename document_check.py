def text_processing(ques1, ques2):
    # from nltk.corpus.reader.wordnet import WordNetError
    from nltk.stem.porter import PorterStemmer
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    from nltk.corpus import wordnet as wn
    from nltk.corpus import stopwords
    import num2words as nw
    import string
    """Function to remove punctions in the strings"""
    r_p1 = list(map(lambda ques: ''.join([word for word in ques1 if word not in string.punctuation]), [ques1]))
    r_p2 = list(map(lambda ques: ''.join([word for word in ques2 if word not in string.punctuation]), [ques2]))

    """Function to create word token from the document"""
    w_t1 = list(map(lambda r_p: ' '.join([nw.num2words(word) if word.isdigit() else word for word in word_tokenize(r_p[0].replace("°", "").replace("²", ""))]), [r_p1]))
    w_t2 = list(map(lambda r_p: ' '.join([nw.num2words(word) if word.isdigit() else word for word in word_tokenize(r_p[0].replace("°", "").replace("²", ""))]), [r_p2]))
    l_w_t1 = len(word_tokenize(r_p1[0]))
    l_w_t2 = len(word_tokenize(r_p1[0]))

    """Function to remove stop words from the document"""
    wn.ensure_loaded()
    words = stopwords.words('english')
    r_s_w1 = list(map(lambda w_t: " ".join([word for word in w_t[0].split() if word not in words]), [w_t1]))
    r_s_w2 = list(map(lambda w_t: " ".join([word for word in w_t[0].split() if word not in words]), [w_t2]))
    l_r_s_w1 = len(word_tokenize(r_s_w1[0]))
    l_r_s_w2 = len(word_tokenize(r_s_w2[0]))

    """Function to stem tokens of string"""
    stemmer = PorterStemmer()
    stems1 = list(map(lambda r_s_w: " ".join([stemmer.stem(word) for word in r_s_w[0].split(" ")]), [r_s_w1]))
    stems2 = list(map(lambda r_s_w: " ".join([stemmer.stem(word) for word in r_s_w[0].split(" ")]), [r_s_w2]))

    """Function to lemmatize tokens of string"""
    lemmatizer = WordNetLemmatizer()
    lamit1 = list(map(lambda stems: " ".join([lemmatizer.lemmatize(word) for word in stems[0].split()]), [stems1]))
    lamit2 = list(map(lambda stems: " ".join([lemmatizer.lemmatize(word) for word in stems[0].split()]), [stems2]))

    # print([lamit1[0], lamit2[0]], [int(l_w_t1 / l_w_t2), int(l_r_s_w1 / l_r_s_w2)])
    return [lamit1[0], lamit2[0]], [int(l_w_t1 / l_w_t2), int(l_r_s_w1 / l_r_s_w2)]


def string_comparater(doc):
    """Checking for Sequence Matcher"""
    import difflib
    s = (difflib.SequenceMatcher(None, doc[0], doc[1])).ratio()
    if int(s * 100) == 0:
        n_s = 0
    elif 0 <= int(s * 100) <= 10:
        n_s = 1
    elif 11 <= int(s * 100) <= 20:
        n_s = 2
    elif 21 <= int(s * 100) <= 30:
        n_s = 3
    elif 31 <= int(s * 100) <= 40:
        n_s = 4
    elif 41 <= int(s * 100) <= 50:
        n_s = 5
    elif 51 <= int(s * 100) <= 60:
        n_s = 6
    elif 61 <= int(s * 100) <= 70:
        n_s = 7
    elif 71 <= int(s * 100) <= 80:
        n_s = 8
    elif 81 <= int(s * 100) <= 90:
        n_s = 9
    else:
        n_s = 10

    """Checking for Fuzzy Wuzzy Matcher Set Ratio"""
    from fuzzywuzzy import fuzz
    f_Set_Ratio = fuzz.token_set_ratio(doc[0], doc[1])
    if int(f_Set_Ratio) == 0:
        f1 = 0
    elif 0 <= int(f_Set_Ratio) <= 10:
        f1 = 1
    elif 11 <= int(f_Set_Ratio) <= 20:
        f1 = 2
    elif 21 <= int(f_Set_Ratio) <= 30:
        f1 = 3
    elif 31 <= int(f_Set_Ratio) <= 40:
        f1 = 4
    elif 41 <= int(f_Set_Ratio) <= 50:
        f1 = 5
    elif 51 <= int(f_Set_Ratio) <= 60:
        f1 = 6
    elif 61 <= int(f_Set_Ratio) <= 70:
        f1 = 7
    elif 71 <= int(f_Set_Ratio) <= 80:
        f1 = 8
    elif 81 <= int(f_Set_Ratio) <= 90:
        f1 = 9
    else:
        f1 = 10
    """Checking for Fuzzy Wuzzy Matcher Sort Ratio"""
    f_Sort_Ratio = fuzz.token_sort_ratio(doc[0], doc[1])
    if int(f_Sort_Ratio) == 0:
        f2 = 0
    elif 0 <= int(f_Sort_Ratio) <= 10:
        f2 = 1
    elif 11 <= int(f_Sort_Ratio) <= 20:
        f2 = 2
    elif 21 <= int(f_Sort_Ratio) <= 30:
        f2 = 3
    elif 31 <= int(f_Sort_Ratio) <= 40:
        f2 = 4
    elif 41 <= int(f_Sort_Ratio) <= 50:
        f2 = 5
    elif 51 <= int(f_Sort_Ratio) <= 60:
        f2 = 6
    elif 61 <= int(f_Sort_Ratio) <= 70:
        f2 = 7
    elif 71 <= int(f_Sort_Ratio) <= 80:
        f2 = 8
    elif 81 <= int(f_Sort_Ratio) <= 90:
        f2 = 9
    else:
        f2 = 10
    # print(n_s, f1, f2)
    return n_s, f1, f2


def check_accuracy(values):
    import numpy as np
    import pickle
    dbfile = open('./model/random_forest_classifier.pkl', 'rb')
    train_data = pickle.load(dbfile)
    p = train_data.predict(np.array(values).reshape(1, -1))[0]
    # print(p)
    return p


def check_similarity(q1, q2):
    # from colorama import Fore, Style
    # s1 = ' '.join([word if word in q2 else Fore.GREEN + word + Style.RESET_ALL for word in q1.split()])
    # s2 = ' '.join([word if word in q1 else Fore.RED + word + Style.RESET_ALL for word in q2.split()])
    # print(s1, "\n" + s2)
    # return s1, s2
    s1 = "|".join([word for word in q1.split() if word not in q2])
    s2 = "|".join([word for word in q2.split() if word not in q1])
    # print(s1, s2)
    return s1, s2


# text_processing('my name is abhishek', 'my name is abinash')
# string_comparater(['name abhishek', 'name abinash'])
# check_accuracy([1, 1, 8, 8, 8])
# check_similarity('name abhishek', 'name abinash')