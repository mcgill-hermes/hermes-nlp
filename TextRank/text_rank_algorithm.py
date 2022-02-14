import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx


def combine_articles():
    # the csv file includes 8 short articles
    df = pd.read_csv("tennis_articles.csv", engine='python')
    # add content from 5 articles to the list sentences
    sentences = []
    for s in df['article_text']:
        sentences.append(sent_tokenize(s))
    # flatten list
    sentences = [y for x in sentences for y in x]
    print("length: ", len(sentences))
    return sentences


def preprecoss_sentences(sentences):
    # remove punctuations, numbers and special characters
    clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")
    # make alphabets lowercase
    clean_sentences = [s.lower() for s in clean_sentences]
    # remove stopwords from the sentences
    clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]
    return clean_sentences


# function to remove stopwords
def remove_stopwords(sen):
    stop_words = stopwords.words('english')
    sen_new = " ".join([i for i in sen if i not in stop_words])
    return sen_new


def extract_word_embedding(input_sentences):
    print("It takes a few seconds to extract word embedding~")
    word_embeddings = {}
    f = open('glove.6B.100d.txt', encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word_embeddings[word] = coefs
    f.close()
    print("Number of extracted word_embeddings: ", len(word_embeddings))
    sentence_vectors = []
    for i in input_sentences:
        if len(i) != 0:
            v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()]) / (len(i.split()) + 0.001)
        else:
            v = np.zeros((100,))
        sentence_vectors.append(v)
    return sentence_vectors


def similarity_matrix(sentences, sentence_vectors):
    sim_mat = np.zeros([len(sentences), len(sentences)])
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1, 100),
                                                  sentence_vectors[j].reshape(1, 100))[0, 0]
    return sim_mat


def apply_page_rank_algorithm(sim_mat):
    nx_graph = nx.from_numpy_array(sim_mat)
    scores = nx.pagerank(nx_graph)
    return scores


def rank_sentences(input_sentences, scores):
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(input_sentences)), reverse=True)
    for i in range(10):
        print(ranked_sentences[i][1])


if __name__ == '__main__':
    sentences = combine_articles()
    clean_sentences = preprecoss_sentences(sentences)
    sentence_vectors = extract_word_embedding(clean_sentences)
    scores = apply_page_rank_algorithm(similarity_matrix(sentences, sentence_vectors))
    rank_sentences(sentences, scores)




