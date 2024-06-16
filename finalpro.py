import os
from nltk.tokenize import word_tokenize
from natsort import natsorted
import pandas as pd
import numpy as np
import math
import warnings

warnings.filterwarnings("ignore")


def read_files(file):
    if 'txt' in file:
        with open(f'files/{file}', 'r') as f:
            return f.read()

print("\n")
documents = []
terms_order = []  # New list to store the order of terms
for file in os.listdir('files'):
    content = read_files(file)
    documents.append(content)
    tokens = word_tokenize(content)
    terms_order.extend(tokens)  # Add terms to the order list
print("Read 10 txt files:")
print(documents)
print("\n")

token_docs = []
for document in documents:
    token_docs.append(word_tokenize(document))

print("Tokenization: ")
print(token_docs)
print("\n")

from nltk.stem import PorterStemmer

# Apply stemming to the tokenized words
stemmer = PorterStemmer()
stemmed_words = [[stemmer.stem(word) for word in document] for document in token_docs]
print("Stemming: ")
print(stemmed_words)
print("\n")

print("\n")
print("Build positional index & display each term: ")

def preprocessing(doc):
    token_docs = word_tokenize(doc)
    prepared_doc = []
    for term in token_docs:
        prepared_doc.append(term)
    return prepared_doc

# Initialize the file no.
fileno = 1

# Initialize the dictionary.
pos_index = {}

# Open files.
try:
    file_names = natsorted(os.listdir("files"))
    print(file_names)

    # For every file.
    for file_name in file_names:
        # Read file contents.
        with open(f'files/{file_name}', 'r') as f:
            doc = f.read()
        # preprocess doc
        final_token_list = preprocessing(doc)
        # For position and term in the tokens.
        for pos, term in enumerate(final_token_list):
            # If term already exists in the positional index dictionary.
            if term in pos_index:
                # Increment total freq by 1.
                pos_index[term][0] = pos_index[term][0] + 1
                # Check if the term has existed in that DocID before.
                if fileno in pos_index[term][1]:
                    pos_index[term][1][fileno].append(pos)
                else:
                    pos_index[term][1][fileno] = [pos]
            # If term does not exist in the positional index dictionary
            else:
                # Initialize the list.
                pos_index[term] = []
                # The total frequency is 1.
                pos_index[term].append(1)
                # The postings list is initially empty.
                pos_index[term].append({})
                # Add doc ID to postings list.
                pos_index[term][1][fileno] = [pos]

        # Increment the file no. counter for document ID mapping
        fileno += 1
except Exception as e:
    print(f"Error building positional index: {e}")

print(pos_index)
print("\n")

def put_query(q, display=1):
    lis = [[] for i in range(10)]
    q = preprocessing(q)
    for term in q:
        if term in pos_index.keys():
            for key in pos_index[term][1].keys():
                if lis[key - 1] != []:
                    if lis[key - 1][-1] == pos_index[term][1][key][0] - 1:
                        lis[key - 1].append(pos_index[term][1][key][0])
                else:
                    lis[key - 1].append(pos_index[term][1][key][0])
    positions = []
    if display == 1:
        for pos, list in enumerate(lis, start=1):
            if len(list) == len(q):
                positions.append('document ' + str(pos))
        return positions
    else:
        for pos, list in enumerate(lis, start=1):
            if len(list) == len(q):
                positions.append('doc' + str(pos))
        return positions

q = 'fools the fear'
put_query(q)

documents = []
file = natsorted(os.listdir('files'))
for file in range(1, 11):
    documents.append(" ".join(preprocessing(read_files(str(file)+'.txt'))))

all_terms = list(set(terms_order))  # Use the unique terms_order list to define terms
terms_order.sort(key=lambda x: all_terms.index(x))  # Sort terms_order based on their order of appearance
for doc in documents:
    for term in doc.split():
        all_terms.append(term)
all_terms = set(all_terms)

def get_tf(document):
    wordDict = dict.fromkeys(all_terms, 0)
    for word in document.split():
        wordDict[word] += 1
    return wordDict







tf_values = get_tf(documents[0])
tf = pd.DataFrame(tf_values.values(), index=tf_values.keys(), columns=['doc1'])

for i in range(1, len(documents)):
    tf_values = get_tf(documents[i])
    tf['doc'+str(i+1)] = tf_values.values()

print("-------------------------------Term Frequency----------------------------")
print(tf)

def weighted_tf(x):
    if x > 0:
        return math.log10(x) + 1
    return 0

w_tf = tf.copy()
for i in range(0, len(documents)):
    w_tf['doc'+str(i+1)] = tf['doc'+str(i+1)].apply(weighted_tf)

print("\n")
print("-----------------------------w tf(1+ log tf)----------------------------")
print(w_tf)
print("\n")

try:
    tdf = pd.DataFrame(columns=['df', 'idf'])
    for i in range(len(tf)):
        in_term = w_tf.iloc[i].values.sum()
        tdf.loc[i, 'df'] = in_term
        tdf.loc[i, 'idf'] = math.log10(10 / (float(in_term)))

    tdf.index = w_tf.index

    print("------------df & idf----------")
    print(tdf)
    print("TF*IDF")
    print("\n")

    tf_idf = w_tf.multiply(tdf['idf'], axis=0)
    print("---------------------------------tf * idf---------------------------------")
    print(tf_idf)
    print("\n")

    def get_doc_len(col):
        return np.sqrt(tf_idf[col].apply(lambda x: x**2).sum())

    doc_len = pd.DataFrame()
    for col in tf_idf.columns:
        doc_len.loc[0, col+'_length'] = get_doc_len(col)

    doc_len_transposed = doc_len.transpose()
    doc_len_swapped = pd.DataFrame(data=doc_len.values, columns=doc_len.columns, index=['length'])
    print("-----------doc length-----------")
    print(doc_len_transposed)
    print("--------------------------------")
    print(doc_len['doc1_length'].values[0])
    print("\n")

    def get_norm_tf_idf(col, x):
        try:
            return x / doc_len[col+'_length'].values[0]
        except:
            return 0

    norm_tf_idf = pd.DataFrame()
    for col in tf_idf.columns:
        norm_tf_idf[col] = tf_idf[col].apply(lambda x : get_norm_tf_idf(col, x))

    print("---------------------------------Normalized tf.idf---------------------------------")
    print(norm_tf_idf)

    print("\n")
    print("---------------------------------After Inserting Queries-----------------------------")

    def get_w_tf(x):
        try:
            return math.log10(x)+1
        except:
            return 0

    def insert_query(q):
        try:
            docs_found = put_query(q, 2)
            if docs_found == []:
                print("Not Found")
                return

            new_q = preprocessing(q)
            query = pd.DataFrame(index=norm_tf_idf.index)
            query['tf'] = [1 if x in new_q else 0 for x in list(norm_tf_idf.index)]
            query['w_tf'] = query['tf'].apply(lambda x: get_w_tf(x))
            product = norm_tf_idf.multiply(query['w_tf'], axis=0)
            query['idf'] = tdf['idf'] * query['w_tf']
            query['tf_idf'] = query['w_tf'] * query['idf']
            query['normalized'] = 0
            for i in range(len(query)):
                query['normalized'].iloc[i] = float(query['idf'].iloc[i]) / math.sqrt(sum(query['idf'].values**2))

            print('Query Details')
            print(query.loc[new_q])
            product2 = product.multiply(query['normalized'], axis=0)
            scores = {}
            for col in put_query(q, 2):
                scores[col] = product2[col].sum()
            product_result = product2[list(scores.keys())].loc[new_q]
            print()
            print('Product (query*matched doc)')
            print(product_result)
            print()
            print('product sum')
            print(product_result.sum())
            print()
            print('Query Length')
            q_len = math.sqrt(sum([x**2 for x in query['idf'].loc[new_q]]))
            print(q_len)
            print()
            print('Cosine Similarity')
            print(product_result.sum())
            print()
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            print('Returned docs')
            for typle in sorted_scores:
                print(typle[0], end=" ")
        except Exception as e:
            print(f"Error processing query: {e}")

            print("\n")

    while True:
        query = input("\nEnter your Query (or type 'exit' to end): ")

        if query.lower() == 'exit':
            break

        insert_query(query)

except Exception as e:
    print(f"An error occurred: {e}")

