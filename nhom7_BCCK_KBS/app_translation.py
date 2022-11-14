import math
from tkinter import messagebox
import tkinter as tk
from turtle import width
from tkinter import Tk
from tkinter import *
from collections import defaultdict
import pickle
from pyvi import ViTokenizer, ViPosTagger
import pandas as pd
import random
import numpy as np
import copy
class Translate(object):
      
# Hàm xử lý khi nút được nhấn
    

    def clicked(self):
        a = main()
        b = ' '.join(a) 
        print('a', b)
        messagebox.showinfo('Result', b)
        
        
        
    def UI(self):
        window = Tk()
        window.title("Translation App")
        window.geometry('600x500')
        lbl = Label(window, text="Vui lòng nhập input",
                    fg="#2b6cb0", font='Helvetica 15 italic')
        lbl.grid(column=0, row=0)
        txt = tk.Entry(window, width=50)
        txt.grid(column=1, row=1)
        txt.focus()
        input = txt.get()
        btn_translate = tk.Button(window, text="Translate", width=10, height=2, command=self.clicked)
        btn_translate.grid(column=1, row=3)
        txt_output = Entry(window, width=10, state='disabled')
        
        window.mainloop()
        return input
    input_text_2 = "tôi là học sinh"

tokenized_stores = {'DataVI': [], 'DataEN': []}
tokenized_tag = defaultdict(list)

# tách câu cho data tiếng việt
file_name = "C:/Users/congv/Desktop/DataVI.txt"
load = open(file_name, encoding='utf-8')
# tach cac cau trong data
sentencesvi = load.read().split("\n")
# for sentence in sentencesvi:
#   token_store = sentence.split(" ")
#   tokenized_stores['DataEN'].append(token_store)
# print(tokenized_stores['DataVI'])
tokenized_stores['DataVI'] = [
    ViTokenizer.tokenize(i).split() for i in sentencesvi]
# gắn tag loại từ cho các từ đầu vào
for i in sentencesvi:
    tag = ViPosTagger.postagging(ViTokenizer.tokenize(i))
    tokenized_tag['DataVI'].append(tag)
file_name = "C:/Users/congv/Desktop/DataEN.txt"
load = open(file_name, encoding='utf-8')
# tach cac cau trong data
sentences = load.read().split("\n")
# tach cac cau thanh cac tu
# hien tai chi tach tu don mot chu cai
for sentence in sentences:
    token_store = sentence.split(" ")
    tokenized_stores['DataEN'].append(token_store)
train_size = len(tokenized_stores['DataEN'])
for index in range(train_size):
    tag = tokenized_tag['DataVI'][index]
    tmp = []
    # tmp = array chứa các từ đã được ghép/tách

    i = 0
    while i in range(len(tag[1])):

        if (i <= len(tag[1])-3):

            L_Nc = tag[1][i] == "L" and (
                tag[1][i+1] == "Nc" or tag[1][i+1] == "N") and tag[1][i+2] == "N"
            E_N = (tag[1][i] == "E" and tag[0][i] == "của") and tag[1][i+1] == "N" and (tag[1]
                                                                                        [i+2] == "N" or tag[1][i+2] == "P" or (tag[1][i+2] == "V" and tag[0][i+2] == "ấy"))

            if (L_Nc or E_N):
                tmp.append(tag[0][i]+" "+tag[0][i+1]+" "+tag[0][i+2])
                i = i+3
                continue

        if (i <= len(tag[1])-2):

            # ghép từ phân loại (determiner) với danh từ
            Nc_N = tag[1][i] == "Nc" and tag[1][i+1] == "N"
            N_N = tag[1][i] == "N" and tag[1][i+1] == "N"  # ghép danh từ ghép
            N_P = tag[1][i] == "N" and tag[1][i+1] == "P"  # ghép danh từ ghép
            N_A = (i == 0 or (i > 0 and (tag[1][i-1] != "Nc" and tag[1][i-1] != "M"))
                   ) and tag[1][i] == "N" and tag[1][i+1] == "A"  # ghép danh tính từ
            # ghép 'của' với một đại từ (của tôi,...)
            E_P = (tag[1][i] == "E" and tag[0][i]
                   == "của") and tag[1][i+1] == "P"
            E_N = (tag[1][i] == "E" and tag[0][i] == "của") and (
                tag[1][i+1] == "N" or tag[1][i+1] == "Np")  # ghép 'của' với một danh từ (của họ, ...)
            # ghép 'những' với một danh từ
            L_N = tag[1][i] == "L" and tag[1][i+1] == "N"
            R_N = tag[1][i] == "R" and tag[1][i +
                                              1] == "N"  # ghép 'hằng', 'ngày'
            N_V = (tag[1][i] == "N" or tag[1][i] == "Nc") and (
                tag[1][i+1] == "V" and tag[0][i+1] == "ấy")  # ghép 'anh'/'cô' 'ấy' thành một từ
            N_E = (tag[1][i] == "N" and tag[0][i] == "mùi") and (
                tag[1][i+1] == "E" and tag[0][i+1] == "của")  # ghép 'mùi', 'của' thành một từ

            if (Nc_N or N_N or N_P or N_A or N_E or E_P or E_N or L_N or R_N or N_V):
                tmp.append(tag[0][i] + " " + tag[0][i+1])
                i = i+2
                continue
        # if (i+1) > len(tag[1])-1:
        tmp.append(tag[0][i])
        i += 1
        # print(str(tmp)+" current index: "+str(i))
    tokenized_stores['DataVI'][index] = tmp
# tạo từ vựng từ các từ đã tách
vn_words = {}
en_words = {}
# neu tu da ton tai trong tu dien thi khong them vao ma chi them so lan xuat hien
# neu tu chua ton tai thi them vao
for key in tokenized_stores:
    if str(key)[4] == "V":
        for sentence in tokenized_stores[key]:
            for word in sentence:
                if word in vn_words:
                    vn_words[word] += 1
                else:
                    vn_words[word] = 1
    else:
        for sentence in tokenized_stores[key]:
            for word in sentence:
                if word in en_words:
                    en_words[word] += 1
                else:
                    en_words[word] = 1
vn_vocab = len(vn_words)
en_vocab = len(en_words)
# tạo P(S|T)
p = {}
uniform = 1/(vn_vocab)
n_iters = 0
max_iters = 20

fine_tune = 0
has_converged = False

while n_iters < max_iters and has_converged == False:
    has_converged = True
    max_change = -1

    n_iters += 1
    count = {}
    total = {}
    for index in range(train_size):
        s_total = {}
        for vn_word in tokenized_stores['DataVI'][index]:
            s_total[vn_word] = 0
            for en_word in tokenized_stores['DataEN'][index]:
                if (vn_word, en_word) not in p:
                    p[(vn_word, en_word)] = uniform
                s_total[vn_word] += p[(vn_word, en_word)]

        for vn_word in tokenized_stores['DataVI'][index]:
            for en_word in tokenized_stores['DataEN'][index]:
                if (vn_word, en_word) not in count:
                    count[(vn_word, en_word)] = 0
                count[(vn_word, en_word)
                      ] += (p[(vn_word, en_word)] / s_total[vn_word])

                if en_word not in total:
                    total[en_word] = 0
                total[en_word] += (p[(vn_word, en_word)] / s_total[vn_word])

    # estimating the probabilities
    for index in range(train_size):
        for en_word in tokenized_stores['DataEN'][index]:
            for vn_word in tokenized_stores['DataVI'][index]:
                if abs(p[(vn_word, en_word)] - count[(vn_word, en_word)] / total[en_word]) > 0.01:
                    has_converged = False
                    max_change = max(max_change, abs(
                        p[(vn_word, en_word)] - count[(vn_word, en_word)] / total[en_word]))
                p[(vn_word, en_word)] = count[(vn_word, en_word)] / total[en_word]
sorted_t = sorted(p.items(), key=lambda k: (k[1], k[0]), reverse=True)
file = open("C:/Users/congv/Desktop/TM-SMTGA.pkl", "wb")
pickle.dump(p, file)
file.close()
model_name = "C:/Users/congv/Desktop/TM-SMTGA.pkl"
pickle_in = open(model_name, "rb")
p = {}
p = pickle.load(pickle_in)
unigrams = {}
bigrams = {}
# training on the train_set


def model(dataset_size, dataset_name):
    global bigrams
    global unigrams
    for index in range(dataset_size):
        token_A = ''
        for en_token in tokenized_stores[dataset_name][index]:
            if en_token not in unigrams:
                unigrams[en_token] = 1
            else:
                unigrams[en_token] += 1

            token_B = en_token
            if (token_A, token_B) not in bigrams:
                bigrams[(token_A, token_B)] = 1
            else:
                bigrams[(token_A, token_B)] += 1
            token_A = token_B


model(train_size, 'DataEN')

bigram_count = len(bigrams)
unigram_count = len(unigrams)


def find_translation(vn_token):
    for element in sorted_t:
        if element[0][0] == vn_token:
            return element[0][1]
    return ""


def get_prob(seq):
    # bigram language model với smoothing
    if len(seq) < 2:
        return 1
    score = 1
    token_A = ''
    for hi_token in seq:
        token_B = hi_token
        if (token_A, token_B) not in bigrams:
            if token_A == token_B:
                score *= 1e-15
            else:
                score *= 1e-10
        else:
            base_token_count = 0
            if token_A in unigrams:
                base_token_count = unigrams[token_A]
            score *= (bigrams[(token_A, token_B)] + 1) / \
                (base_token_count + unigram_count)
        token_A = token_B
    return score


def vietnamese_concatenator(source_sentence):
    tmp = []
    i = 0
    # print(source_sentence[0], source_sentence[1])
    while i in range(len(source_sentence[1])):
        if (i <= len(source_sentence[1])-3):

            L_Nc = source_sentence[1][i] == "L" and (
                source_sentence[1][i+1] == "Nc" or source_sentence[1][i+1] == "N") and source_sentence[1][i+2] == "N"
            E_N = (source_sentence[1][i] == "E" and source_sentence[0][i] == "của") and source_sentence[1][i+1] == "N" and (
                source_sentence[1][i+2] == "N" or source_sentence[1][i+2] == "P" or (source_sentence[1][i+2] == "V" and source_sentence[0][i+2] == "ấy"))

            if (L_Nc or E_N):
                tmp.append(
                    source_sentence[0][i]+" "+source_sentence[0][i+1]+" "+source_sentence[0][i+2])
                i = i+3
                continue

        if (i <= len(source_sentence[1])-2):

            # ghép từ phân loại (determiner) với danh từ
            Nc_N = source_sentence[1][i] == "Nc" and source_sentence[1][i+1] == "N"
            # ghép danh từ ghép
            N_N = source_sentence[1][i] == "N" and source_sentence[1][i+1] == "N"
            # ghép danh từ ghép
            N_P = source_sentence[1][i] == "N" and source_sentence[1][i+1] == "P"
            N_A = (i == 0 or (i > 0 and (source_sentence[1][i-1] != "Nc" and source_sentence[1][i-1] != "M"))
                   ) and source_sentence[1][i] == "N" and source_sentence[1][i+1] == "A"  # ghép danh tính từ
            # ghép 'của' với một đại từ (của tôi,...)
            E_P = (source_sentence[1][i] == "E" and source_sentence[0]
                   [i] == "của") and source_sentence[1][i+1] == "P"
            E_N = (source_sentence[1][i] == "E" and source_sentence[0][i] == "của") and (
                source_sentence[1][i+1] == "N" or source_sentence[1][i+1] == "Np")  # ghép 'của' với một danh từ (của họ, ...)
            # ghép 'những' với một danh từ
            L_N = source_sentence[1][i] == "L" and source_sentence[1][i+1] == "N"
            # ghép 'hằng', 'ngày'
            R_N = source_sentence[1][i] == "R" and source_sentence[1][i+1] == "N"
            N_V = (source_sentence[1][i] == "N" or source_sentence[1][i] == "Nc") and (
                source_sentence[1][i+1] == "V" and source_sentence[0][i+1] == "ấy")  # ghép 'anh'/'cô' 'ấy' thành một từ
            N_E = (source_sentence[1][i] == "N" and source_sentence[0][i] == "mùi") and (
                source_sentence[1][i+1] == "E" and source_sentence[0][i+1] == "của")  # ghép 'mùi', 'của' thành một từ

            if (Nc_N or N_N or N_P or N_A or N_E or E_P or E_N or L_N or R_N or N_V):
                tmp.append(source_sentence[0][i] +
                           " " + source_sentence[0][i+1])
                i = i+2
                continue
        tmp.append(source_sentence[0][i])
        i += 1
    return tmp
# tìm từ thay thế cho các từ chưa có trong cơ sở dữ liệu bằng 2 phương pháp (xâu chung con dài nhất và các từ xuất hiện nhiều nhất)


def LCSubStr(str1, str2, N, M):

    LCSuff = [[0 for k in range(M+1)] for l in range(N+1)]
    mx = 0
    for i in range(N + 1):
        for j in range(M + 1):
            if (i == 0 or j == 0):
                LCSuff[i][j] = 0
            elif (str1[i-1] == str2[j-1]):
                LCSuff[i][j] = LCSuff[i-1][j-1] + 1
                if (mx < LCSuff[i][j]):
                    mx = LCSuff[i][j]
            else:
                LCSuff[i][j] = 0
    return mx


def fix_word(source_text):
    for i in range(len(source_text[0])):
        if source_text[0][i] not in vn_words:
            max_len = 0
            tmp_str = ''
            for word in vn_words:
                tmp = LCSubStr(word, source_text[0][i], len(
                    word), len(source_text[0][i]))
                if (tmp > max_len):
                    max_len = tmp
                    tmp_str = word
            if (max_len > 2):
                source_text[0][i] = tmp_str
    return source_text
# tìm các bản dịch cho các từ kề với từ không có trong cơ sở dữ liệu


def replace_word(source_text, i):
    tmp_near = []
    if i == 0:
        trans_word_2 = find_translation(source_text[0][i+1])
        find_word_2 = find_near(trans_word_2)
        blank = ""
        find_word_1 = find_near(blank)
        tmp_near.append(find_word_1)
        tmp_near.append(find_word_2)
    elif i == len(source_text[0])-1:
        trans_word_2 = find_translation(source_text[0][i-1])
        find_word_2 = find_near(trans_word_2)
        tmp_near.append(find_word_2)
    else:
        trans_word_2 = find_translation(source_text[0][i+1])
        find_word_2 = find_near(trans_word_2)
        trans_word_1 = find_translation(source_text[0][i-1])
        find_word_1 = find_near(trans_word_1)
        tmp_near.append(find_word_1)
        tmp_near.append(find_word_2)
    return tmp_near


sort_bigrams = sorted(
    bigrams.items(), key=lambda k: (k[1], k[0]), reverse=True)


def find_near(word):
    tmp_word = []
    count = 2
    for token in sort_bigrams:
        if (token[0][0] == word):
            tmp_word.append(token[0][1])
            count -= 1
            if count == 0:
                return tmp_word

# tìm 3 bản dịch cho từng từ trong câu
# duyệt từng từ(cụm từ) đã được chuẩn hóa theo tri thức và tìm top 3 bản dịch có xác suất cao nhất
input_text = Translate.input_text_2
input_text = input_text.lower()
input_tag = ViPosTagger.postagging(ViTokenizer.tokenize(input_text))
source_text = [vietnamese_concatenator(input_tag)]


def find_n_translation(source_tok):
    fix_word(source_tok)
    k = 0
    trans_table = defaultdict(list)
    for word in source_tok[0]:
        k += 1
        c = 3
        if word not in vn_words:
            replace = replace_word(source_text, k-1)
            for i in replace[0]:
                trans_table[word].append(i)
                p[(word, i)] = 10e-5
        else:
            for i in sorted_t:
                if (c <= 0):
                    break
                if (i[0][0] == word):
                    trans_table[word].append(i[0][1])
                    c -= 1
    return trans_table


trans_table = find_n_translation(source_text)


def MT_prob(source_text, seq):
    sou_len = len(source_text[0])
    tar_len = len(seq)
    scores = 1
    flag = {}
    for i in range(sou_len):
        for j in range(tar_len):
            if seq[j] in trans_table[source_text[0][i]]:
                if (source_text[0][i], seq[j]) not in flag:
                    scores *= p[(source_text[0][i], seq[j])]
                    flag[(source_text[0][i], seq[j])] = 1
                else:
                    scores *= 1e-15
    return scores


def fitness(member, source_text):
    prob_lm = get_prob(member)
    prob_tm = MT_prob(source_text, member)
    score = math.log(prob_lm, 10)*10 + math.log(prob_tm, 10)
    return score


def Single_crossover(a, b):
    A = list(a)
    B = list(b)

    length = len(a)
    # tìm vị trí để hoán đổi
    k = random.randint(0, length-1)
    for i in range(k, len(a)):
        A[i], B[i] = B[i], A[i]

    return A, B


def mutate(member, probability):
    new_member = copy.deepcopy(member)
    for i in range(1, len(new_member)):
        if random.random() < probability:
            location_1 = random.randint(0, len(new_member)-1)
            location_2 = random.randint(0, len(new_member)-1)
            new_member[location_1], new_member[location_2] = new_member[location_2], new_member[location_1]
    return new_member


def create_new_member(trans_table, source_text):
    # xây dựng một thứ tự sắp xếp cho câu
    source_len = len(source_text[0])
    member = []
    flag = {}
    go = True
    for i in source_text[0]:
        for j in trans_table[i]:
            flag[j] = 0
    k = 0
    while go:
        for i in source_text[0]:
            k += 1
            word = random.sample(trans_table[i], 1)[0]
            if flag[word] == 0:
                rand_word = word
                member.append(rand_word)
                flag[word] = 1
            else:
                while (flag[word] != 0):
                    word = random.sample(trans_table[i], 1)[0]
                rand_word = word
                member.append(rand_word)
                flag[word] = 1
            if k == source_len:
                go = False
    return member


def create_first_population(trans_table, source_text):
    population = []
    for i in range(100):
        member = create_new_member(trans_table, source_text)
        population.append(member)
    return population


def scores_of_population(population, source_text):
    scores = []
    for i in range(len(population)):
        scores.append([fitness(population[i], source_text)])
    return scores


def rankSelect(population, source_text):
    rank = keeper_gen(population, source_text)
    i = random.randint(0, len(rank)/5)
    return population[rank[i][0]]


def keeper_gen(population, source_text):
    list_gen = {}
    for i in range(len(population)):
        score = fitness(population[i], source_text)
        list_gen[i] = score
    list_gen = sorted(list_gen.items(), key=lambda k: (
        k[1], k[0]), reverse=True)
    # list_gen = sorted(list_gen.items(), key=lambda item: item[1])
    return list_gen

# main of Genetic Algorithm


def main():
    trans_table = find_n_translation(source_text)
    # create the first population
    population = create_first_population(trans_table, source_text)
    best = []
    for i in range(100):
        print("Step:", i)
        new_population = []
        # evaluate the fitness of current population
        scores = scores_of_population(population, source_text)
        best = population[np.argmax(scores)]
        probability = fitness(best, source_text)               # chờ LM và TM
        print(best)
        print(probability)
        if probability > -10:
            break
        # crossover
        for j in range(15):
            new_1, new_2 = Single_crossover(rankSelect(
                population, source_text), rankSelect(population, source_text))
            new_population = new_population + [new_1, new_2]
        # mutation
        for i in range(len(new_population)):
            new_population[i] = np.copy(mutate(new_population[i], 0.4))
        new_population += [population[np.argmax(scores)]]
        keepers = keeper_gen(population, source_text)
        len_pop = len(new_population)
        for j in range(100-len_pop):
            new_population += [population[keepers[j][0]]]
        population = copy.deepcopy(new_population)
    # print("Translation for: '", input_text, "'is",best)
    print('best: ', best)
    return best




translate = Translate()
translate.UI()
