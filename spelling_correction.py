""" Author: Ranganath Srinivasan Kalaimani
Submitted for Monash University
Student ID: 29360714"""
# Description - A spel checker program is built for checking and returning correct spelling for incorrect spelling


import re
from collections import Counter

# The  correction is by taking single word as input, which is passed
# for conversion process. There can be possibilities of having a letter missing or an extra letter present in a word 
# A candidates are generated from the correction function whihc is returned as a set. 

class Spelling_Correction:

    def __init__(self, corpose_word_list):
        self.opened = open('big.txt').read()
        self.words_list = re.findall(r'\w+', self.opened.lower())
        self.words_list += corpose_word_list
        self.WORD = Counter(self.words_list)
        # print(self.words_list)

    def known(self,word): # this 
        know_list = []    
        for w in word:
            if w in self.WORD: 
                know_list.append(w)
        return set(know_list)

    def conversion_1(self,word):

        letters = 'abcdefghijklmnopqrstuvwxyz' # alphabets are defined

        split = []
        for i in range(len(word) + 1): # the given word is split which is paired as left, right in a list 
            left = word[:i]
            right = word[i:]
            split.append((left, right)) 

        delete = []
        for i, j in split:
            if j:
                temp = i + j[1:] # for each and every iteration a letter is  removed from the word 
                delete.append(temp)

        transpose = []
        for i, j in split:
            if len(j) > 1:
                temp = i + j[1] + j[0] + j[2:]
                transpose.append(temp)

        replace = [] # replacing the letters in the words with the predefined letters 
        for i, j in split:
            if j:
                for each in range(len(letters)):
                    temp = i + letters[each] + j[1:]
                    replace.append(temp)

        insert = []

        for i, j in split: # inserting the letters with words for the iteration 
        
            for each in range(len(letters)):
                temp = i + letters[each] + j
                insert.append(temp)
        # print("set: ", set(delete + transpose + replace + insert))
        return set(delete + transpose + replace + insert)

    def conversion_2(self,word):
        for e1 in self.conversion_1(word):
            for e2 in self.conversion_1(e1):
                return e2

    def correction(self,word):

        word = word.lower()

        if len(self.known([word])) > 0:
            # print("first")
            canditates = self.known([word])

        elif self.known(self.conversion_1(word)):
            # print("second")
            canditates = self.known(self.conversion_1(word))

        elif self.known(self.conversion_2(word)):
            # print("third")
            canditates = self.known(self.conversion_2(word))

        print("candidates", len(list(canditates)[0]))
        if len(set(map(len,list(canditates)))) not in (0,1) or len(list(canditates)[0]) > 1:
            return canditates
        else:
            return set()
