''' Author: Ranganath Srinivasan Kalaimani
Submitted for Monash University - FIT5166
Student ID: 29360714'''

# Descrption - This py file will retrieve documents based upon the query given in the terminal
# Two segements are there in this file 1. Index part and 2. Search part
# The index part will read all files and will create an inverted index file which has term frequency idf for all terms in a document
# The Search part will calculate cosine similarity for the given query and will retrieve documents based upon that.

from spelling_correction import Spelling_Correction
from ast import literal_eval #
from nltk.stem import PorterStemmer
import math
import os
import sys
import re

# Tokenization class which has a tokenization function
class Tokenization:
    def __init__(self):
        self.tokens_list = []
    # This function takes a opened file as a input argument, and it tokenizes the file with several regular expressions.
    # Regex patterns are compiled for matching several patterns that occur in the file.
    def tokenization(self, file):

        IP_rx = re.compile(r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})') # Regex for IP Address
        email_rx = re.compile(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+') # Regex for Email
        date_rx = re.compile(r'((?:0[1-9]|[12][0-9]|3[01])[./-](?:(?:0?[1-9]|1[0-2])|(?:\w+))[./-](?:(?:\d{2})?\d{2}))') # Regex for Date
        URL_rx = re.compile(r'(https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+)') # Regex for URL
        single_quotes_rx = re.compile(r'(?:^|\s)\'([^\']*?)\'(?:$|\s)') # Regex for single Quotes
        hyphenated_rx = re.compile(r"([\w]+(?:\n-[\w]+)+)") # Regex for hyphenated words
        name1_rx = re.compile(r"([A-Z][\w]+[.'\s?](?:[A-Z]['.]\s?)[A-Z][\w]+(?:.[A-Z][\w]+)?)") # Regex for finding names
        cont_capital_rx = re.compile(r"([A-Z][a-z]+[ ](?:[A-Z][a-z]+[ ]?)+)") # Regex for matching continoous captial words
        acronyms_rx = re.compile(r"((?:[A-Z]\.)+(?:[A-Z]+))") # Regex for acronyms
        contraction_rx = re.compile(r"([\w]+'[\w]+)") # regex for contraction words

        puncList = [".", ";", ":", "!", "?", "/", "\\", ",", "#", "@", "$", "&", ")", "(", "\"",'\n','-','_']
        # punctuation list is pre defined for removing punctuation present in the list from the tokenized words

        # Empty lists are declared for storing the matched words in their coresponding list
        IP_address_list= []
        email_list = []
        date_list = []
        URL_list = []
        single_quotes = []
        hyphenated_list = []
        name1_list = []
        cont_capital_list = []
        acronyms_list = []
        contraction_list = []

        # Getting IP address from the file
        if re.search(IP_rx,file):
            IP_address_list = re.findall(IP_rx,file) # all matched groups are stored in the list
            # print('IP',IP_address_list)
            for i in range(len(IP_address_list)):
                file = file.replace(IP_address_list[i], '') # after a group is matched, its removed from the file
                IP_address_list[i] = str(IP_address_list[i]).strip('`()*&^%$#@!+_-~{}[]:;?/.,\' ')  # strip function is used for obtained string present in the list

        # Getting Email patterns from the file
        if re.search(email_rx,file):
            email_list = re.findall(email_rx,file) # all matched groups are stored in the list
            # print('Email', email_list)
            for i in range(len(email_list)):
                file = file.replace(email_list[i], '') # after a group is matched, its removed from the file
                email_list[i] = str(email_list[i]).strip('`()*&^%$#@!+_-~{}[]:;?/.,\' ') # strip function is used for obtained string present in the list

        # Getting Date patterns from the file
        if re.search(date_rx,file):
            date_list = re.findall(date_rx,file) # all matched groups are stored in the list
            # print('date', date_list)
            for i in range(len(date_list)):
                file = file.replace(date_list[i], '') # after a group is matched, its removed from the file
                date_list[i] = str(date_list[i]).strip('`()*&^%$#@!+_-~{}[]:;?/.,\' ') # strip function is used for obtained string present in the list

        # Getting URL patterns from the file
        if re.search(URL_rx, file):
            URL_list = re.findall(URL_rx, file) # all matched groups are stored in the list
            for i in range(len(URL_list)):
                file = file.replace(URL_list[i], '') # after a group is matched, its removed from the file
                URL_list[i] = str(URL_list[i]).strip('`()*&^%$#@!+_-~{}[]:;?/.,\' ') # strip function is used for obtained string present in the list

        no_removal = URL_list + date_list + email_list + IP_address_list
        # URL, Date, Email and IP adress list are combined as a single list

        # Getting Single quotes patterns from the file
        if re.search(single_quotes_rx,file):
            single_quotes = re.findall(single_quotes_rx,file) # all matched groups are stored in the list
            # print('Quotes', single_quotes)
            for i in range(len(single_quotes)):
                file = file.replace(single_quotes[i], '') # after a group is matched, its removed from the file
                single_quotes[i] = str(single_quotes[i]).strip('`()*&^%$#@!+_-~{}[]:;?/.,\' ') # strip function is used for obtained string present in the list

        # Getting Hyphenated patterns from the file
        if re.search(hyphenated_rx,file):
            hyphenated_list = re.findall(hyphenated_rx,file) # all matched groups are stored in the list
            # print('hyphenated_rx',hyphenated_list)
            for i in range(len(hyphenated_list)):
                file = file.replace(hyphenated_list[i], '') # after a group is matched, its removed from the file
                hyphenated_list[i] = str(hyphenated_list[i]).strip('`()*&^%$#@!+_-~{}[]:;?/.,\' ') # strip function is used for obtained string present in the list

        # Getting name patterns from the file
        if re.search(name1_rx,file):
            name1_list = re.findall(name1_rx,file) # all matched groups are stored in the list
            # print('name',name1_list)
            for i in range(len(name1_list)):
                file = file.replace(name1_list[i], '') # after a group is matched, its removed from the file
                name1_list[i] = str(name1_list[i]).strip('`()*&^%$#@!+_-~{}[]:;?/.,\' ') # strip function is used for obtained string present in the list

        # Getting contraction word patterns from the file
        if re.search(contraction_rx,file):
            contraction_list = re.findall(contraction_rx,file) # all matched groups are stored in the list
            # print('contraction',contraction_list)
            for i in range(len(contraction_list)):
                file = file.replace(contraction_list[i], '') # after a group is matched, its removed from the file
                if '\'s' in contraction_list[i]:
                    contraction_list[i] = contraction_list[i].replace('\'s','')
                contraction_list[i] = str(contraction_list[i]).strip('`()*&^%$#@!+_-~{}[]:;?/.,\' ') # strip function is used for obtained string present in the list

        # Getting continous capital words patterns from the file
        if re.search(cont_capital_rx,file):
            cont_capital_list = re.findall(cont_capital_rx,file) # all matched groups are stored in the list
            # print('cont_capital',cont_capital_list)
            for i in range(len(cont_capital_list)):
                file = file.replace(cont_capital_list[i], '') # after a group is matched, its removed from the file
                cont_capital_list[i] = str(cont_capital_list[i]).strip('`()*&^%$#@!+_-~{}[]:;?/.,\' ') # strip function is used for obtained string present in the list

        # Getting acronyms patterns from the file
        if re.search(acronyms_rx,file):
            acronyms_list = re.findall(acronyms_rx,file) # all matched groups are stored in the list
            # print('acronyms',acronyms_list)
            for i in range(len(acronyms_list)):
                file = file.replace(acronyms_list[i], '') # after a group is matched, its removed from the file
                acronyms_list[i] = str(acronyms_list[i]).strip('`()*&^%$#@!+_-~{}[]:;?/.,\' ') # strip function is used for obtained string present in the list

        self.tokens_list =  hyphenated_list + name1_list + cont_capital_list +acronyms_list + contraction_list + single_quotes
        # The tokens_list contains all tokens from hyphenated list, name list, cont_capital_list + acronyms List and from single quote list

        # The tokens_list is iterated for removing \n whihc also has '-',
        # this case is occuring for hyphenated words that are matched from the file
        for i in range(len(self.tokens_list)):
            if '\n' in self.tokens_list[i]:
                self.tokens_list[i] = self.tokens_list[i].replace('\n','')
                if '-' in self.tokens_list[i]:
                    self.tokens_list[i] = self.tokens_list[i].replace('-','')

        # All the tokens in tokens_list are iterated for punctuation removal
        for punct in range(len(puncList)):
            for word in range(len(self.tokens_list)):
                if puncList[punct] in self.tokens_list[word]:
                    self.tokens_list[word] = self.tokens_list[word].replace(puncList[punct], '')

        # After all the regex patterns are obtained, rest of the file split stored in words_list
        words_list = file.split()
        # print('words',words_list)

        # Punctuation removal is done for the words_list
        for punct in range(len(puncList)):
            for word in range(len(words_list)):
                if puncList[punct] in words_list[word]:
                    words_list[word] = words_list[word].replace(puncList[punct],'')

        # print('words_no_punc', words_list)
        words_filtered = []
        for each in range(len(words_list)):
            if words_list[each] != "''" and words_list[each] != '':
                words_filtered.append(words_list[each])


        # print('filtered',words_filtered)
        self.tokens_list = self.tokens_list + words_filtered + no_removal # the tokens_list is updated by combining all the list containing tokens
        return self.tokens_list # the final list is returned for a single file

class Stopword_removal:

    # In this class a function is defined for removing the stopwords from the tokens,
    # The stp_process function takes two arguments, tokens_list and stopword file.

    def __init__(self):
        self.final = []

    def stp_process(self, file, list):

        stopwords_file = open(file, 'r') # the stopwords file is opened
        stopwords_list = stopwords_file.readlines()
        for i in range(len(stopwords_list)):
            stopwords_list[i] = stopwords_list[i].replace('\n', '')

        # the token list is iterated and if the token is not present in the stopwords the token is appened to a new list
        for j in range(len(list)):
            list[j] = list[j].lower()
            if list[j] not in stopwords_list:
                if len(list[j]) >= 3:
                    self.final.append(list[j])
        return self.final

class Stemming:

    # In this class is defined for stemming process
    # A function is defined as stemming_process which takes a list as an input argument
    # Porter Stemmer is used from nltk package

    def __init__(self):
        self.stemmed = []

    def stemming_process(self,token_list):
        stemmer = PorterStemmer()
        for i in range(len(token_list)):
            self.stemmed.append(stemmer.stem(token_list[i]))
        return self.stemmed

class Term_Frequency:

    # A function named term_freq is defined for calculating the term frequency in a document
    # a dictonary is returned from this function which has term and its frequnecy for a document

    def __init__(self):
        self.main_dict = {}

    def term_freq(self,tokens):

        for i in range(len(tokens)):
            freq = []
            temp_dict = {}
            compare = tokens[i]
            for each in range(len(compare)):
                freq.append(compare.count(compare[each]))
                if compare[each] not in temp_dict:
                    temp_dict[compare[each]] = freq[each]
            self.main_dict[i] = temp_dict
            print('term_freq_file:', i+1)
        return self.main_dict

class Document_Frequency:

    # This function is defined for calculating document frequncy of a term
    # function takes a list as an input argument and it returns dictionary of term and its document frequency

    def __init__(self):
        self.main_dict = {}

    def doc_freq(self,all_file_tokens):
        compare_list = []
        for i in range(len(all_file_tokens)):
            temp = all_file_tokens[i]
            for each in range(len(temp)):
                if temp[each] not in compare_list:
                    compare_list.append(temp[each])
            print('Document Frequency calculated for: ',i+1)
        # print(compare_list)

        for i in range(len(compare_list)):
            for each in range(len(all_file_tokens)):
                if compare_list[i] in all_file_tokens[each]:
                    if compare_list[i] in self.main_dict:
                        self.main_dict[compare_list[i]] += 1 # if the term is present in the dictionary as key the count is incremented
                    else:
                        self.main_dict[compare_list[i]]  = 1 # if a new term is added the count is initialized

        return self.main_dict


class Calculation:

# This class have two functions

# idf function for calculating idf for terms
# weight is calculated for all the terms which is used for searching process

    def __init__(self):
        self.idf_dict = {}
        self.weight_dict = {}

    def idf_cal(self,freq_dict,doc_list):
        N= len(doc_list) # the total number of documents
        # print('freq_dict',freq_dict)
        for word,freq in freq_dict.items():
            idf = round(math.log(N/(freq+1)),3) # idf formula
            self.idf_dict[word] = idf # adding the calculated idf and its term to the dictionary
        # print(self.idf_dict)
        return self.idf_dict
# A dictionary is returned from this function

    def weight(self,term_freq,idf_dict):

        self.weight_dict = {}
        for term1, idf in idf_dict.items():
            for term2, value in term_freq.items():
                if type(value) == dict:
                    if term2 == term1:
                        for doc_id, freq in value.items():
                            weight = round((freq * idf), 3) # weight of the term is calculated

# The weight for the term is updated for the respective document in weight dictionary
                            if doc_id in self.weight_dict:
                                self.weight_dict[doc_id].update({term1: weight})
                            else:
                                self.weight_dict[doc_id] = {}
                                self.weight_dict[doc_id].update({term1: weight})

        return self.weight_dict
# A nested dictionary is returned from this function

# A function is defined for calculating cosine similarity, it takes a list and a dictionar as an input argument
def cosine_similarity(query_list, weight_dict):
    cosine_dict = {}
    final_dict = {}
    # Two dictionaries are defined for storing the calculated cosine similarity with its term in the dictionary
    #print("weight dict", weight_dict)
    for doc_id, value in weight_dict.items():
        list_weight = []

        total_weight = list(value.values()) # all the weights are stored inca separate list
        # print('Total_weight',total_weight)

        for term2, weight in value.items():
            if term2 in query_list:  # if the terms are same that are present in query list and in the weight dict, the weight is appnded to a saperate list
                list_weight.append(weight)
        # print('list weight',list_weight)


        sum_denom_weight = 0
        for i in range(len(total_weight)):
            z = total_weight[i]
            sum_denom_weight += (z*z)

        sum_num = 0
        if len(list_weight) != 0:
            for j in range(len(list_weight)):
                x = list_weight[j]
                sum_num += x

        sum_query_weight = len(query_list)


        cosine = round((sum_num / math.sqrt(sum_denom_weight * sum_query_weight)),3)
        # print('cosine:', cosine)

        if cosine != 0:
            cosine_dict[doc_id] = cosine
            # added to a dictionary with the doc_id and its obtained cosine value
    cosine_list = sorted(cosine_dict, key=cosine_dict.get, reverse=True) # the dictionary is sorted by values

    for i in range(len(cosine_list)):
        for key,value in cosine_dict.items():
            if cosine_list[i] == key:
                final_dict[key] = value
# a dictionary is returned from this function
    return final_dict

def invert_index_file_1(idf_dict, term_freq_dict): # function for creating inverted index file
    # the function takes two input arguments, idf dictionary and term frequency dictionary
    x = ''
    for term1, idf in idf_dict.items():
        x += str(term1) + ","
        for doc_id, value in term_freq_dict.items():
            if term1 in value.keys():
                x += 'd' + str(doc_id + 1) + "," + str(value[term1]) + ","
        x += str(idf) + "\n"
    # the string is concatenated and the string is returned
    return x

file_name_dict = {}

# In Search Index, a function is defined named searching which takes a opened file as an input argument
class Search_Index:

    def __int__(self):
        self.term_freq_dict = {}
        self.idf_dict = {}

# this function returns two dictionary, which is read from file

    def searching(self,file):

        opened = file.readlines()
        term_freq_dict = {}
        idf_dict = {}

        # each line is read from file
        for line in opened:
            # print(line)
            line = line.split(',') # line are split with ',' which creates a list

            term = line[0] # first element of the list will be term
            idf_dict[term] = float(line[len(line) - 1].replace('\n', '')) # last element in the list will be idf value
            # idf dictionary is created

            del(line[0])
            del(line[len(line) - 1])

            temp_dict = {}

            for i in range(0,len(line),2):
                temp_dict[line[i]] = int(line[i+1])
            term_freq_dict[term] = temp_dict

        return term_freq_dict,idf_dict

def main():

    arguments =  sys.argv # sys.argv is used for using command line arguments

    freq_obj = Term_Frequency() # Object for term frequency is created
    doc_freq_obj = Document_Frequency() # Object for document frequency is created
    cal_obj = Calculation() # Object for calculating inverse document frequency and weight is created

    if str(arguments[1]) == 'Index' or str(arguments[1]) == 'index': # if the second argument is index, this condition is satisfied
        all_file_tokens = []
        Path = str(arguments[2]) #the third argument from the command line which has path for all files
        filelist = os.listdir(Path) # those files are stored in a list
        filelist = sorted(filelist) # sorted list
        index_init = 0
        for i in filelist:
            print('Currently Running :', i)
            if i.endswith(".txt"): # only txt files are taken
                index_init += 1
                file_name_dict["d"+str(index_init)] = i

                with open(Path+'/'+i, 'r',encoding='utf-8-sig') as file_open:

                    file = file_open.read()
                    file =  str(file)

                tokens = Tokenization() # object for tokenization process is created
                stop_obj = Stopword_removal() # object for stopword removal process is created
                stem_obj = Stemming() # object for stemming process is created

                tokens_list = tokens.tokenization(file)
                filtered = stop_obj.stp_process(str(arguments[4]),tokens_list)
                stem_list = stem_obj.stemming_process(filtered)
                all_file_tokens.append(stem_list)

        file_name = open('DOC_NAMES.txt','w')
        file_name.write(str(file_name_dict))
        file_name.close()

        print('Tokenization Over')

        freq_each_term = freq_obj.term_freq(all_file_tokens) # term frequency is calculated
        # print('Freq',freq_each_term)
        doc_frequency = doc_freq_obj.doc_freq(all_file_tokens) # document frequency is calculated
        # print('DOc_freq',doc_frequency)
        idf_frequency = cal_obj.idf_cal(doc_frequency,all_file_tokens) # idf is calculated
        # print('IDF - Calculated',idf_frequency)

        inverted = invert_index_file_1(idf_frequency, freq_each_term)
        inverted_file = open(str(arguments[3])+'/inverted_index.txt','w',encoding='utf-8-sig')
        inverted_file.write(inverted) # inverted index file is created
        inverted_file.close()
        print('Inverted Index File - Created')

    elif str(arguments[1]) == 'search' or str(arguments[1] == 'Search'): # if the 2nd command line argument is search, this condition is satisfied

        search_obj = Search_Index()
        index_file = open(str(arguments[2]) + '/inverted_index.txt', 'r', encoding='utf-8-sig')
        search = search_obj.searching(index_file)
        term_freq, idf = search

        query = " ".join(arguments[4:])

        tokens = Tokenization() # object for tokenization process is created
        stop_obj = Stopword_removal() # object for stopword removal process is created
        stem_obj = Stemming() # object for stemming process is created

        query_tokens = tokens.tokenization(query)

        spell_check_words = []  # a list is created for storing the spell checked words
        new_words = set()  # corrected words are stored in a set
        spelling_obj = Spelling_Correction(idf.keys())  # all unique tokens are sent to the spelling correction function

        for each in range(len(query_tokens)):

            new_words = spelling_obj.correction(
                query_tokens[each])  # the tokenized query is iterated and each term is sent for spelling correctiong
            spell_check_words_list = list(new_words)

            if len(spell_check_words_list) > 1 or (
                    len(spell_check_words_list) == 1 and spell_check_words_list[0] != query_tokens[each]):
                spell_check_dict = dict(enumerate(spell_check_words_list, start=1))

                print("Involving spell checker for '", query_tokens[each],
                      "' Please check and press corresponding index value (Default - Run with initial query words)")

                for item, val in spell_check_dict.items():
                    print(item, val)
                chosen_word_index = input(
                    "Enter your opinion (index): ")  # user choice of word is obtained stored in choosen word variable
                chosen_word = ""

                if chosen_word_index != "" and int(chosen_word_index) in spell_check_dict.keys():
                    chosen_word = spell_check_dict[int(chosen_word_index)]
                else:
                    chosen_word = query_tokens[each]
            else:
                chosen_word = query_tokens[each]  # the same word is given to the query list
            spell_check_words.append(chosen_word)

        query_stop_words = stop_obj.stp_process('stopwords.txt', spell_check_words)
        query_stemmed = stem_obj.stemming_process(query_stop_words)

        weight_cal = cal_obj.weight(term_freq, idf) # weight is calculated, which takes idf dictionary and term frequency dictionary

        file_names = open('DOC_NAMES.txt', 'r')
        names = file_names.read()

        names = literal_eval(names)
        cosine = cosine_similarity(query_stemmed, weight_cal) # cosine similarity is calculated for the stemmed query with the term weights

        temp_dict = {}

        if len(cosine) != 0:
            for key, value in names.items():
                for i, j in cosine.items():
                    if key == i:
                        temp_dict[value] = j
            temp_list = sorted(temp_dict, key=temp_dict.get, reverse=True) # the obtained dictionary is sorted by values
            mid_dict = {}
            for i in range(len(temp_list)):
                for key, value in temp_dict.items():
                    if temp_list[i] == key:
                        mid_dict[key] = value

            if len(mid_dict) < int(arguments[3]):
                for i, j in mid_dict.items():
                    print(str(i) + "," + str(j))
            else:

                temp_counter = 0

                for i, j in mid_dict.items():
                    if temp_counter < int(arguments[3]):  # if the length of the dictionari is less than the range argument
                        temp_counter += 1
                        print(str(i) + "," + str(j)) # the dictionary is printed
        else:
            print('There are no relevant documents for this query') # if there are no relevant documents for the query

    else:
        print('Error - Enter your choice correctly Index or Search') # if the choice is incorrect

if __name__ == "__main__":
    main()
