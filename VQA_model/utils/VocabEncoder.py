#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sylvain
"""

# Encodeur du vocabulaireß

import json

MAX_ANSWERS = 100
LEN_QUESTION = 20


class VocabEncoder():
    # Création du dictionnaire en parcourant l'ensemble du JSON (des questions ou des réponses)
    def __init__(self, JSONFile, string=None, questions=True, range_numbers=False):
        self.encoder_type = 'answer'
        if questions:
            self.encoder_type = 'question'
        self.questions = questions
        self.range_numbers = range_numbers
        
        words = {}  
        
        if JSONFile != None:
            with open(JSONFile) as json_data:
                self.data = json.load(json_data)[self.encoder_type + 's']
        else:
            if questions:
                self.data = [{'question':string}]
            else:
                self.data = [{'answer':string}]
            
        
        for i in range(len(self.data)):
            if self.data[i]["active"]:
                sentence = self.data[i][self.encoder_type]
                if sentence[-1] == "?" or sentence[-1] == ".":
                    sentence = sentence[:-1]
                
                tokens = sentence.split()
                for token in tokens:
                    token = token.lower()
                    if range_numbers and token.isdigit() and not questions:
                        num = int(token)
                        if num > 0 and num <= 10:
                            token = "between 0 and 10"
                        if num > 10 and num <= 100:
                            token = "between 10 and 100"
                        if num > 100 and num <= 1000:
                            token = "between 100 and 1000"
                        if num > 1000:
                            token = "more than 1000"

                    if token[-2:] == 'm2' and not questions:
                        num = int(token[:-2])
                        if num > 0 and num <= 10:
                            token = "between 0m2 and 10m2"
                        if num > 10 and num <= 100:
                            token = "between 10m2 and 100m2"
                        if num > 100 and num <= 1000:
                            token = "between 100m2 and 1000m2"
                        if num > 1000:
                            token = "more than 1000m2"
                    if token not in words:
                        words[token] = 1
                    else:
                        words[token] += 1
                
        sorted_words = sorted(words.items(), key=lambda kv: kv[1], reverse=True)
        self.words = {
            "no": 0,
            "yes": 1,
            "0m2": 2,
            "between 0m2 and 10m2": 3,
            "between 10m2 and 100m2": 4,
            "between 100m2 and 1000m2": 5,
            "more than 1000m2": 6,
            "0": 7,
            "1": 8,
            "2": 9,
            "3": 10,
            "4": 11,
            "5": 12,
            "6": 13,
            "7": 14,
            "8": 15,
            "9": 16,
            "10": 17,
            "11": 18,
            "12": 19,
            "13": 20,
            "14": 21,
            "15": 22,
            "16": 23,
            "17": 24,
            "18": 25,
            "19": 26,
            "20": 27,
            "21": 28,
            "22": 29,
            "23": 30,
            "24": 31,
            "25": 32,
            "26": 33,
            "27": 34,
            "28": 35,
            "29": 36,
            "30": 37,
            "31": 38,
            "32": 39,
            "33": 40,
            "34": 41,
            "35": 42,
            "36": 43,
            "37": 44,
            "38": 45,
            "39": 46,
            "40": 47,
            "41": 48,
            "42": 49,
            "43": 50,
            "44": 51,
            "45": 52,
            "46": 53,
            "47": 54,
            "48": 55,
            "49": 56,
            "50": 57,
            "51": 58,
            "52": 59,
            "53": 60,
            "54": 61,
            "55": 62,
            "56": 63,
            "57": 64,
            "58": 65,
            "59": 66,
            "60": 67,
            "61": 68,
            "62": 69,
            "63": 70,
            "64": 71,
            "65": 72,
            "66": 73,
            "67": 74,
            "68": 75,
            "69": 76,
            "70": 77,
            "71": 78,
            "72": 79,
            "73": 80,
            "74": 81,
            "75": 82,
            "76": 83,
            "77": 84,
            "78": 85,
            "79": 86,
            "80": 87,
            "81": 88,
            "82": 89,
            "83": 90,
            "84": 91,
            "85": 92,
            "86": 93,
            "89": 94,
        }
        # self.list_words = ['<EOS>']
        # for i, word in enumerate(sorted_words):
        #     if self.encoder_type == 'answer':
        #         if i >= MAX_ANSWERS:
        #             break
        #     self.words[word[0]] = i + 1
        #     self.list_words.append(word[0])
    
    #Encodage d'une phrase (question ou réponse) à partir du dictionnaire crée plus tôt.        
    def encode(self, sentence):
        res = []
        if sentence[-1] == "?" or sentence[-1] == ".":
            sentence = sentence[:-1]
            
        tokens = sentence.split()
        for token in tokens:
            token = token.lower()
            if self.range_numbers and token.isdigit() and not self.questions:
                num = int(token)
                if num > 0 and num <= 10:
                    token = "between 0 and 10"
                if num > 10 and num <= 100:
                    token = "between 10 and 100"
                if num > 100 and num <= 1000:
                    token = "between 100 and 1000"
                if num > 1000:
                    token = "more than 1000"
                    
            if token[-2:] == 'm2' and not self.questions:
                num = int(token[:-2])
                if num > 0 and num <= 10:
                    token = "between 0m2 and 10m2"
                if num > 10 and num <= 100:
                    token = "between 10m2 and 100m2"
                if num > 100 and num <= 1000:
                    token = "between 100m2 and 1000m2"
                if num > 1000:
                    token = "more than 1000m2"
            res.append(self.words[token])
        
        if self.questions:
            res.append(self.words['<EOS>'])
        
        if self.questions:
            while len(res) < LEN_QUESTION:
                res.append(self.words['<EOS>'])
            res = res[:LEN_QUESTION]
        return res
    
    
    def getVocab(self):
        self.list_words = [
                "no",
                "yes",
                "0m2",
                "between 0m2 and 10m2",
                "between 10m2 and 100m2",
                "between 100m2 and 1000m2",
                "more than 1000m2",
                "0",
                "1",
                "2",
                "3",
                "4",
                "5",
                "6",
                "7",
                "8",
                "9",
                "10",
                "11",
                "12",
                "13",
                "14",
                "15",
                "16",
                "17",
                "18",
                "19",
                "20",
                "21",
                "22",
                "23",
                "24",
                "25",
                "26",
                "27",
                "28",
                "29",
                "30",
                "31",
                "32",
                "33",
                "34",
                "35",
                "36",
                "37",
                "38",
                "39",
                "41",
                "42",
                "43",
                "44",
                "45",
                "46",
                "47",
                "48",
                "49",
                "50",
                "51",
                "52",
                "53",
                "54",
                "55",
                "56",
                "57",
                "58",
                "59",
                "60",
                "61",
                "62",
                "63",
                "64",
                "65",
                "66",
                "67",
                "68",
                "69",
                "70",
                "71",
                "72",
                "73",
                "74",
                "75",
                "76",
                "77",
                "78",
                "79",
                "80",
                "81",
                "82",
                "83",
                "84",
                "85",
                "86",
                "89",
        ]
        return self.list_words
    
    #Décodage d'une phrase (seulement utilisé pour l'affichage des résultats)
    def decode(self, sentence):
        res = ""
        for i in sentence:
            if i == 0:
                break
            res += self.list_words[i]
            res += " "
        res = res[:-1]
        if self.questions:
            res += "?"
        return res
        
            
            
            
        