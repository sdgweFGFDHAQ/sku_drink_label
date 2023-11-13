#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import random
from random import shuffle
import jieba
import argparse
# import logging
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger = logging.getLogger("eda")


class EDA(object):
    __doc__ = """eda class"""

    _stop_words_path = '../resources/stopwords.txt'
    random.seed(2023)

    def __init__(self, num_aug=9, synonyms_model=None):
        """
        EDA初始化
        :param num_aug: 增强的数据量
        :param synonyms_model: 训练的相似词模型
        """
        self.num_aug = num_aug
        self.synonyms_model = synonyms_model

    def _load_stop_words(self):
        """
        load stop words
        :return:
        """
        file_path = self._stop_words_path
        # 当前位置的绝对路径
        file_path = os.path.join(os.path.dirname(__file__), file_path)
        # logger.debug(f"loading stop words with file:{file_path}")
        stop_words = list()
        with open(file_path, 'r', encoding='utf8') as reader:
            for stop_word in reader:
                stop_words.append(stop_word[:-1])
        return stop_words

    def get_synonyms(self, word):
        """
        获取word最相近的词语
        :param word: 待获取的词语
        :return:
        """
        vec = self.synonyms_model
        try:
            return [str(i[0]) for i in vec.similar_by_word(word=word, topn=3)]
        except KeyError:
            return [word]

    def synonym_replacement(self, words, n):
        """同义词替换
        替换一个语句中的n个单词为其同义词
        :param words: 待处理的一个文本
        :param n: 随机替换的词语数
        :return:
        """
        new_words = words.copy()
        unique_words = []
        for word in new_words:
            if word not in unique_words:
                unique_words.append(word)
        weights = [i + 1 for i in range(len(unique_words))]
        num_replaced = 0
        for _ in range(len(unique_words)):
            random_word = random.choices(unique_words, weights=weights)[0]
            synonyms_words = self.get_synonyms(random_word)
            weights.reverse()
            if len(synonyms_words) >= 1:
                synonym = random.choices(synonyms_words, weights=weights)[0]
                new_words = [synonym if word == random_word else word for word in new_words]
                num_replaced += 1
            if num_replaced >= n:
                break
        return new_words

    def add_word(self, new_words):
        """
        找一个词语的相似词语，随机在语句中插入
        :param new_words:
        :return:
        """
        synonyms_word = []
        counter = 0
        while len(synonyms_word) < 1:
            random_word = new_words[random.randint(0, len(new_words) - 1)]
            synonyms_word = self.get_synonyms(random_word)
            counter += 1
            if counter >= 10:
                return
        random_synonym = random.choice(synonyms_word)
        random_idx = random.randint(0, len(new_words) - 1)
        new_words.insert(random_idx, random_synonym)

    def random_swap(self, words, n):
        """
        Randomly swap two words in the sentence n times
        :param words: 语句
        :param n: 交换n个词语
        :return:
        """
        new_words = words.copy()
        for _ in range(n):
            new_words = self.swap_word(new_words)
        return new_words

    @staticmethod
    def swap_word(new_words):
        """
        词语交换
        :param new_words:
        :return:
        """
        random_idx_1 = random.randint(0, len(new_words) - 1)
        random_idx_2 = random_idx_1
        counter = 0
        while random_idx_2 == random_idx_1:
            random_idx_2 = random.randint(0, len(new_words) - 1)
            counter += 1
            if counter > 3:
                return new_words
        new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1]
        return new_words


    def eda(self, sentence, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1):
        """
        eda main function
        :param sentence: 待处理的语句
        :param alpha_sr: 同义词替换词语比例
        :param alpha_rs: 随机交换比例
        :return:
        """

        def enhance(method_inner, param_inner):
            # logger.debug(f"use method:{method_inner.__name__}")
            tmp_result = []
            for _ in range(num_new_per_technique):
                a_words = method_inner(*param_inner)
                tmp_result.append(' '.join(a_words))
            return tmp_result

        seg_list = jieba.cut(sentence)
        seg_list = " ".join(seg_list)
        words = list(seg_list.split())
        num_words = len(words)
        augmented_sentences = []
        num_new_per_technique = int(self.num_aug / 4) + 1  # every method generate sentence number
        n_sr = max(1, int(alpha_sr * num_words))  # cal synonym replace number
        n_ri = max(1, int(alpha_ri * num_words))
        n_rs = max(1, int(alpha_rs * num_words))

        # 同义词替换sr，随机插入ri，随机交换rs，随机删除rd
        methods = [self.synonym_replacement, self.random_swap]
        params = [(words, n_sr), (words, n_ri), (words, n_rs), (words, p_rd)]
        for method, param in zip(methods, params):
            augmented_sentences.extend(enhance(method, param))

        shuffle(augmented_sentences)

        if self.num_aug >= 1:
            augmented_sentences = augmented_sentences[:self.num_aug]
        else:
            keep_prob = self.num_aug / len(augmented_sentences)
            augmented_sentences = [s for s in augmented_sentences if random.uniform(0, 1) < keep_prob]
        augmented_sentences.append(seg_list)
        return augmented_sentences

