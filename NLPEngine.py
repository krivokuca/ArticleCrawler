"""
 The NLP Engine contains methods that can summarize blocks of text, calculate a sentiment score, perform name/entity recognition and keyword extraction.
 Copyright 2023 Daniel Krivokuca <dan@voku.xyz>
"""
import math
import re
from collections import Counter
import spacy

# This directory is used for storing the custom NLP files (if enabled)
CHECKPOINT_STORAGE_DIRECTORY = "./storage/models/"
class NLPEngine():
    def __init__(self):

        self.CHECKPOINT_STORAGE = CHECKPOINT_STORAGE_DIRECTORY
        self.CHECKPOINTS = {
            'sentiment': 'vestra_sentiment_model.pt',
            'sentiment_vocab': 'vestra_sentiment_model',
            'extractor_stopwords': 'extractor_stopwords_en'
        }
        # load our sentiment model
        self.nlp_tokenizer = self._load_full_nlp()
        self.nlp_is_lg = True
        self.stopwords = self._load_stopwords()
        self.sentiment_model = False

    def get_sentiment(self, text):
        """
        TODO: replace with better solution!
        """
        return -1.0
        sentiment = 0.0
        if len(text) == 0:
            return False

        tokens = self.nlp_tokenizer(text)
        rev_text = " ".join([x.lemma_.lower() for x in tokens])
        sentiment = self.sentiment_model.predict(rev_text)
        sentiment = int(sentiment[0][0].replace("__label__", ""))
        sentiment = self._normalize_sentiment(sentiment)

        return sentiment

    def stem_sentences(self, body):
        """
        Stems a body into sentences
        """
        return self._split_sentences(body)

    def get_keywords(self, text, num_words=10):
        '''
        Extracts keywords from a block of text
        returns a dict where the keys are the keywords
        and the values are the relative score of each keyword
        '''
        text = self._split_words(text)
        if text:
            num_words = len(text)
            text = [x for x in text if x not in self.stopwords]
            frequent = {}

            # count the number of occurences in the text for each word
            for word in text:
                if word in frequent:
                    frequent[word] += 1
                else:
                    frequent[word] = 1

            # the minimum size is either going to be our keyword limit
            # or the length of our keywords if its under that limit
            min_size = min(num_words, len(frequent))

            # sort the keywords by how often they appear
            keywords = sorted(frequent.items(), key=lambda x: (
                x[1], x[0]), reverse=True)

            keywords = keywords[:min_size]
            keywords = dict((x, y) for x, y in keywords)
            for k in keywords:
                # rank each keyword
                score = keywords[k] * 1.0 / max(num_words, 1)
                keywords[k] = score * 1.5 + 1
            return dict(keywords)
        else:
            return {}

    def get_summary(self, headline, text, num_sentences=5):
        '''
        Returns a summarized version of an article
        given the article headline and the text of
        the article
        '''
        summaries = []
        sentences = self._split_sentences(text)
        keywords = self.get_keywords(text)
        title_words = self._split_words(headline)

        ranked_sentences = self._score_calc(
            sentences, title_words, keywords).most_common(num_sentences)
        for rank in ranked_sentences:
            summaries.append(rank[0])
        summaries.sort(key=lambda s: s[0])
        return [s[1] for s in summaries]

    def get_entities(self, headline, text):
        '''
        Extracts a list of entities from a headline and text
        Returns a dictionary where the key is the extracted
        entity and the value is the entity value. Entity values
        are:
            PERSON - for a person/historical figure
            GPE - Government/Political Entity
            ORG - Organization
            MONEY - Monetary value
        '''
        text = "{} {}".format(headline, text)
        doc = self.nlp_tokenizer(text)
        ents = {}
        for entity in doc.ents:
            ents[entity.text] = entity.label_
        return ents

    def _split_sentences(self, text):
        '''
        Splits a string into sentences
        '''

        tokens = self.nlp_tokenizer(text)
        sentences = []
        sents = list(tokens.sents)
        for sent in sents:
            sentences.append(str(sent).replace("\n", ""))
        return sentences

    def _split_words(self, text):
        '''
        Splits a sentence into an array of words
        '''
        try:
            text = re.sub(r'[^\w ]', '', text)  # strip special chars
            return [x.strip('.').lower() for x in text.split()]
        except TypeError:
            return None

    def _score_calc(self, sentences, title_words, keywords):
        """
        Calculates the total score given the sentences, title words and
        keywords
        """
        # finnicky, too high and the summary makes no sense, too low and the
        # amount of sentences ranked good enough for the summary is too high
        # and it gets too long
        ideal_score = 20.0
        len_sentences = len(sentences)
        ranks = Counter()
        for i, s in enumerate(sentences):
            sentence = self._split_words(s)
            title_features = self._title_calc(title_words, sentence)
            sen_length = (1 - math.fabs(ideal_score -
                                        len(sentence)) / ideal_score)
            sen_position = self._position_calc(i+1, len_sentences)
            std_feature = self._standard_calc(sentence, keywords)
            int_feature = self._intersect_calc(sentence, keywords)
            frequency = (std_feature + int_feature) / 2.0 * 10.0
            total_score = (title_features*1.5 + frequency*2.0 +
                           sen_length*1.0 + sen_position*1.0)/4.0
            ranks[(i, s)] = total_score
        return ranks

    def _standard_calc(self, words, keywords):
        score = 0.0
        if(len(words) == 0):
            return 0

        for word in words:
            if word in keywords:
                score += keywords[word]
        final_score = (1.0 / math.fabs(len(words)) * score) / 10.0
        return final_score

    def _intersect_calc(self, words, keywords):
        '''
        Calculate how often the keywords intersect the words
        '''
        if(len(words) == 0):
            return 0
        summed = 0
        first = []
        second = []
        for i, word in enumerate(words):
            if word in keywords:
                score = keywords[word]
                if first == []:
                    first = [i, score]
                else:
                    second = first
                    first = [i, score]
                    difference = first[0] - second[0]
                    summed += (first[1] * second[1]) / (difference ** 2)

        # calculate the number of intersections
        ints = len(set(keywords.keys()).intersection(set(words))) + 1
        return (1 / (ints * (ints + 1.0)) * summed)

    def _title_calc(self, title, sentence):
        '''
        Calculates a score given how often the words in the title
        appear in the words in the sentence
        '''
        if title:
            title = [x for x in title if x not in self.stopwords]
            counter = 0.0
            for word in sentence:
                if(word not in self.stopwords and word in title):
                    counter += 1.0
            return counter / max(len(title), 1)
        else:
            return 0

    def _position_calc(self, i, size):
        '''
        Weighs the word depending on where it appears
        in the sentence and how long the word is. The
        returned value is a
        '''
        normalized = i * 1.0 / size
        if (normalized > 1.0):
            return 0
        elif (normalized > 0.9):
            return 0.15
        elif (normalized > 0.8):
            return 0.04
        elif (normalized > 0.7):
            return 0.04
        elif (normalized > 0.6):
            return 0.06
        elif (normalized > 0.5):
            return 0.04
        elif (normalized > 0.4):
            return 0.05
        elif (normalized > 0.3):
            return 0.08
        elif (normalized > 0.2):
            return 0.14
        elif (normalized > 0.1):
            return 0.23
        elif (normalized > 0):
            return 0.17
        else:
            return 0

    def _normalize_sentiment(self, sent):
        """
        Normalizes the sentiment from 1-5 to to 0.0 - 1.0
        """
        normalized = sent / 5
        return normalized

    def _load_full_nlp(self):
        '''
        Loads the large spacy english model
        '''
        # spacy.prefer_gpu() # bug with spacy, segfaults often if using gpu
        nlp = spacy.load("en_core_web_lg")
        return nlp

    def _load_stopwords(self):
        '''
        Returns an array of stopwords
        '''
        f_stop = self.CHECKPOINT_STORAGE + \
            self.CHECKPOINTS['extractor_stopwords']
        stopwords = Counter()
        with open(f_stop, 'r', encoding='utf-8') as f:
            stopwords.update(set([w.strip() for w in f.readlines()]))
        return stopwords
