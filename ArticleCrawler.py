"""
ArticleCrawler.py
The main class responsible for crawling and downloading financial news articles.
Copyright 2023 Daniel Krivokuca <dan@voku.xyz>
"""
import ast
import datetime
import faulthandler
import hashlib
import socket
import string
import time
import uuid

import numpy as np
import requests
from elasticsearch import Elasticsearch

from NewsFilter import NewsFilter

# the main logging directory
LOGGING_DIRECTORY = "./logs/"

# A file of stopwords
STOPWORDS_FILE_LOCATION = "./stopwords_en"

class ArticleCrawler():

    def __init__(self, conf, db=False):
        """
        if the DB is not passed then articles cannot be cached

        Parameters:
            - conf (dict) :: The full configuration dict
            - db (bool|dbconf)
        """
        self.filter = NewsFilter()
        self.conf = conf
        self.sources = self.filter.get_sources()
        self.db = ArticleDatabaseHandler(db, es=False)
        self.push = ArticlePushService(self.conf)
        self.url_mem_list = None

    def crawl_articles(self, ticker=False, timeout=7200, extract_sentiments=False, headless=False):
        """
        Crawls all the article sources given a list of tickers.

        Params:
        ticker - a list of tickers, If false only the top headlines will be used
        timeout - Time, in seconds, after which to stop the execution
        logger - EventLogger class is expected if not False. If false, no logs will be generated
        extract_sentiments - if True, the batch_sentiment_update function will be run for crawled articles
        headless - if True, a POST will be sent to the /secure_ingest endpoint of the main backend API server
        """

        headline_shortcodes = [x['Shortcode']
                               for x in self.sources if "Headlines" in x['Name']]
        named_shortcodes = [x['Shortcode']
                            for x in self.sources if x['Shortcode'] not in headline_shortcodes]

        def _time_check(start_time, end_time):
            # here
            return not((end_time - start_time) > 5400)

        if not ticker:
            results = []
            for shortcode in headline_shortcodes:

                headlines = self.filter.get_headlines(False, shortcode)
                if headlines:
                    results = results + headlines

            stats_obj = self._parse_articles(results, 0)

            if not extract_sentiments:
                stats_obj['message'] = "Finished crawling all headlines"
                return self._response_payload(ticker, True, stats_obj)

            """ TODO: un deprecate
            response = self.batch_update_sentiment(
                aids=stats_obj['writtenlist'])
            """
            response = {}
            stats_obj['message'] = "Finished crawling and extracting all articles"
            return self._response_payload(ticker, True, {**stats_obj, **response})

        # start the timer
        start_time = time.time()

        for tick in ticker:

            # check if we're outta time
            if not _time_check(start_time, time.time()):
                return_payload = self._response_payload(
                    ticker, True, "Finished parsing {} Tickers before running out of time ".format(len(ticker)))
                return return_payload

            # get the corporate name as well as the ticker value
            corporate_name = self.db.get_company_name(tick)
            if corporate_name:
                for ch in string.punctuation:
                    corporate_name = corporate_name.replace(ch, "")

            print(corporate_name)

            articles = []
            aids = []

            for shortcode in named_shortcodes:
                headlines = self.filter.get_headlines(tick, shortcode)

                if corporate_name:

                    named_headlines = self.filter.get_headlines(
                        corporate_name.lower())
                    if named_headlines:
                        articles = articles + named_headlines

                if headlines:
                    articles = articles + headlines

            stats_object = self._parse_articles(articles, start_time)

            if not stats_object:
                continue
            aids = aids + stats_object['writtenlist']

            """TODO: Undeprecate
            if extract_sentiments:
                response = self.batch_update_sentiment(aids=aids)
            """
            for aid in aids:
                self.db.add_sentiment_to_queue(aid)

        return self._response_payload(ticker, True, "Finished crawling all articles for {} tickers".format(len(ticker)))

    def batch_update_sentiment(self, ticker_object=False, aids=False, timeframe=False, logger=False):
        """
        Crawls through indexed articles, extracting ticker names and sentiments and storing them in the
        article_sentiment_cache table.

        Parameters:
            - ticker_object :: A list of ticker dictionaries
            - aids :: A list of articles ID's to index
            - timeframe :: If provided, only articles starting from timeframe[0] to timeframe[1] will used
        """

        faulthandler.enable()

        def _name_strip(corporate_name):
            """
            Strips any superflous punctuation and corporate terms from a companies name
            """
            corporate_name = corporate_name.translate(
                str.maketrans('', '', string.punctuation))
            stopwords = ['corporation', 'corp', 'llc', 'company',
                         'limited liability company', 'inc', 'incorporated', 'limited']

            corporate_name = corporate_name.split(" ")
            corporate_name = [x for x in corporate_name if x not in stopwords]
            corporate_name = ' '.join(corporate_name)
            return corporate_name.lower()

        def _fuzz(ticker, entity, threshold=90):
            """
            Does a simple fuzzy search
            DEPRECATED SINCE ITS SINGLE CORE PERF IS SO SLOOOOOOW
            Reimplement when multiprocessing is supported
            """
            return ticker.lower() == entity.lower()
            rows = len(ticker) + 1
            cols = len(entity) + 1
            distance = np.zeros((rows, cols), dtype=int)

            for i in range(1, rows):
                for k in range(1, cols):
                    distance[i][0] = i
                    distance[0][k] = k

            for col in range(1, cols):
                for row in range(1, rows):
                    if ticker[row-1] == entity[col-1]:
                        cost = 0
                    else:
                        cost = 2

                    distance[row][col] = min(distance[row-1][col] + 1,
                                             distance[row][col-1] + 1,
                                             distance[row-1][col-1] + cost)

            ratio = ((len(ticker)+len(entity)) -
                     distance[row][col]) / (len(ticker)+len(entity))
            ratio = ratio * 100
            return ratio >= threshold

        ticker_obj = self.db.get_fundamentals(
            False) if not ticker_object else ticker_object

        tickers = {}
        now = datetime.datetime.now().strftime("%Y-%m-%d")
        for tick in ticker_obj:
            tickers[tick['ticker']] = _name_strip(tick['name'])

        num_tickers = len(ticker_obj)
        current_ticker = 0
        num_articles = 0
        no_articles = 0

        aids = aids if aids and type(
            aids) == list else self.db.list_article_ids()

        aids.reverse()
        body_stopwords = self.db._stopwords()

        # this dict is returned upon successful completion of the function
        stats_dict = {
            "num_failed_aids": 0,
            "num_nostock_aids": 0,
            "num_completed_aids": 0,
            "num_total_aids": len(aids),
            "nostock_aids": [],
            "failed_aids": []
        }

        for aid in aids:
            try:
                article = self.db.get_article_bodies(aid=aid)[0]
                entities = ast.literal_eval(article['entities'])
                named_stocks = []

                for ticker in list(tickers.keys()):
                    corporate_name = tickers[ticker]

                    for entity in entities:
                        if entities[entity] != "ORG":
                            continue
                        ticker_match = _fuzz(ticker, entity, 85)

                        corporate_match = _fuzz(corporate_name, entity, 85)
                        if ticker_match or corporate_match:
                            named_stocks.append(ticker)

                if len(named_stocks) == 0:
                    # no named stocks exiss
                    stats_dict['num_nostock_aids'] += 1
                    continue

                stemmed_sentences = self.filter._sentence_stemmer(
                    article['body'])
                for sentence in stemmed_sentences:
                    # get the sentences sentiment before we remove stopwords
                    sent = self.filter._sentiment(sentence)

                    if not sent:
                        continue

                    sentence = [x for x in sentence.split(
                        " ") if x not in body_stopwords]
                    sentence = " ".join(sentence)

                    for ticker in named_stocks:
                        if ticker in sentence or corporate_name in sentence:

                            self.db.update_general_sentiment(
                                {article['timestamp']: sent, "fragment": sentence}, ticker, 0)

                # log the aid as finished (if the logger is available)

                if logger:
                    logger.log(
                        aid, "DONE", "{}/batch.log".format(LOGGING_DIRECTORY))

            except Exception as e:
                if logger:
                    logger.log(
                        "[{}] EXCEPTION HAS OCCURED. {}".format(e), "ERROR")
                continue

        return stats_dict

    def _parse_articles(self, headlines, runtime):

        elapsed = int(time.time())
        robject = {
            "skipped": 0,
            "written": 0,
            "total": len(headlines),
            "skiplist": [],
            "writtenlist": []
        }

        for headline in headlines:
            try:

                timestamp = headline['date']

                if "body" in list(headline.keys()) and headline["headline"] != "Google News":
                    article = self.filter._format_article(
                        headline['link'], body_dict=headline)

                else:
                    article = self.filter.parse_article(headline['link'])

                if not article:
                    robject['skipped'] = robject['skipped'] + 1
                    robject['skiplist'].append(headline['link'])
                    continue

                article['timestamp'] = timestamp

                # commit to the database
                has_posted = self.db.insert_article(article)

                self._debug_log(
                    elapsed, headline['link'], len(has_posted) > 0)

                if not has_posted:
                    robject['skipped'] = robject['skipped'] + 1
                    robject['skiplist'].append(headline['link'])
                    continue

                robject['written'] = robject['written'] + len(has_posted)
                robject['writtenlist'] = robject['writtenlist'] + has_posted

            except Exception as e:

                self._debug_log(elapsed, "~~EXCEPTION~~ ({}) {}".format(
                    headline['link'], e), False)

                print(e)

                return robject

        return robject

    def _debug_log(self, runtime, url, has_posted):
        """
        Writes an entry to the debug log csv
        runtime - how long the crawler has been running
        url - URL of the article at that point
        has_posted Whether or not the article has posted
        """
        file_name = "{}/crawl_{}.csv".format(LOGGING_DIRECTORY, datetime.datetime.now().strftime("%Y-%m-%d"))
        with open(file_name, "a+") as f:
            f.write("{:.2f},{},{}\n".format(runtime, url, has_posted))
        return True

    def _response_payload(self, ticker, status, message):
        """
        Returns a formatted dict as a payload

        """
        payload = {
            "ticker": ticker,
            "response": {
                "status": status,
                "message": message
            }
        }
        return payload


class ArticleDatabaseHandler:
    def __init__(self, db=False, headless=False, redis=False, es=False):
        """
        Database abstraction layer to deal with the articles table
        """
        if not db and not headless:
            raise Exception("Either headless, db or both must be specified")
        self.db = db
        self.redis = redis
        # If false, the Elasticsearch article index will not update on new articles
        # TODO allow for setting this through the events API
        self.replicate_es = False
        if es:
            self.es = Elasticsearch(["localhost:9200"])
        else:
            self.es = None
        self.es_index_name = "vfin_articles"

        self.abstract_tree = None if not self.redis else AbstractTransformer()

    def insert_article(self, article_object):
        """
        Inserts either one article object or multiple objects. If the article_object
        is a list it will assume multiple objects exist. Returns a list of aids added to the database
        """
        if type(article_object) == dict:
            article_object = [article_object]

        if type(article_object) != list:
            return False

        aids = []
        for article in article_object:
            for k in list(article.keys()):
                if k == "sentiment" or k == "timestamp":
                    continue
                article[k] = str(article[k]).replace('"', '\"')
            aid = uuid.uuid4().hex
            a = article

            # check to see if the url is already cached
            check = self.check_article_url(a['url'])
            if check:
                continue

            if type(a['timestamp']) == str and len(a['timestamp']) == 2:

                # due to a bug with the XML structure of the PR Newswire source sometimes
                # the language short codes can be parsed as the 'date' field in the
                # NewsFilter headline parser. To remedy this we just set the timestamp
                # to whatever the current timestamp is
                date_int == datetime.datetime.now()

            elif a['timestamp'] and type(a['timestamp']) == float or type(a['timestamp']) == int:
                date_int = datetime.datetime.fromtimestamp(int(a['timestamp']))
            else:
                date_int = datetime.datetime.strptime(
                    a['timestamp'], "%Y-%m-%d %H:%M:%S")

            # hash the url
            self.hash_url(a["url"], aid)

            insertion_query = "INSERT INTO `articles`(`id`, `aid`, `url`, `timestamp`, `headline`, `body`, `summary`, `keywords`, `entities`, `sentiment`) VALUES(NULL, %s, %s, %s, %s, %s, %s, %s, %s, %s);"
            insert_object = (aid, a['url'], date_int, a['headline'], a['body'],
                             a['summary'], a['keywords'], a['entities'], a['sentiment'])

            results = self._exec(insertion_query, insert_object)

            if type(results) == bool and not results:
                print("ArticleCrawler Article Insertion Error")

            if self.replicate_es:
                a['aid'] = aid
                a['timestamp'] = date_int.isoformat()
                a['id'] = 0
                if not isinstance(a['sentiment'], float) or not isinstance(a['sentiment'], int):
                    a['sentiment'] = -1.0
                response = self.es.index(index=self.es_index_name, body=a)

            aids.append(aid)

        return aids

    def check_article_url(self, url):
        """
        Checks to see if the article is already cached. True if it is, False if it isn't
        """
        url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()
        selection_query = "SELECT `aid` FROM `article_hashes` WHERE `hash` = %s;"
        cursor = self.db.cursor()
        cursor.execute(selection_query, url_hash)
        results = cursor.fetchall()
        cursor.close()
        return len(results) != 0

    def hash_url(self, url, aid):
        """
        Inserts an aid into the article_hashes table
        """
        cursor = self.db.cursor()
        hashed_url = hashlib.sha256(url.encode("utf-8")).hexdigest()
        hash_query = "INSERT INTO `article_hashes`(`id`, `aid`, `hash`) VALUES(NULL, %s, %s);"
        cursor.execute(hash_query, (aid, hashed_url))
        cursor.close()

    def get_company_name(self, ticker):
        """
        Retrieves the name of a company given the companies ticker
        """
        cursor = self.db.cursor()
        sql = "SELECT `name` FROM `corporate_info` WHERE `ticker` = '{}' LIMIT 1;".format(
            ticker)
        results = self._exec(sql)
        if results:
            name = results[0]['name']
            return name
        return results

    def list_article_ids(self):
        """
        Returns a list of article ids

        Params:
            - page :: If page < 0 then every result will be returned. else pagination will be used with
                      each page resulting in a page_offset shift
        """
        list_query = "SELECT `aid` FROM `articles`;"

        results = self._exec(list_query)
        return [x['aid'] for x in results]

    def get_fundamentals(self, ticker):
        """
        Returns the fundamentals given a stock ticker or tickers.
        ticker can either be a string or an array of stock values
        """
        if ticker:
            get_query = "SELECT * FROM `corporate_info` WHERE `ticker` = %s LIMIT 1;"
            results = self._exec(get_query, ticker)
        else:
            get_query = "SELECT * FROM `corporate_info`;"
            results = self._exec(get_query)

        return results

    def _debug_log(self, runtime, url, has_posted):
        """
        Writes an entry to the debug log csv
        runtime - how long the crawler has been running
        url - URL of the article at that point
        has_posted Whether or not the article has posted
        """
        file_name = "{}/crawl_{}.csv".format(LOGGING_DIRECTORY, datetime.datetime.now().strftime("%Y-%m-%d"))
        with open(file_name, "a+") as f:
            f.write("{:.2f},{},{}\n".format(runtime, url, has_posted))
        return True

    def get_article_bodies(self, period=False, aid=False):
        """
        Gets a list of articles given the period. If period is false all articles are retrieved

        Parameters:
            - period :: An array with the first datetime object being the begin date and the 2nd datetime
                        object being the end date for the search

            - aid :: Can be a string or a list
            Returns:
                - return_array :: A return array of articles
        """
        if not period:
            get_query = "SELECT `aid`, `timestamp`, `body`, `entities`, `headline`, `sentiment` FROM `articles` WHERE `aid` = '{}' ORDER BY `timestamp` DESC;".format(
                aid)

        else:
            if type(period) != list:
                return False

            date_from = period[0]
            date_to = period[1]
            get_query = "SELECT `aid`, `url`, `timestamp`, `headline`, `body`, `summary`, `keywords`, `entities`, `sentiment` FROM `articles` WHERE `timestamp` BETWEEN '{}' AND '{}' ORDER BY `timestamp` DESC;".format(
                date_from, date_to)

        results = self._exec(get_query)
        return results

    def update_general_sentiment(self, sentiment_object, ticker, cache_type, check_cache=False):
        """
        Updates the `article_sentiment_cache` table, adding the new sentiments to it
        If check_cache is true, a query will be run to ensure the sentiment fragment hasn't already been cached
        """
        check_query = "SELECT COUNT(*) AS `count` FROM `article_sentiment_cache` WHERE `ticker` = '{}' AND `published_on` = '{}';"
        cache_update = "INSERT INTO `article_sentiment_cache`(`id`, `published_on`, `cached_on`, `type`, `ticker`, `sentiment`, `sentence_fragment`) VALUES(NULL, %s, %s, %s, %s, %s, %s);"
        cached_on = datetime.datetime.now().strftime("%Y-%m-%d")
        sentiment_keys = list(sentiment_object.keys())

        for sdate in sentiment_keys:
            sentiment = sentiment_object[sdate]
            fragment = sentiment_object["fragment"]
            if type(sdate) == str:
                published_on = datetime.datetime.strptime(sdate, "%Y-%m-%d")
            elif type(sdate) == datetime.datetime:
                published_on = sdate

            else:
                raise Exception("{} Type unsupported".format(str(type(sdate))))

            res = self._exec(cache_update, (published_on,
                             cached_on, cache_type, ticker, sentiment, fragment))

        return False

    def add_sentiment_to_queue(self, aid):
        """
        Adds an article to the sentiment extraction waitlist
        """
        cursor = self.db.cursor()
        insertion_query = "INSERT INTO `sentiment_extraction_waitlist`(`id`, `aid`, `cached_on`) VALUES (NULL, %s, NULL);"
        cursor.execute(insertion_query, aid)
        return True

    def _stopwords(self):
        """
        Returns a big list of stop words
        """
        with open(STOPWORDS_FILE_LOCATION) as f:
            stoplist = [line.rstrip('\n') for line in f]

        return stoplist

    def _exec(self, query, insert_obj=False):
        """
        Wrapper for executing a query. If the query is not successfully executed an error
        is raised
        """
        cursor = self.db.cursor()
        query = str(query)
        try:
            if insert_obj:
                cursor.execute(query, insert_obj)
            else:
                cursor.execute(query)
            results = cursor.fetchall()
            cursor.close()
            return results
        except Exception as e:
            print(e)
            return False

    def _serialize_dict(self, d,):
        string = str(d)
        return string.encode()

    def _deserialize_dict(self, serialized_dict):
        d = serialized_dict.decode('utf-8')
        decoded_tree = ast.parse(d, mode='eval')
        self.ast_transformer.visit(decoded_tree)
        clause = compile(decoded_tree, '<AST>', 'eval')

        # this is the most dangerous bit of code in this entire repo. if any bad data is
        # passed to this class during the deserialization of the ranked training data then
        decoded_dict = eval(clause, dict(Byte=bytes()))
        return decoded_dict


class ArticlePushService:
    def __init__(self, conf):
        """
        Configures the article push service

        Parameters:
            - conf (dict) :: The full configuration dict

        Returns:
            - bool :: True if success, False if not
        """
        self.server_name = socket.getfqdn()
        self.config = conf
        self.push_secret = self.config["ArticlePushSecret"]

    def push_articles(self, peer, articles):
        """
        Pushes an article to a peer given the peer name

        Parameters:
            - peer (str) :: The name of the peer to push to
            - article (list) :: The article to push

        Returns:
            - has_succeeded (bool)
            - msg (False|str) :: The error message if has_succeeded is False
        """
        peer_conf = False
        for server in self.config["ServerInformation"]:
            if server["ServerName"] == peer:
                peer_conf = server

        if not peer_conf:
            return False

        endpoint = "{}/api/internal/cache-article".format(
            peer_conf["BaseDomain"])
        payload = {
            "article_push_secret": self.push_secret,
            "data": articles
        }

        response = requests.post(url=endpoint, json=payload)
        try:
            response = response.json()
        except:
            return False, "Malformed response body"

        if not response["success"]:
            # log the error
            return False, response["message"]
        return True, False


class AbstractTransformer(ast.NodeTransformer):
    """
    Because ast.literal_eval cannot parse the byte() primitive
    type in python, we need to institute this abstract syntax tree
    hack to brute force the AST to visit our bytes
    """
    ALLOWED_NAMES = set(['Bytes', 'None', 'False', 'True'])
    ALLOWED_NODE_TYPES = set([
        'Expression',
        'Tuple',
        'Call',
        'Name',
        'Load',
        'Constant',
        'Str',
        'Bytes',
        'Num',
        'List',
        'Dict',
    ])

    def visit_name(self, node):
        if not node.id in self.ALLOWED_NAMES:
            raise RuntimeError(
                "Name access to {} is not allowed".format(node.id))

        return self.generic_visit(node)

    def generic_visit(self, node):
        nodetype = type(node).__name__
        if nodetype not in self.ALLOWED_NODE_TYPES:
            raise RuntimeError("Invalid nodetype {}".format(nodetype))

        return ast.NodeTransformer.generic_visit(self, node)
