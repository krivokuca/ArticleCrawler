"""
NewsFilter.py
Responsible for parsing and extracting information from articles given their schema (defined in the schema file)
Copyright 2023 Daniel Krivokuca <dan@voku.xyz>
"""
# pyright: reportMissingImports=false, reportUnusedVariable=warning, reportUnboundVariable=false
import json

import re
import subprocess
import time
import xml.etree.ElementTree as ET
from datetime import datetime


import feedparser
import html2text
import requests
from goose3 import Goose
from goose3.configuration import Configuration
from selenium import webdriver
from selenium.webdriver.firefox.options import Options

from ErrorLogger import ErrorLogger
from NLPEngine import NLPEngine

# SET THE MAIN SCHEMA FILE HERE
SCHEMA_FILE = "./schema"
class NewsFilter():
    '''
    This class acts as a wrapper for a news source, offering generic functions that scrape
    and parse articles from the website as well as provide summarization and NLP sentiment
    analysis of article content. NOTE* Instantiating this class also instatiates the NLPEngine
    which takes a long time thus making this class incredibly compute intensive if it needs
    to be instantiated multiple times.
    '''

    def __init__(self, shortcode=False):
        """
        Initialize the NewsFilter object with either the URL of an article, the shortcode of
        """
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.149 Safari/537.36"
        }
        self.shortcode = shortcode
        self.driver_opts = Options()
        self.driver_opts.headless = True
        self.html_cleaner = html2text.HTML2Text()
        self.html_cleaner.ignore_links = True
        self.html_cleaner.ignore_images = True
        self.driver = webdriver.Firefox(options=self.driver_opts)
        self.errorlogger = ErrorLogger()
        self.sympath = self.errorlogger._get_sympath()
        with open(SCHEMA_FILE, 'r') as f:
            self.schema = json.load(f)
        self.shortcodes = [x['Shortcode'] for x in self.schema['Sources']]
        self.vars = list(self.schema['Variables'].keys())
        self.headline_keys = self.schema['HeadlineKeys']
        self.xml_parsable_shortcodes = ['GNX', 'PRS']
        self.nlp = NLPEngine()

    def get_headlines(self, ticker, shortcode=False):
        '''
        Gets the headlines given the stocks ticker
        '''

        endpoint = False
        shortcode = self.shortcode if not shortcode else shortcode
        if not shortcode or shortcode not in self.shortcodes:
            return False

        for source in self.schema['Sources']:
            if source['Shortcode'] == shortcode:
                iterkey = source['IteratorKey']
                keyvals = source['KeyVals']
                endpoint = source['HeadlineEndpoint']
                isjson = (source['EndpointType'] == 'JSON')
                headlinekeys = self.headline_keys
                # because of the 'language' value in the HeadlineKeys schema value we
                # need to filter it out if its not in the HasLanguageKey array
                if shortcode not in self.schema['HasLanguageKeys']:
                    headlinekeys = headlinekeys[0:4]

        if not endpoint:
            return False

        endpoint = endpoint if self.vars[0] not in endpoint else endpoint.replace(
            self.vars[0], ticker)

        if not isjson:
            return self.get_xml(shortcode, ticker, url=endpoint)

        if shortcode in ['ABN']:
            response = requests.get(endpoint)
            json_data = json.loads(response.content.decode('utf-8'))
            article_list = self.parse_apjson(json_data)
            return article_list

        try:
            response = requests.get(endpoint)

            if response.status_code == 200:

                data = json.loads(response.content.decode('utf-8'))

                iterkey = data if not iterkey else data[iterkey]
                results = []
                for entry in iterkey:
                    entry_obj = {}

                    for i in range(len(headlinekeys)):
                        if not entry[keyvals[i]]:
                            continue

                        # less sloppy date --> unix timestamp conversion
                        if (i == 3 and type(entry[keyvals[i]]) is not int and shortcode in ['GNH', 'GNW']):
                            ts = datetime.strptime(
                                entry[keyvals[i]], "%Y-%m-%dT%H:%M:%SZ").timestamp()
                            entry_obj[headlinekeys[i]] = ts

                        elif (i == 2 and '<' in entry[keyvals[i]]):
                            entry_obj[headlinekeys[i]] = re.sub(
                                r'<(.*?)>', '', entry[keyvals[i]])
                        else:
                            entry_obj[headlinekeys[i]
                                      ] = entry[keyvals[i]]
                    results.append(entry_obj)

                # remove duplicates from the results
                urls = []
                orig_results = []
                for result in results:
                    if result['link'] not in urls:
                        urls.append(result['link'])
                        orig_results.append(result)
                    else:
                        continue

                return orig_results

            else:
                print("not found")
                return False

        except Exception as e:
            self.errorlogger.log(
                2, e)
            return False

    def get_xml(self, shortcode, ticker=False, url=False):

        def _iter_node_children(parent):
            child = parent.firstChild
            while child != None:
                yield child
                child = child.nextSibling
        '''
        Parses a source if it's XML
        '''
        if shortcode in self.xml_parsable_shortcodes:
            schema = False
            for source in self.schema['Sources']:
                if source['Shortcode'] in self.xml_parsable_shortcodes:
                    schema = source

            if not schema:
                raise Exception(
                    "Schema for Shortcode `{}` not found".format(shortcode))

            url = schema['HeadlineEndpoint'] if not url else url
            url = re.sub(r"\{(.*?)\}", ticker, url) if ticker else url
            response = requests.get(url)
            if response.status_code == 200:
                try:
                    data = response.content.decode('utf-8')
                    root = ET.ElementTree(ET.fromstring(data)).getroot()

                except Exception as e:
                    print(e)
                else:
                    results = []
                    for element in root.findall("channel/item"):
                        element_dict = {}
                        element_lang_flag = True
                        for child in list(element):
                            tag = child.tag

                            if tag not in schema['KeyVals']:
                                continue

                            i = schema['KeyVals'].index(tag)
                            text = self.html_cleaner.handle(child.text)

                            if i == 0:
                                text = text.strip().replace(
                                    "\n", "").replace('\u00ad', '')
                            else:
                                text = text.strip().replace(
                                    "\n", " ").replace('\u00ad', '')

                            if i == 3:
                                if shortcode == "PRS":
                                    text = text.replace("+0000", "")
                                text = text.replace("GMT", "")
                                text = text.strip().rstrip().replace("\n", "").replace('\u00ad', '')
                                dt = datetime.strptime(
                                    text, "%a, %d %b %Y %H:%M:%S")
                                text = dt.strftime("%Y-%m-%d %H:%M:%S")

                            if i == 4:
                                text = text.replace("\n", "")
                                if len(text) > 0 and text != "en-US":

                                    element_lang_flag = False

                            element_dict[self.schema['HeadlineKeys'][i]] = text

                        if element_lang_flag:
                            results.append(element_dict)
                    return (results)

        else:
            f = feedparser.parse(url)
            results = []
            for e in f['entries']:
                entry = {}
                ts = re.sub(r'(-\d{4})', '', e['published'])
                ts = ts.rstrip()
                timestamp = time.mktime(datetime.strptime(
                    ts, "%a, %d %b %Y %H:%M:%S").timetuple())
                entry[self.headline_keys[0]] = e['link']
                entry[self.headline_keys[1]] = e['title']
                entry[self.headline_keys[2]] = False
                entry[self.headline_keys[3]] = timestamp
                results.append(entry)
            return results

    def parse_apjson(self, page):
        """
        Parses the Associated Press JSON page
        """
        articles = []
        for card in page['cards']:

            if card is None:

                continue

            if len(card['contents']) == 0:
                continue

            published_date = card['publishedDate']
            if published_date:
                published_date = time.mktime(datetime.strptime(
                    card['publishedDate'], "%Y-%m-%d %H:%M:%S").timetuple())
            else:
                published_date = datetime.now()
            article_dict = {
                'link': card['contents'][0]['localLinkUrl'],
                'headline': card['contents'][0]['headline'],
                'summary': card['contents'][0]['flattenedFirstWords'],
                'date': published_date,
                'body': self.html_cleaner.handle(
                    card['contents'][0]['storyHTML']).replace("\n", " ")
            }
            articles.append(article_dict)

        return articles

    def get_stocklist(self):
        '''
        Returns a dict of the stock list
        '''
        return False

    def parse_article(self, url):
        """
        Takes the articles URL and extracts all available information from the article using the NLPEngine

        Parameters:
            - url (str) :: The URL for the page

        Returns:
            - output(dict) :: A dictionary containing the following keys:
                [url, timestamp, headline, body, summary, keywords, entities]
        """

        if "https://news.google.com/" == url[0:24]:
            url = url.split("/")
            url.insert(3, "__i")
            url.insert(5, "rd")
            url = "/".join(url)

        with Goose({'browser_user_agent': self.headers['User-Agent'], 'strict': False}) as g:
            try:
                article = g.extract(url=url)
            except Exception:
                article_html = self.driver.get(url)
                article = g.extract(raw_html=article_html)

            content = self._format_article(url, article)

        return content

    def _deprecated_parse_article(self, url):
        '''
        Takes an article and extracts all the available information
        from the article using the NLPEngine. The function returns a
        dict with the following keys
            [url, timestamp, headline, body, summary, keywords, entities]
        '''

        try:
            config = Configuration()
            config.strict = False
            with Goose({'browser_user_agent': self.headers['User-Agent'], 'strict': False}) as g:
                article = g.extract(url=url)
                if (article.title == "Access to this page has been denied." or article.cleaned_text == ""):
                    article_html = self._parse_article_nojs(
                        url, output_html=True)
                    article = g.extract(raw_html=article_html)
                content = self._format_article(url, article)
                g.close()

                return content

        except Exception as e:
            # log the error
            print(e)
            self.errorlogger.log(
                2, "Could not establish connection to {}, 403. Retrying....".format(url))

    def _parse_article_nojs(self, url, output_html=False):
        """
        If no Javascript is available this function uses a headless Firefox
        session to load the DOM into and then returns the HTML. Because of
        the increased overhead, this function is much more RAM-hungry and
        takes longer to execute than the regular parse_article function
        Returns : the html of the page
        """
        self.driver.get(url)
        source = self.driver.page_source

        self.driver.close()
        if output_html:
            return source

        with Goose() as g:
            article = g.extract(raw_html=source)
            content = self._format_article(url, article)
            return content

    def _format_article(self, url, article=False, body_dict=False):
        return_content = {}
        if article:
            return_content['url'] = url
            return_content['headline'] = article.title
            return_content['body'] = article.cleaned_text.replace(
                "\n", "").replace("\t", "")
            return_content['summary'] = self.nlp.get_summary(
                return_content['headline'], return_content['body'], num_sentences=8)

        elif body_dict and type(body_dict) == dict:
            return_content['url'] = body_dict['link']
            return_content['headline'] = body_dict['headline']
            return_content['body'] = body_dict['body']
            return_content['summary'] = self.nlp.get_summary(
                return_content['headline'], return_content['body'], num_sentences=8)

        return_content['keywords'] = list(
            self.nlp.get_keywords(return_content['body']).keys())

        list_limiter = min(len(return_content['keywords']), 11)

        return_content['keywords'] = return_content['keywords'][0:list_limiter]
        return_content['entities'] = self.nlp.get_entities(
            return_content['headline'], return_content['body'])

        return_content['sentiment'] = self.nlp.get_sentiment(
            return_content['body'])
        return return_content

    def get_sources(self):
        return self.schema['Sources']

    def get_schema(self):
        return self.schema

    def _test(self):
        url = 'https://news.google.com/__i/rss/rd/articles/CBMiM2h0dHBzOi8vbmV3cy51bWFuaXRvYmEuY2EvbWVkaWNhbC1maXRuZXNzLWZhY2lsaXR5L9IBAA?oc=5'
        article = self.parse_article(url)
        print(article)

    def _sentence_stemmer(self, body):
        """
        Stems a body into sentences
        """
        return self.nlp.stem_sentences(body)

    def _sentiment(self, text):
        return self.nlp.get_sentiment(text)

    def _kill_browser(self):
        """
        Kills any running firefox headless instances
        """
        subprocess.run(['pkill', '-f', '"firefox"'],
                       stdout=subprocess.PIPE).stdout().read()
        return True

    def _close_driver(self):
        self.driver.close()
