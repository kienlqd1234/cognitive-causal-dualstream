#!/usr/local/bin/python
import logging
import logging.config
import yaml
import itertools
import os
import io
import json
import sys


class PathParser:

    def __init__(self, config_path):
        self.root = './'
        self.log = os.path.join(self.root, config_path['log'])

        self.data = os.path.join(self.root, config_path['data'])
        self.res = os.path.join(self.root, config_path['res'])
        self.graphs = os.path.join(self.root, config_path['graphs'])
        self.checkpoints = os.path.join(self.root, config_path['checkpoints'])

        self.glove = os.path.join(self.res, config_path['glove'])

        self.retrieved = os.path.join(self.data, config_path['tweet_retrieved'])
        self.preprocessed = os.path.join(self.data, config_path['tweet_preprocessed'])
        self.movement = os.path.join(self.data, config_path['price'])
        self.vocab = os.path.join(self.res, config_path['vocab_tweet'])


def update_config(config_path):
    """Update the configuration with a new config file."""
    global config, config_model, dates, config_stocks, stock_symbols, ss_size, path_parser
    
    # Load new config
    with open(config_path, 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    
    # Update all dependent variables
    config_model = config['model']
    #config_path_dict = config['paths'] # Store the paths dictionary
    dates = config['dates']
    config_stocks = config['stocks']
    
    stock_symbols = list(itertools.chain.from_iterable(config_stocks.values()))
    ss_size = len(stock_symbols)
    path_parser = PathParser(config_path=config['paths'])


config_fp = os.path.join(os.path.dirname(__file__), 'config.yml')
#config = yaml.load(file(config_fp, 'r'))
with open(config_fp, 'r') as config_file:
    config = yaml.load(config_file, Loader=yaml.FullLoader)
config_model = config['model']

dates = config['dates']

config_stocks = config['stocks']  # a list of lists
stock_symbols = list(itertools.chain.from_iterable(config_stocks.values()))
#list_of_lists = [config_stocks[key] for key in config_stocks]
#stock_symbols = list(itertools.chain.from_iterable(list_of_lists))
#stock_symbols = config_stocks
ss_size = len(stock_symbols)

path_parser = PathParser(config_path=config['paths'])

# logger
logger = logging.getLogger('my_logger')

# Only configure if no handlers are already present for this logger
if not logger.hasHandlers():
    logger.setLevel(logging.DEBUG)

    log_dir = os.path.dirname(path_parser.log)
    os.makedirs(log_dir, exist_ok=True)  # Create the log directory if it doesn't exist

    log_fp = os.path.join(path_parser.log, '{0}.log'.format('model'))
    file_handler = logging.FileHandler(log_fp)
    console_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    # Consider adding logger.propagate = False if this configuration is applied

with io.open(str(path_parser.vocab), 'r', encoding='utf-8') as vocab_f:
    vocab = json.load(vocab_f)
    vocab_size = len(vocab) + 1  # for unk
