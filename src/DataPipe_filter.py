#!/usr/local/bin/python
import os
import io
import json
import numpy as np
from datetime import datetime, timedelta
import random
from ConfigLoader import logger, path_parser, config_model, dates, stock_symbols, vocab, vocab_size, config as global_config
from MeaningAwareSelection import MeaningAwareSelection



class DataPipe:

    def __init__(self):
        # load path
        self.movement_path = path_parser.movement
        self.tweet_path = path_parser.preprocessed
        self.vocab_path = path_parser.vocab
        self.glove_path = path_parser.glove

        # load dates
        self.train_start_date = dates['train_start_date']
        self.train_end_date = dates['train_end_date']
        self.dev_start_date = dates['dev_start_date']
        self.dev_end_date = dates['dev_end_date']
        self.test_start_date = dates['test_start_date']
        self.test_end_date = dates['test_end_date']

        # load model config
        self.batch_size = config_model['batch_size']
        self.shuffle = config_model['shuffle']

        self.max_n_days = config_model['max_n_days']
        self.max_n_words = config_model['max_n_words']
        self.max_n_msgs = config_model['max_n_msgs']

        self.word_embed_type = config_model['word_embed_type']
        self.word_embed_size = config_model['word_embed_size']
        self.stock_embed_size = config_model['stock_embed_size']
        self.init_stock_with_word= config_model['init_stock_with_word']
        self.price_embed_size = config_model['word_embed_size']
        self.y_size = config_model['y_size']

        assert self.word_embed_type in ('rand', 'glove')

        # Initialize MeaningAwareSelection with a configurable NLI model, defaulting to bert-base-uncased
        # Allow override from global_config.yml (e.g., LLM_SETTINGS.nli_model_name)
        # default_nli_model = "bert-base-uncased" # Could be "textattack/bert-base-uncased-mnli" for better NLI
        # nli_model_to_use = global_config.get('LLM_SETTINGS', {}).get('nli_model_name', default_nli_model)
        
        # logger.info(f"DataPipe_filter is configuring MeaningAwareSelection with NLI model: {nli_model_to_use}")
        local_model_path = r"D:\FinalYear\KLTN\PEN\PEN-main\PEN-main\LLM_Model\GPT2_FINGPT_QA" # Using raw string for Windows path
        logger.info(f"DataPipe_filter is configuring MeaningAwareSelection with local FinGPT model path: {local_model_path}")

        llm_config_for_mas = {
            "type": "fingpt_gpt2_qa_generative", 
            "model_name_or_path": local_model_path, # Use the local path
            "max_new_tokens": 10, 
            "temperature": 0.1,
            #"type": "hf_nli_relevance", 
            #"model_name_or_path": nli_model_to_use,
            # "hypothesis_template": "This news is important for the {ticker} company." # Optional: override default
            # "device": "cuda"  # Optional: uncomment to force cuda
        }
        self.meaning_aware_selector = MeaningAwareSelection(config={"llm_config": llm_config_for_mas})

        # Build and store mappings
        self.word2id = self.index_token(vocab, key='token', type='word')
        self.idx2word = self.index_token(vocab, key='id', type='word')
        self.stock2id = self.index_token(stock_symbols, key='token', type='stock')
        self.idx2stock = self.index_token(stock_symbols, key='id', type='stock')

    @staticmethod
    def _convert_token_to_id(token, token_id_dict):
        if token not in token_id_dict:
            token = 'UNK'
        return token_id_dict[token]

    def _get_start_end_date(self, phase):
        """
            phase: train, dev, test, unit_test
            => start_date & end_date
        """
        assert phase in {'train', 'dev', 'test', 'whole', 'unit_test'}
        if phase == 'train':
            return self.train_start_date, self.train_end_date
        elif phase == 'dev':
            return self.dev_start_date, self.dev_end_date
        elif phase == 'test':
            return self.test_start_date, self.test_end_date
        elif phase == 'whole':
            return self.train_start_date, self.test_end_date
        else:
            return '2012-07-23', '2012-08-05'  # '2014-07-23', '2014-08-05'

    def _get_batch_size(self, phase):
        """
            phase: train, dev, test, unit_test
        """
        if phase == 'train':
            return self.batch_size
        elif phase == 'unit_test':
            return 5
        else:
            return min(32, self.batch_size)  # Use a reasonable batch size for eval

    def index_token(self, token_list, key='id', type='word'):
        assert key in ('id', 'token')
        assert type in ('word', 'stock')
        indexed_token_dict = dict()

        if type == 'word':
            token_list_cp = list(token_list)  # un-change the original input
            token_list_cp.insert(0, 'UNK')  # for unknown tokens
        else:
            token_list_cp = token_list

        if key == 'id':
            for id_val in range(len(token_list_cp)): # Renamed id to id_val to avoid shadowing built-in
                indexed_token_dict[id_val] = token_list_cp[id_val]
        else:
            for id_val in range(len(token_list_cp)):  # Renamed id to id_val
                try:
                    token = token_list_cp[id_val]
                    indexed_token_dict[token] = id_val
                except (IndexError, KeyError) as e:
                    print(f"Error accessing token_list_cp[{id_val}]: {e}") # Use id_val
                continue
        return indexed_token_dict

    def build_stock_id_word_id_dict(self):
        # load vocab, user, stock list
        stock_id_word_id_dict = dict()

        vocab_id_dict = self.index_token(vocab, key='token')
        id_stock_dict = self.index_token(stock_symbols, type='stock')

        for (stock_id, stock_symbol) in id_stock_dict.items():
            stock_symbol_lower = stock_symbol.lower() # Use a different var name
            if stock_symbol_lower in vocab_id_dict:
                stock_id_word_id_dict[stock_id] = vocab_id_dict[stock_symbol_lower]
            else:
                stock_id_word_id_dict[stock_id] = None
        return stock_id_word_id_dict

    def _convert_words_to_ids(self, words, vocab_id_dict):
        """
            Replace each word in the data set with its index in the dictionary

        :param words: words in tweet
        :param vocab_id_dict: dict, vocab-id
        :return:
        """
        return [self._convert_token_to_id(w, vocab_id_dict) for w in words]

    def _get_prices_and_ts(self, ss, main_target_date):

        def _get_mv_class(data, use_one_hot=False):
            mv = float(data[1])
            if self.y_size == 2:
                if mv <= 1e-7:
                    return [1.0, 0.0] if use_one_hot else 0
                else:
                    return [0.0, 1.0] if use_one_hot else 1

            if self.y_size == 3:
                threshold_1, threshold_2 = -0.004, 0.005
                if mv < threshold_1:
                    return [1.0, 0.0, 0.0] if use_one_hot else 0
                elif mv < threshold_2:
                    return [0.0, 1.0, 0.0] if use_one_hot else 1
                else:
                    return [0.0, 0.0, 1.0] if use_one_hot else 2

        def _get_y(data):
            return _get_mv_class(data, use_one_hot=True)

        def _get_prices(data):
            return [float(p) for p in data[3:6]]

        def _get_mv_percents(data):
            return _get_mv_class(data)

        ts, ys, prices, mv_percents, main_mv_percent = list(), list(), list(), list(), 0.0
        d_t_min = main_target_date - timedelta(days=self.max_n_days-1)

        stock_movement_path = os.path.join(str(self.movement_path), '{}.txt'.format(ss))
        with io.open(stock_movement_path, 'r', encoding='utf8') as movement_f:
            for line in movement_f:  # descend
                data = line.split('\t')
                t = datetime.strptime(data[0], '%Y-%m-%d').date()
                if t == main_target_date:
                    ts.append(t)
                    ys.append(_get_y(data))
                    main_mv_percent = data[1]
                    if -0.005 <= float(main_mv_percent) < 0.0055:
                        return None
                if d_t_min <= t < main_target_date:
                    ts.append(t)
                    ys.append(_get_y(data))
                    prices.append(_get_prices(data))
                    mv_percents.append(_get_mv_percents(data))
                if t < d_t_min:
                    prices.append(_get_prices(data))
                    mv_percents.append(_get_mv_percents(data))
                    break

        T_val = len(ts) # Renamed T to T_val
        if len(ys) != T_val or len(prices) != T_val or len(mv_percents) != T_val:
            return None

        for item in (ts, ys, mv_percents, prices):
            item.reverse()

        prices_and_ts = {
            'T': T_val, # Use T_val
            'ts': ts,
            'ys': ys,
            'main_mv_percent': main_mv_percent,
            'mv_percents': mv_percents,
            'prices': prices,
        }

        return prices_and_ts

    def _get_unaligned_corpora(self, ss, main_target_date, vocab_id_dict_param): # Renamed vocab_id_dict
        def get_ss_index(word_seq, stock_symbol_str):
            stock_symbol_query = stock_symbol_str.lower()
            ss_idx = len(word_seq) - 1 # Renamed ss_index to ss_idx
            if stock_symbol_query in word_seq:
                ss_idx = word_seq.index(stock_symbol_query)
            else:
                if '$' in word_seq:
                    dollar_index = word_seq.index('$')
                    if dollar_index + 1 < len(word_seq) and stock_symbol_query == word_seq[dollar_index + 1]:
                         ss_idx = dollar_index + 1
                    elif dollar_index + 1 < len(word_seq) and stock_symbol_query in word_seq[dollar_index + 1]:
                        ss_idx = dollar_index + 1
                    else:
                        for index_val in range(dollar_index + 1, len(word_seq)): # Renamed index to index_val
                            if stock_symbol_query in word_seq[index_val]:
                                ss_idx = index_val
                                break
            return ss_idx

        unaligned_corpora = list()
        stock_tweet_path = os.path.join(str(self.tweet_path), ss)

        d_d_max = main_target_date - timedelta(days=1)
        d_d_min = main_target_date - timedelta(days=self.max_n_days)

        current_day = d_d_max # Renamed d to current_day
        while current_day >= d_d_min:
            msg_fp = os.path.join(stock_tweet_path, current_day.isoformat())
            daily_raw_texts = []
            if os.path.exists(msg_fp):
                with open(msg_fp, 'r') as tweet_f:
                    for line in tweet_f:
                        try:
                            msg_dict = json.loads(line)
                            text = msg_dict['text']
                            if text:
                                daily_raw_texts.append(text)
                        except json.JSONDecodeError:
                            logger.warning(f"Skipping malformed JSON line in {msg_fp}: {line.strip()}")
                            continue
                        except KeyError:
                            logger.warning(f"Skipping line with missing 'text' field in {msg_fp}: {line.strip()}")
                            continue
            
            if not daily_raw_texts:
                current_day -= timedelta(days=1)
                continue

            try:
                relevant_texts, _ = self.meaning_aware_selector.filter_texts(daily_raw_texts, ss)
            except Exception as e:
                logger.error(f"Error during MeaningAwareSelection for stock {ss}, day {current_day}: {e}")
                relevant_texts = []

            if relevant_texts:
                word_mat = np.zeros([self.max_n_msgs, self.max_n_words], dtype=np.int32)
                n_word_vec = np.zeros([self.max_n_msgs, ], dtype=np.int32)
                ss_index_vec = np.zeros([self.max_n_msgs, ], dtype=np.int32)
                msg_counter = 0 # Renamed msg_id to msg_counter

                for rel_text in relevant_texts:
                    if msg_counter >= self.max_n_msgs:
                        break
                    
                    words_tokenized = rel_text.split()
                    words = words_tokenized[:self.max_n_words]
                    
                    word_ids = self._convert_words_to_ids(words, self.word2id) # Using self.word2id
                    n_words = len(word_ids)

                    if n_words > 0:
                        n_word_vec[msg_counter] = n_words
                        word_mat[msg_counter, :n_words] = word_ids
                        ss_index_vec[msg_counter] = get_ss_index(words, ss) 
                        msg_counter += 1
                
                if msg_counter > 0:
                    n_msgs_val = msg_counter # Renamed n_msgs to n_msgs_val
                    unaligned_corpora.append((current_day, word_mat[:n_msgs_val], n_word_vec[:n_msgs_val], ss_index_vec[:n_msgs_val], n_msgs_val))
            
            current_day -= timedelta(days=1)

        return unaligned_corpora

    def _trading_day_alignment(self, ts, T_param, unaligned_corpora_list): # Renamed T and unaligned_corpora
        aligned_word_tensor = np.zeros([T_param, self.max_n_msgs, self.max_n_words], dtype=np.int32)
        aligned_ss_index_mat = np.zeros([T_param, self.max_n_msgs], dtype=np.int32)
        aligned_n_words_mat = np.zeros([T_param, self.max_n_msgs], dtype=np.int32)
        aligned_n_msgs_vec = np.zeros([T_param, ], dtype=np.int32)

        aligned_msgs = [[] for _ in range(T_param)]
        aligned_ss_indices = [[] for _ in range(T_param)]
        aligned_n_words = [[] for _ in range(T_param)]
        aligned_n_msgs = [[] for _ in range(T_param)] # This was sum of n_msgs, should be list of counts

        corpus_t_indices = []
        max_threshold = 0 # This seems unused or needs context

        for corpus_item in unaligned_corpora_list: # Renamed corpus to corpus_item
            day_val = corpus_item[0] # Renamed d to day_val
            found_t = False
            for t_idx in range(T_param): # Renamed t to t_idx
                if day_val < ts[t_idx]: # Assuming ts is sorted ascendingly, this logic might need review
                    corpus_t_indices.append(t_idx)
                    found_t = True
                    break
            if not found_t: # If day_val is >= all ts, append last index or handle as per logic
                 corpus_t_indices.append(T_param -1 if T_param > 0 else 0)


        if len(corpus_t_indices) != len(unaligned_corpora_list):
             logger.warning(f"Mismatch in corpus_t_indices ({len(corpus_t_indices)}) and unaligned_corpora ({len(unaligned_corpora_list)}) lengths.")
             # Fallback or error handling needed here, for now, proceed with potential issues.
        

        for i in range(len(unaligned_corpora_list)):
            corpus_data = unaligned_corpora_list[i] # Renamed corpus to corpus_data
            # Ensure corpus_t_indices[i] is a valid index for aligned lists
            if i >= len(corpus_t_indices): # Safety check
                logger.warning(f"Index {i} out of bounds for corpus_t_indices. Skipping this corpus item.")
                continue
            target_t_idx = corpus_t_indices[i]
            
            # Unpack: (date_obj, word_matrix_for_day, n_words_for_each_msg, ss_indices_for_each_msg, num_relevant_msgs_for_day)
            _, word_mat, n_word_vec, ss_index_vec, n_msgs_count = corpus_data # Renamed n_msgs to n_msgs_count

            # Original logic seemed to aggregate messages if multiple unaligned_corpora items mapped to the same t_idx
            # The append structure: (d, word_mat[:n_msgs], n_word_vec[:n_msgs], ss_index_vec[:n_msgs], n_msgs)
            # word_mat itself is already shaped [n_msgs, max_n_words] for that day.
            
            # If current target_t_idx already has messages, this extends them.
            # This assumes that if multiple entries in unaligned_corpora_list map to the same day in `ts`,
            # their messages should be concatenated for that day.
            # The original structure of unaligned_corpora per day needs to be clear.
            # Let's assume for now that each entry in unaligned_corpora_list corresponds to a unique day's worth of *filtered* messages.
            # And `corpus_t_indices` maps it to the correct slot in the `ts` timeline.
            
            # Simplified: if unaligned_corpora provides one entry per unique day from d_d_min to d_d_max
            # and ts aligns with these days.
            # The original logic of extending lists might be for a different structure of unaligned_corpora.
            # Re-evaluating alignment based on the assumption: 
            # unaligned_corpora_list has (date, daily_word_mat, daily_n_word_vec, daily_ss_idx_vec, daily_n_msg_count)
            # And ts is the target timeline. We need to map data from unaligned_corpora_list's dates to ts's dates.

            if 0 <= target_t_idx < T_param: # Ensure target_t_idx is valid
                # Store directly, assuming one unaligned_corpora entry per relevant day.
                # If a day in `ts` has no corresponding entry in `unaligned_corpora_list` after filtering, it will remain zeros.
                
                # This section needs careful review based on how `ts` and `unaligned_corpora_list` dates match.
                # The original approach of extending lists suggests multiple entries in unaligned_corpora_list
                # could map to the same `t` in `ts`.
                # For now, let's stick to the direct assignment for a single day's processed data.
                # The `corpus_t_indices` logic maps the unaligned day to the `ts` index.
                
                # Replace previous content for this t_idx, or decide on aggregation strategy
                aligned_msgs[target_t_idx] = word_mat # word_mat is [n_msgs_count, max_n_words]
                aligned_ss_indices[target_t_idx] = ss_index_vec # [n_msgs_count]
                aligned_n_words[target_t_idx] = n_word_vec # [n_msgs_count]
                aligned_n_msgs[target_t_idx] = n_msgs_count # scalar: number of messages for this day


        def is_eligible(): # Eligibility logic might need review based on new data structure
            # Original: n_fails = len([0 for n_msgs_list_for_day in aligned_n_msgs if sum(n_msgs_list_for_day) == 0])
            # New: aligned_n_msgs[t] is now a scalar (count of msgs for day t) or an empty list if no data.
            n_zero_msg_days = 0
            for n_msg_count_for_day in aligned_n_msgs:
                if isinstance(n_msg_count_for_day, int) and n_msg_count_for_day == 0:
                    n_zero_msg_days +=1
                elif not n_msg_count_for_day: # Handles empty list case if a `t` was not populated
                    n_zero_msg_days +=1
            return n_zero_msg_days <= max_threshold # max_threshold is 0, so no days can have zero messages.

        if not is_eligible():
            # This eligibility check might be too strict if some days legitimately have no relevant news.
            # logger.warning(f"Sample for stock {ss} on {main_target_date} not eligible due to zero relevant messages on too many days.")
            return None


        for t_idx in range(T_param): # Renamed t to t_idx
            # n_msgs_for_day is now a scalar from aligned_n_msgs[t_idx]
            n_msgs_for_day = aligned_n_msgs[t_idx]
            if not isinstance(n_msgs_for_day, int) or n_msgs_for_day == 0 : # If it's an empty list or 0 msgs
                aligned_n_msgs_vec[t_idx] = 0
                continue

            current_n_msgs = min(n_msgs_for_day, self.max_n_msgs)
            aligned_n_msgs_vec[t_idx] = current_n_msgs

            if current_n_msgs > 0:
                # aligned_msgs[t_idx] should be the word_mat for that day already
                day_word_mat = aligned_msgs[t_idx]
                day_ss_indices = aligned_ss_indices[t_idx]
                day_n_words_vec = aligned_n_words[t_idx]

                if day_word_mat is not None and len(day_word_mat) >= current_n_msgs:
                     aligned_word_tensor[t_idx, :current_n_msgs] = day_word_mat[:current_n_msgs]
                if day_ss_indices is not None and len(day_ss_indices) >= current_n_msgs:
                     aligned_ss_index_mat[t_idx, :current_n_msgs] = day_ss_indices[:current_n_msgs]
                if day_n_words_vec is not None and len(day_n_words_vec) >= current_n_msgs:
                     aligned_n_words_mat[t_idx, :current_n_msgs] = day_n_words_vec[:current_n_msgs]


        aligned_info_dict = {
            'msgs': aligned_word_tensor,
            'ss_indices': aligned_ss_index_mat,
            'n_words': aligned_n_words_mat,
            'n_msgs': aligned_n_msgs_vec,
        }

        return aligned_info_dict

    def sample_gen_from_one_stock(self, vocab_id_dict_param, stock_id_dict_param, s, phase): # Renamed params
        start_date, end_date = self._get_start_end_date(phase)
        stock_movement_path = os.path.join(self.movement_path, f'{s}.txt')
        main_target_dates = []
        
        if not os.path.exists(stock_movement_path):
            logger.error(f"File {stock_movement_path} not found.")
            return

        with open(stock_movement_path, 'r') as movement_f:
            for line in movement_f:
                data = line.split('\t')
                main_target_date = datetime.strptime(data[0], '%Y-%m-%d').date()
                main_target_date_str = main_target_date.isoformat()

                if start_date <= main_target_date_str < end_date:
                    main_target_dates.append(main_target_date)

        if self.shuffle:
            random.shuffle(main_target_dates)

        for main_target_date_val in main_target_dates: # Renamed main_target_date
            unaligned_corpora = self._get_unaligned_corpora(s, main_target_date_val, self.word2id) # Use self.word2id
            prices_and_ts = self._get_prices_and_ts(s, main_target_date_val)
            if not prices_and_ts:
                continue

            aligned_info_dict = self._trading_day_alignment(prices_and_ts['ts'], prices_and_ts['T'], unaligned_corpora)
            if not aligned_info_dict:
                continue

            sample_dict = {
                'stock': self._convert_token_to_id(s, self.stock2id), # Use self.stock2id
                'main_target_date': main_target_date_val.isoformat(),
                'T': prices_and_ts['T'],
                'ts': prices_and_ts['ts'],
                'ys': prices_and_ts['ys'],
                'main_mv_percent': prices_and_ts['main_mv_percent'],
                'mv_percents': prices_and_ts['mv_percents'],
                'prices': prices_and_ts['prices'],
                'msgs': aligned_info_dict['msgs'],
                'ss_indices': aligned_info_dict['ss_indices'],
                'n_words': aligned_info_dict['n_words'],
                'n_msgs': aligned_info_dict['n_msgs'],
            }

            yield sample_dict

    def batch_gen(self, phase):
        batch_size_val = self._get_batch_size(phase) # Renamed batch_size
        # vocab_id_dict and stock_id_dict are no longer needed here as class members self.word2id and self.stock2id are used.
        
        generators = [self.sample_gen_from_one_stock(self.word2id, self.stock2id, s, phase) for s in stock_symbols]

        while True:
            stock_batch = np.zeros([batch_size_val, ], dtype=np.int32)
            T_batch = np.zeros([batch_size_val, ], dtype=np.int32)
            
            y_batch = np.zeros([batch_size_val, self.max_n_days, self.y_size], dtype=np.float32)
            main_mv_percent_batch = np.zeros([batch_size_val, ], dtype=np.float32)
            mv_percent_batch = np.zeros([batch_size_val, self.max_n_days], dtype=np.float32)
            price_batch = np.zeros([batch_size_val, self.max_n_days, 3], dtype=np.float32)
            word_batch = np.zeros([batch_size_val, self.max_n_days, self.max_n_msgs, self.max_n_words], dtype=np.int32)
            ss_index_batch = np.zeros([batch_size_val, self.max_n_days, self.max_n_msgs], dtype=np.int32)
            n_msgs_batch = np.zeros([batch_size_val, self.max_n_days], dtype=np.int32)
            n_words_batch = np.zeros([batch_size_val, self.max_n_days, self.max_n_msgs], dtype=np.int32)

            sample_counter = 0 # Renamed sample_id
            while sample_counter < batch_size_val:
                if not generators: # Check if generators list is empty
                    raise StopIteration("No more data from any stock generator.")

                gen_id = random.randint(0, len(generators)-1)
                try:
                    sample_dict = next(generators[gen_id])
                    T_val = sample_dict['T'] # Renamed T
                    stock_batch[sample_counter] = sample_dict['stock']
                    T_batch[sample_counter] = T_val
                    y_batch[sample_counter, :T_val] = sample_dict['ys']
                    main_mv_percent_batch[sample_counter] = sample_dict['main_mv_percent']
                    mv_percent_batch[sample_counter, :T_val] = sample_dict['mv_percents']
                    price_batch[sample_counter, :T_val] = sample_dict['prices']
                    word_batch[sample_counter, :T_val] = sample_dict['msgs']
                    ss_index_batch[sample_counter, :T_val] = sample_dict['ss_indices']
                    n_msgs_batch[sample_counter, :T_val] = sample_dict['n_msgs']
                    n_words_batch[sample_counter, :T_val] = sample_dict['n_words']

                    sample_counter += 1
                except StopIteration:
                    del generators[gen_id]
                    # No need to check 'if generators:' here, the check at the start of the inner loop handles it.
                except Exception as e:
                    logger.error(f"Error processing sample from generator {gen_id}: {e}")
                    # Optionally remove problematic generator or try next
                    if generators: # If there are still generators, remove the problematic one
                        del generators[gen_id]
                    else: # No generators left
                        raise StopIteration("Error in last active generator.")


            batch_dict = {
                'batch_size': sample_counter, # Use actual number of samples collected
                'stock_batch': stock_batch[:sample_counter], # Slice arrays to actual size
                'T_batch': T_batch[:sample_counter],
                'y_batch': y_batch[:sample_counter],
                'main_mv_percent_batch': main_mv_percent_batch[:sample_counter],
                'mv_percent_batch': mv_percent_batch[:sample_counter],
                'price_batch': price_batch[:sample_counter],
                'word_batch': word_batch[:sample_counter],
                'ss_index_batch': ss_index_batch[:sample_counter],
                'n_msgs_batch': n_msgs_batch[:sample_counter],
                'n_words_batch': n_words_batch[:sample_counter],
            }

            yield batch_dict

    def batch_gen_by_stocks(self, phase):
        # This batch size is very large, intended for per-stock processing.
        # Consider if self._get_batch_size(phase) or a different logic is more appropriate
        # if this method is also used where dynamic batch sizing is desired.
        # For now, keeping original large batch_size for this specific method.
        batch_size_fixed = 2000 
        # vocab_id_dict and stock_id_dict are no longer needed here.

        for s_symbol in stock_symbols: # Renamed s to s_symbol
            gen = self.sample_gen_from_one_stock(self.word2id, self.stock2id, s_symbol, phase)

            # Initialize arrays with the fixed large batch size
            stock_batch = np.zeros([batch_size_fixed, ], dtype=np.int32)
            T_batch = np.zeros([batch_size_fixed, ], dtype=np.int32)
            day_batch = list() # Stays as list, converted to array later if needed, or sliced
            Ts_list = list() # Renamed Ts to Ts_list

            n_msgs_batch = np.zeros([batch_size_fixed, self.max_n_days], dtype=np.int32)
            n_words_batch = np.zeros([batch_size_fixed, self.max_n_days, self.max_n_msgs], dtype=np.int32)
            y_batch = np.zeros([batch_size_fixed, self.max_n_days, self.y_size], dtype=np.float32)
            price_batch = np.zeros([batch_size_fixed, self.max_n_days, 3], dtype=np.float32)
            mv_percent_batch = np.zeros([batch_size_fixed, self.max_n_days], dtype=np.float32)
            word_batch = np.zeros([batch_size_fixed, self.max_n_days, self.max_n_msgs, self.max_n_words], dtype=np.int32)
            ss_index_batch = np.zeros([batch_size_fixed, self.max_n_days, self.max_n_msgs], dtype=np.int32)
            main_mv_percent_batch = np.zeros([batch_size_fixed, ], dtype=np.float32)

            sample_counter = 0 # Renamed sample_id
            while sample_counter < batch_size_fixed: # Ensure we don't exceed allocated array size
                try:
                    sample_info_dict = next(gen)
                    T_val = sample_info_dict['T'] # Renamed T
                    stock_batch[sample_counter] = sample_info_dict['stock']
                    T_batch[sample_counter] = sample_info_dict['T']
                    Ts_list.append(sample_info_dict['ts'])
                    day_batch.append(sample_info_dict['main_target_date'])
                    y_batch[sample_counter, :T_val] = sample_info_dict['ys']
                    main_mv_percent_batch[sample_counter] = sample_info_dict['main_mv_percent']
                    mv_percent_batch[sample_counter, :T_val] = sample_info_dict['mv_percents']
                    price_batch[sample_counter, :T_val] = sample_info_dict['prices']
                    word_batch[sample_counter, :T_val] = sample_info_dict['msgs']
                    ss_index_batch[sample_counter, :T_val] = sample_info_dict['ss_indices']
                    n_msgs_batch[sample_counter, :T_val] = sample_info_dict['n_msgs']
                    n_words_batch[sample_counter, :T_val] = sample_info_dict['n_words']

                    sample_counter += 1
                except StopIteration:
                    break # No more samples for this stock
                except Exception as e: # Catch other potential errors from sample_gen
                    logger.error(f"Error processing sample for stock {s_symbol} in batch_gen_by_stocks: {e}")
                    break # Stop processing this stock on error


            n_sample_threshold = 1
            if sample_counter < n_sample_threshold:
                continue # Skip this stock if not enough samples collected

            # Slice all batch arrays to the actual number of samples collected (sample_counter)
            batch_dict = {
                's': s_symbol,
                'batch_size': sample_counter,
                'stock_batch': stock_batch[:sample_counter],
                'T_batch': T_batch[:sample_counter],
                'Ts': Ts_list, # Ts_list is already of correct length
                'day_tar' : day_batch, # day_batch is already of correct length
                'y_batch': y_batch[:sample_counter],
                'main_mv_percent_batch': main_mv_percent_batch[:sample_counter],
                'mv_percent_batch': mv_percent_batch[:sample_counter],
                'price_batch': price_batch[:sample_counter],
                'word_batch': word_batch[:sample_counter],
                'ss_index_batch': ss_index_batch[:sample_counter],
                'n_msgs_batch': n_msgs_batch[:sample_counter],
                'n_words_batch': n_words_batch[:sample_counter],
            }

            yield batch_dict

    def sample_mv_percents(self, phase):
        main_mv_percents = []
        for s_symbol in stock_symbols: # Renamed s
            start_date, end_date = self._get_start_end_date(phase)
            stock_mv_path = os.path.join(str(self.movement_path), '{}.txt'.format(s_symbol))
            main_target_dates = []

            with open(stock_mv_path, 'r') as movement_f:
                for line in movement_f:
                    data = line.split('\t')
                    main_target_date = datetime.strptime(data[0], '%Y-%m-%d').date()
                    main_target_date_str = main_target_date.isoformat()

                    if start_date <= main_target_date_str < end_date:
                        main_target_dates.append(main_target_date)

            for main_target_date_val in main_target_dates: # Renamed main_target_date
                prices_and_ts = self._get_prices_and_ts(s_symbol, main_target_date_val)
                if not prices_and_ts:
                    continue
                main_mv_percents.append(prices_and_ts['main_mv_percent'])

            logger.info('finished: {}'.format(s_symbol))

        return main_mv_percents

    def init_word_table(self):
        word_table_init_val = np.random.random((vocab_size, self.word_embed_size)) * 2 - 1 # Renamed word_table_init
        # [-1.0, 1.0]

        if self.word_embed_type != 'rand': # Check for inequality
            n_replacement = 0
            # vocab_id_dict is now self.word2id
            
            with io.open(self.glove_path, 'r', encoding='utf-8') as f:
                for line in f:
                    tuples = line.split()
                    word, embed = tuples[0], [float(embed_col) for embed_col in tuples[1:]]
                    if word in ['<unk>', 'unk']:
                        word = 'UNK'
                    if word in self.word2id: # Use self.word2id
                        n_replacement += 1
                        word_id_val = self.word2id[word] # Use self.word2id, Renamed word_id
                        word_table_init_val[word_id_val] = embed

            logger.info('ASSEMBLE: word table #replacement: {}'.format(n_replacement))
        return word_table_init_val 