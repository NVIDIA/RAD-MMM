# SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
""" adapted from https://github.com/keithito/tacotron """

import re
import numpy as np
from collections import defaultdict
from . import cleaners
from .cleaners import Cleaner
from .symbols import get_symbols
from .grapheme_dictionary import Grapheme2PhonemeDictionary

#########
# REGEX #
#########

# Regular expression matching text enclosed in curly braces for encoding
_curly_re = re.compile(r'(.*?)\{(.+?)\}(.*)')

# Regular expression matching words and not words
_words_re = re.compile(r"([a-zA-ZÀ-ž]+['][a-zA-ZÀ-ž]+|[a-zA-ZÀ-ž]+)|([{][^}]+[}]|[^a-zA-ZÀ-ž{}]+)")

_phonemizer_language_map = {
    'hi_HI': 'hi', 
    'hi': 'hi',
    'mar_MAR': 'mr',
    'te_TE': 'te',
    'pt_BR': 'pt-br',
    'en_US': 'en-us',
    'en': 'en-us',
    'de_DE': 'de',
    'fr_FR': 'fr-fr',
    'es_ES': 'es',
    'es_CO': 'es-419',
    'es_AR': 'es-419',
    'es_CL': 'es-419',
    'es_PE': 'es-419',
    'es_PR': 'es-419',
    'es_VE': 'es-419',
    'es_MX': 'es-419',
    'en_ES': 'en-us',
    'en_MN': 'en-us',
    'en_UK': 'en-gb'
}

def lines_to_list(filename):
    with open(filename, encoding='utf-8') as f:
        lines = f.readlines()
    lines = [l.rstrip() for l in lines]
    return lines


class TextProcessing(object):
    def __init__(self, symbol_set, cleaner_name, heteronyms_path,
                 phoneme_dict_path, p_phoneme, handle_phoneme,
                 handle_phoneme_ambiguous, prepend_space_to_text=False,
                 append_space_to_text=False, add_bos_eos_to_text=False,
                 encoding='latin-1',
                 dict_split_token='\t', external_symbol_set_path=None,
                 g2p_type='phonemizer',
                 phonemizer_cfg=None):
        
        self.g2p_type = g2p_type

        if heteronyms_path is not None and heteronyms_path != '':
            self.heteronyms = set(lines_to_list(heteronyms_path))
        else:
            self.heteronyms = []
        
        if g2p_type == 'phonemizer':
            self.phonemedict = None
        else:
            self.phonemedict = Grapheme2PhonemeDictionary(
                phoneme_dict_path, encoding=encoding, split_token=dict_split_token)

        self.cleaner_names = cleaner_name
        self.cleaner = Cleaner(cleaner_name, self.phonemedict)

        self.p_phoneme = p_phoneme
        self.handle_phoneme = handle_phoneme
        self.handle_phoneme_ambiguous = handle_phoneme_ambiguous

        if g2p_type == 'phonemizer':
            self.phonemizer_cfg = phonemizer_cfg
            self.phonemizer_backend_dict = {}
            for language, lang_phoneme_dict_path in phonemizer_cfg.items():
                print("loading: ", lang_phoneme_dict_path)
                self.phonemizer_backend_dict[language] = Grapheme2PhonemeDictionary(
                    lang_phoneme_dict_path, encoding=encoding,
                    split_token=dict_split_token,
                    language=language)
        
        self.symbols, self.markers, self.placeholder_set, self.dipthongs_set \
            = get_symbols(symbol_set, external_symbol_set_path)
        
        self.prepend_space_to_text = prepend_space_to_text
        self.append_space_to_text = append_space_to_text
        self.add_bos_eos_to_text = add_bos_eos_to_text

        if add_bos_eos_to_text:
            self.symbols.append('<bos>')
            self.symbols.append('<eos>')

        # Mappings from symbol to numeric ID and vice versa:
        self.symbol_to_id = {s: i for i, s in enumerate(self.symbols)}
        self.id_to_symbol = {i: s for i, s in enumerate(self.symbols)}

        
    def text_to_sequence(self, text):
        sequence = []

        # Check for curly braces and treat their contents as phoneme:
        while len(text):
            m = _curly_re.match(text)
            if not m:
                sequence += self.symbols_to_sequence(text)
                break
            sequence += self.symbols_to_sequence(m.group(1))
            sequence += self.phoneme_to_sequence(m.group(2))
            text = m.group(3)

        return sequence

    def sequence_to_text(self, sequence):
        result = ''
        for symbol_id in sequence:
            if symbol_id in self.id_to_symbol:
                s = self.id_to_symbol[symbol_id]
                # Enclose phoneme back in curly braces:
                if len(s) > 1 and s[0] == '@':
                    s = '{%s}' % s[1:]
                result += s
        return result.replace('}{', ' ')

    def clean_text(self, text):
        text = self.cleaner(text)
        return text

    def parse_placeholder(self, marker, text, placeholder_type):
        placeholder_set = self.placeholder_set[placeholder_type]
        parsed_token = None

        if placeholder_type == 'right' and len(text) > 1:
            # make sure text at index+1 gets applied the marker
            syllable = text[1]
            parsed_token = marker + syllable
            remaining_text = text[2:]
        elif placeholder_type == 'other':
            # marker is separate
            parsed_token = marker
            remaining_text = text[1:]
        else:
            # to apply marker to text[0]
            syllable = text[0]
            parsed_token = syllable + marker
            remaining_text = text[2:]

        return parsed_token, remaining_text

    def parse_phonemized_text(self, text):
        """
        recursively get the token string and split it based on markers and placeholders
        args: text: input text to be parsed
        returns list of tokens
        """

        if len(text) == 0:
            return []

        parsed_tokens = [] # return can be a list of tokens

        if text[0] in self.placeholder_set['right']:
            # find which marker and apply parsing to the rest of the string
            # marker application with right placeholder
            parsed_token, remaining_text = self.parse_placeholder(text[0], text, 'right')

        elif text[0] in self.placeholder_set['other']:
            # marker application with other placeholder
            parsed_token, remaining_text = self.parse_placeholder(text[0], text, 'other')
        else:
            if len(text) > 1 and text[1] in self.placeholder_set['left']:
                lookahead_character = text[1]
                parsed_token, remaining_text = self.parse_placeholder(lookahead_character, text, 'left')
            elif len(text) > 1:
                parsed_token = text[0]
                remaining_text = text[1:]
                for i in range(len(text)):
                    if text[:i+1] in self.dipthongs_set:
                        parsed_token = text[:i+1]
                        remaining_text = text[i+1:]
            else:
                # no marker match, must be independent syllable, leave as is
                parsed_token = text[0]
                remaining_text = text[1:]

        tokens = [parsed_token] + self.parse_phonemized_text(remaining_text)
        return tokens

    
    def symbols_to_sequence(self, symbols):
        cur_symbols = []
        for s in symbols:
            if s in self.symbol_to_id:
                cur_symbols.append(self.symbol_to_id[s])
            else:
                if self.placeholder_set == None:
                    for sym in symbols:
                        if sym != '@':
                            if '@' + sym in self.symbol_to_id:
                                cur_symbols.append(self.symbol_to_id['@' + sym])
                            
                else:
                    tokens = self.parse_phonemized_text(s)
                    for token in tokens:
                        if token != '@':
                            if '@' + token in self.symbol_to_id:
                                cur_symbols.append(self.symbol_to_id['@' + token])
                            else:
                                # parse character by character
                                for sym in token:
                                    if sym != '@':
                                        if '@' + sym in self.symbol_to_id:
                                            cur_symbols.append(self.symbol_to_id['@' + sym])
                                        
        return cur_symbols
        
    def phoneme_to_sequence(self, text):
        return self.symbols_to_sequence(['@' + s for s in text.split()])

    def get_phoneme(self, word, phoneme_dict=None):
        print(word)
        print(phoneme_dict)
        phoneme_suffix = ''

        if phoneme_dict == None:
            phoneme_dict = self.phonemedict
        else:
            phoneme = phoneme_dict.lookup(word)
            if phoneme is None:
                return word 
            phoneme = "{" + ' '.join(phoneme) + phoneme_suffix + "}"
            return phoneme

        if word.lower() in self.heteronyms:
            return word

        if len(word) > 2 and word.endswith("'s"):
            phoneme = phoneme_dict.lookup(word)
            if phoneme is None:
                phoneme = phoneme_dict.lookup(word[:-2])
                phoneme_suffix = '' if phoneme is None else ' Z'

        elif len(word) > 1 and word.endswith("s"):
            phoneme = phoneme_dict.lookup(word)
            if phoneme is None:
                phoneme = phoneme_dict.lookup(word[:-1])
                phoneme_suffix = '' if phoneme is None else ' Z'
        else:
            phoneme = phoneme_dict.lookup(word)

        if phoneme is None:
            return word

        if len(phoneme) > 1:
            if self.handle_phoneme_ambiguous == 'first':
                phoneme = phoneme[0]
            elif self.handle_phoneme_ambiguous == 'random':
                phoneme = np.random.choice(phoneme)
            elif self.handle_phoneme_ambiguous == 'ignore':
                return word
        else:
            phoneme = phoneme[0]

        phoneme = "{" + phoneme + phoneme_suffix + "}"

        return phoneme

    def encode_text(self, text, return_all=False, language=None, is_phonemized=False):
        if not is_phonemized:
            print(f'{text} is NOT phonemized...')
            print(language)
            text_clean = self.clean_text(text)
            text = text_clean
            text_phoneme = ''
            if self.g2p_type == 'custom':    
                if self.p_phoneme > 0:
                    text_phoneme = self.convert_to_phoneme(text)
                    text = text_phoneme
                text_encoded = self.text_to_sequence(text)
            elif self.g2p_type == 'phonemizer':
                # replace with lookup
                assert language is not None
                text_phoneme = self.convert_to_phoneme(text, phoneme_dict=self.phonemizer_backend_dict[language])
                print(f'{language}|{text_phoneme}')
                text_encoded = self.text_to_sequence(text_phoneme)
                print(f'{language}|{text_encoded}')
                

        else:
            # text is already phonemized
            text_phoneme = text
            text_encoded = self.text_to_sequence(text_phoneme)

        if self.prepend_space_to_text:
            text_encoded.insert(0, self.symbol_to_id[' '])

        if self.append_space_to_text:
            text_encoded.append(self.symbol_to_id[' '])

        if self.add_bos_eos_to_text:
            text_encoded.insert(0, self.symbol_to_id['<bos>'])
            text_encoded.append(self.symbol_to_id['<eos>'])

        if return_all:
            return text_encoded, text_clean, text_phoneme

        return text_encoded

    def convert_to_phoneme(self, text, phoneme_dict=None):
        if self.handle_phoneme == 'sentence':
            if np.random.uniform() < self.p_phoneme:
                words = _words_re.findall(text)
                text_phoneme = [
                    self.get_phoneme(word[0], phoneme_dict=phoneme_dict)
                    if (word[0] != '') else re.sub(r'\s(\d)', r'\1', word[1])
                    for word in words]
                text_phoneme = ''.join(text_phoneme)
                text = text_phoneme
        elif self.handle_phoneme == 'word':
            words = _words_re.findall(text)
            text_phoneme = [
               re.sub(r'\s(\d)', r'\1', word[1]) if word[0] == '' else (
                    self.get_phoneme(word[0], phoneme_dict=phoneme_dict)
                    if np.random.uniform() < self.p_phoneme
                    else word[0])
                for word in words]
            text_phoneme = ''.join(text_phoneme)
            text = text_phoneme
        elif self.handle_phoneme != '':
            raise Exception("{} handle_phoneme is not supported".format(
                self.handle_phoneme))
        return text
