import pythainlp

from typing import Text, List, Any, Type, Dict

from rasa.nlu.tokenizers.tokenizer import Token, Tokenizer
from rasa.nlu.training_data import Message


class NewmmPyThaiNLP(Tokenizer):
    defaults = {
        # Flag to check whether to split intents
        "intent_tokenization_flag": False,
        # Symbol on which intent should be split
        "intent_split_symbol": "_",
        # Regular expression to detect tokens
        "token_pattern": None,
    }

    def tokenize(self, message: Message, attribute: Text) -> List[Token]:
        text = message.get(attribute)
        words = pythainlp.tokenize.word_tokenize(
            text, engine='newmm', keep_whitespace=False)

        if not words:
            words = [text]

        return self._convert_words_to_tokens(words, text)