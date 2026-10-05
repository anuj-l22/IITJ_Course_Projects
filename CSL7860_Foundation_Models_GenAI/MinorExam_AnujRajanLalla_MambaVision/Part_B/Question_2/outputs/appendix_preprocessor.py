import re, unicodedata
from unidecode import unidecode

EMOJI_CLASS = re.compile("[" "\U0001F300-\U0001FAFF" "\U00002700-\U000027BF" "\U00002600-\U000026FF" "]+", flags=re.UNICODE)

def light_preprocess(text: str) -> str:
    """
    Light, safe preprocessor for messy user inputs:
      1) Unicode NFKC normalization (fold width/compatibility forms).
      2) Remove emoji/pictographs.
      3) Strip diacritics to ASCII via unidecode (keeps intent).
      4) Collapse repeated whitespace.
    """
    s = unicodedata.normalize("NFKC", text)
    s = EMOJI_CLASS.sub(" ", s)
    s = unidecode(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s
