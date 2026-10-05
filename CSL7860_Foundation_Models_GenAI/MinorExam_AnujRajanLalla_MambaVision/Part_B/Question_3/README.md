# Q3_Needle-in-a-Haystack & Lost-in-the-Middle.ipynb

Notebook that builds a long context, inserts a unique needle code, asks
“Final: <code>”, and evaluates Exact Match (EM).

Source texts (neutral/CC) — used to build the long context:

cricket, hp, non_fiction, orogeny, recipe (plain-text files)

corpus_concat
Concatenation of the five source texts after light whitespace normalization.

Needle variants (each contains exactly one line like
“The secret ticket code is BaKaLuFFY22.”):

long_start — needle inserted near the beginning

long_middle — needle inserted near the midpoint

long_end — needle inserted near the end

results (csv)
Per-run records from the notebook.

requirements.txt ( Obtained from the colab notebook )

# For reproducing 

Just rerun the cells in the notebook .
