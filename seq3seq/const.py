from ivbase.utils.constants import SMILES_ALPHABET as VOCAB
try:
	with open("expts/stat.txt") as IN:
		ATOM_ALPHABET = list(sorted(IN.readline().split("=")[-1].strip().split(",")))
		VOCAB = list('#%)(+*-/.1032547698:=@[]\\cons') + ATOM_ALPHABET + ['se']
except:
	pass

PADVAL = 0
# adding bos and eos
BOS = ">"
EOS = "<"
VOCAB = [BOS] + VOCAB + [EOS]