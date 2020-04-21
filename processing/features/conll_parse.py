import pandas as pd
import stanfordnlp
import pickle

# stanfordnlp.download('en')

def conll_parse(sentences, file_path):

	config = {
			'processors': 'tokenize,lemma,pos,depparse',
			'tokenize_pretokenized': True
			 }

	stanford_parser = stanfordnlp.Pipeline(**config)
	n_steps = len(sentences)

	import time
	start_time = time.time()

	with open(str(file_path), 'w') as f:
		for i, sentence in enumerate(list(sentences)):
			doc_obj = stanford_parser(sentence)

			f.write(doc_obj.conll_file.conll_as_string())

			if i%100==0:
				completion = (i + 1) / n_steps
				comp_time = time.time() - start_time
				print(f'{completion*100:.1f}% complete. Elapsed time: {comp_time:.2f}s')

	print(f'Finished. Total computation time: {time.time() - start_time}')
