# GPT Pretrained Tokenizer
from tokenizers import Tokenizer
from transformers import GTP2Tokenizer, GPT2TokenizerFast


def fetch_encoder(params):
	no_dataset = params.no_dataset or False
	if no_dataset:
		return None

	dataset = next(iter(params.dataset_configs.values()))
	path = dataset.tokenizer_path
	is_pretrained = dataset.tokenizer_is_pretrained or False

	if is_pretrained:
		tok = GPT2TokenizerFast.from_pretrained(path)
		# Will add a padding token id of 50257 at run-time
		tok.add_special_tokens({'pad_token': '<|padding|>'})
		return tok

	return Tokenizer.from_file(path)

# GTP2Tokenizer and Tokenizer have different ways of fetching token ids
def encode(encoder, text):
	result = encoder.encode(text)
	if isinstance(result, list):
		return result
	return result.ids


def train(base_dir, output_dir, file_type, vocab_size):
	archives = glob(str(data_path / f"*.{file_type}"))
	if os.path.exists(output_dir):
		shutil.rmtree(output_dir)

	if not output_dir.is_dir():
		output_dir.mkdir()

		for arch in tqdm(archives):
			name = os.path.basename(arch).split(".")[0] + ".txt"
			fp = output_dir / name

			if file_type == 'xz':
				g = Reader(arch).stream_data()
				with open(fp, "w") as f:
					for s in g:
						f.write(s)
						f.write("\n\n")
			elif file_type == 'txt':
				shutil.copyfile(str(arch), str(fp))

	data_files = glob(str(output_dir / '*.txt'))
	data_files = random.sample(data_files, int(0.2 * len(data_files)))

	assert len(data_files) > 0, 'No data files found'

	tokenizer = Tokenizer(models.BPE())

	# Customize pre-tokenization and decoding
	tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
	tokenizer.decoder = decoders.ByteLevel()
	tokenizer.post_process = processors.ByteLevel(trim_offsets=True)
	tokenizer.normalizer = NFKC()

	# Add then train
	trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=2, special_tokens=["<|endoftext|>", "<|padding|>"])
	tokenizer.train(trainer, data_files)

	# And then save it
	tokenizer_path = output_dir / "byte-level-bpe.tokenizer.json"
	tokenizer.save(str(tokenizer_path), pretty=True)

	print(f"Tokenizer saved at {str(tokenizer_path)}")