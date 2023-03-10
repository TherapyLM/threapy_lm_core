
def threapylm(mtf_features, other_features, hparams, mesh, variable_dtype, context=None):

	x, batch_dim, sequence_dim, embd_dim, vocab_dim, embed_sequence_dim = parse_inputs(mtf_features, other_features)

	if is_incremental_inference(context):
		x = mtf.gather(x, context.position - 1, sequence_dim)
		x = mtf.reshape(x, [batch_dim])

	use_axial_pos_emb = exists(hparams.axial_pso_emb)
	use_rotary_emb = exists(hparams.rotary_emb)

	# Text encoding
	wte = mtf.get_variable(
		mesh,
		"wte",
		mtf.Shape([vocab_dim, embd_dim]),
		initializer=tf.random_normal_initializer(stddev=0.02),
		master_dtype=variable_dtype.master_dtype,
		slice_dtype=variable_dtype.slice_dtype,
		activation_dtype=variable_dtype.activation_dtype,
	)

	with tf.variable_scope("token_embd"):
		# Text embedding
		h = mtf.gather(wte, x, vocab_dim)
		if hparams.embed_dropout > 0 and hparams.mode == "train":
			h = mtf.dropout(h, rate=hparams.embed_dropout, name="wte_dropout")

	# Position embedding
	if use_rotary_emb:
		wpe = None
		layer_pos_emb = rotary_positional_emb()

def block(hparams, scope, layer_num, bias, sequence_dim, memory_length_dim, pos_emb, variable_dtype, context=None):
	use_mlp_glu = hparams.mlp_glu == True
	use_scale_norm = hparams.scalenorm == True
	use_moe = exists(hparams.moe_layers) and (layer_num in hparams.moe_layers)
	use_rezero = hparams.rezero == True
	macaron_attn = hparams.macaron == True

	def fn(x):
		with tf.variable_scope(scope):
			nx = x.shape[-1]
			if use_rezero:
				prenorm = identity
			elif use_scale_norm:
				prenorm = scalenorm
			else:
				prenorm = layer_norm

			pre_residual_fn = rezero if use_rezero else identity
			attn_type = hparams.attention_types.layer_num

			if macaron_attn:
				mult = 0.5
				mlp_fn = mlp_glu if use_mlp_glu else mlp
				intermediate_size = nx.size * 4 * (1 if not use_mlp_glu else 2)
				# Define intermediate layer of mlp - to split
