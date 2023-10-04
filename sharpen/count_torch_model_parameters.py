def count_parameters(model):
	trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print(f'The model has {trainable_params:,} trainable parameters')
	
	total_params = sum(p.numel() for p in model.parameters())
	print(f'The model has {total_params:,} total parameters')
	
	return total_params, trainable_params
