from fuzzywuzzy import process

def enhanced_robust_map(saved_weights, model, threshold=80, order_window=2):
	"""
	Enhanced robustly map weights using tensor shape, approximate order, and fuzzy matching.
	
	Parameters:
	- saved_weights: state_dict of the saved weights.
	- model: an instance of the PyTorch model with the desired architecture.
	- threshold: Score threshold for fuzzy matching.
	- order_window: Allowable deviation in order-based matching.
	
	Returns:
	- A new state_dict with names from the model but values from saved_weights.
	"""
	
	model_state = model.state_dict()
	model_keys = list(model_state.keys())
	saved_keys = list(saved_weights.keys())
	
	converted_weights = {}
	unmatched_saved_keys = []
	
	for saved_idx, saved_key in enumerate(saved_keys):
		matched = False
		
		# Step 1: Shape-based and approximate order matching
		for model_idx, model_key in enumerate(model_keys):
			if abs(saved_idx - model_idx) <= order_window and saved_weights[saved_key].shape == model_state[model_key].shape:
				converted_weights[model_key] = saved_weights[saved_key]
				matched = True
				break
		
		if not matched:
			unmatched_saved_keys.append(saved_key)
	
	# Step 2: Fuzzy string matching for unmatched keys
	for saved_key in unmatched_saved_keys:
		possible_matches = [k for k in model_keys if saved_weights[saved_key].shape == model_state[k].shape]
		closest_match, score = process.extractOne(saved_key, possible_matches)
		
		if score > threshold and closest_match not in converted_weights:
			converted_weights[closest_match] = saved_weights[saved_key]
		else:
			raise ValueError(f"Failed to find a robust match for saved key {saved_key}.")
	
	return converted_weights
