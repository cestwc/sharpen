import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import numpy as np
import torch

def view(images, max_images=5, normalise = False):
	
	if isinstance(images, torch.Tensor):
		images = images.cpu().numpy()
	else:
		assert (isinstance(images, np.ndarray))
	
	if images.ndim == 2:
		images[np.newaxis, np.newaxis, :]
	if images.ndim == 3:
		images = images[np.newaxis, :]
	else:
		assert (images.ndim == 4)
	
	
	images = images[:max_images]
	
	dim_to_move_last = np.where(np.array(images.shape[1:]) < 4)[0][0] + 1
	images = np.transpose(images, (0, *range(1, dim_to_move_last), *range(dim_to_move_last+1, 4), dim_to_move_last))
	
	
	for idx, img in enumerate(images):
	
		norm = mcolors.Normalize(vmin=img.min(), vmax=img.max()) if normalise else None
	
		plt.subplot(1, len(images), idx + 1)
	
		if img.shape[-1] == 1:
			plt.imshow(img[:, :, 0], cmap='gray', norm=norm)
		else:
			plt.imshow(img, norm=norm)
		
		plt.axis('off')
	
	plt.show()
	return images
