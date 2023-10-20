import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch

def view(images, max_images=5, normalise=False, bounding_boxes=None, axis = False):
	"""
	Display images with optional bounding boxes.
	
	Args:
		images (torch.Tensor or np.ndarray): The input images as a tensor or NumPy array.
		max_images (int): The maximum number of images to display.
		normalise (bool): Whether to normalize the image intensity.
		bounding_boxes (list of list of numpy.ndarray): A list of lists of bounding boxes, where each inner list
			contains bounding boxes for a single image. Each bounding box is a NumPy array with shape (4,)
			representing (x1, y1, x2, y2).
	
	Returns:
		None
	"""
	if isinstance(images, torch.Tensor):
		images = images.cpu().numpy()
	else:
		assert isinstance(images, np.ndarray)
	
	if images.ndim == 2:
		images = images[np.newaxis, np.newaxis, :]
	if images.ndim == 3:
		images = images[np.newaxis, :]
	else:
		assert images.ndim == 4
	
	if bounding_boxes is not None:
		if bounding_boxes.ndim == 2:
			bounding_boxes = bounding_boxes[np.newaxis, :]
		else:
			assert (bounding_boxes.ndim == 3)
		assert (len(bounding_boxes) == len(images))
	
	images = images[:max_images]
	
	dim_to_move_last = np.where(np.array(images.shape[1:]) < 4)[0][0] + 1
	images = np.transpose(images, (0, *range(1, dim_to_move_last), *range(dim_to_move_last + 1, 4), dim_to_move_last))
	
	for idx, img in enumerate(images):
		norm = mcolors.Normalize(vmin=img.min(), vmax=img.max()) if normalise else None
		plt.subplot(1, len(images), idx + 1)
	
		if img.shape[-1] == 1:
			plt.imshow(img[:, :, 0], cmap='gray', norm=norm)
		else:
			plt.imshow(img, norm=norm)
	
		if bounding_boxes is not None:
			boxes = bounding_boxes[idx]
			for box in boxes:
				x1, y1, x2, y2 = box
				rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
				plt.gca().add_patch(rect)
	
		if not axis:
			plt.axis('off')
	
	plt.show()
	return images
