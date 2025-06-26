import os
import re
import torch
import numpy as np
import pandas as pd
from PIL import Image as PILImage
from datasets import Dataset, Features, Image, ClassLabel, load_dataset, DownloadConfig

DEFAULT_CLASS_NAMES = {
	"cifar10": [
		"airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"
	],
	"cifar100": [
		'apple', 'aquarium fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak tree', 'orange', 'orchid', 'otter', 'palm tree', 'pear', 'pickup truck', 'pine tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow tree', 'wolf', 'woman', 'worm'
	],
	"tinyimagenet": [
		'n01443537', 'n01629819', 'n01641577', 'n01644900', 'n01698640', 'n01742172', 'n01768244', 'n01770393', 'n01774384', 'n01774750', 'n01784675', 'n01882714', 'n01910747', 'n01917289', 'n01944390', 'n01950731', 'n01983481', 'n01984695', 'n02002724', 'n02056570', 'n02058221', 'n02074367', 'n02094433', 'n02099601', 'n02099712', 'n02106662', 'n02113799', 'n02123045', 'n02123394', 'n02124075', 'n02125311', 'n02129165', 'n02132136', 'n02165456', 'n02226429', 'n02231487', 'n02233338', 'n02236044', 'n02268443', 'n02279972', 'n02281406', 'n02321529', 'n02364673', 'n02395406', 'n02403003', 'n02410509', 'n02415577', 'n02423022', 'n02437312', 'n02480495', 'n02481823', 'n02486410', 'n02504458', 'n02509815', 'n02666347', 'n02669723', 'n02699494', 'n02769748', 'n02788148', 'n02791270', 'n02793495', 'n02795169', 'n02802426', 'n02808440', 'n02814533', 'n02814860', 'n02815834', 'n02823428', 'n02837789', 'n02841315', 'n02843684', 'n02883205', 'n02892201', 'n02909870', 'n02917067', 'n02927161', 'n02948072', 'n02950826', 'n02963159', 'n02977058', 'n02988304', 'n03014705', 'n03026506', 'n03042490', 'n03085013', 'n03089624', 'n03100240', 'n03126707', 'n03160309', 'n03179701', 'n03201208', 'n03255030', 'n03355925', 'n03373237', 'n03388043', 'n03393912', 'n03400231', 'n03404251', 'n03424325', 'n03444034', 'n03447447', 'n03544143', 'n03584254', 'n03599486', 'n03617480', 'n03637318', 'n03649909', 'n03662601', 'n03670208', 'n03706229', 'n03733131', 'n03763968', 'n03770439', 'n03796401', 'n03814639', 'n03837869', 'n03838899', 'n03854065', 'n03891332', 'n03902125', 'n03930313', 'n03937543', 'n03970156', 'n03977966', 'n03980874', 'n03983396', 'n03992509', 'n04008634', 'n04023962', 'n04070727', 'n04074963', 'n04099969', 'n04118538', 'n04133789', 'n04146614', 'n04149813', 'n04179913', 'n04251144', 'n04254777', 'n04259630', 'n04265275', 'n04275548', 'n04285008', 'n04311004', 'n04328186', 'n04356056', 'n04366367', 'n04371430', 'n04376876', 'n04398044', 'n04399382', 'n04417672', 'n04456115', 'n04465666', 'n04486054', 'n04487081', 'n04501370', 'n04507155', 'n04532106', 'n04532670', 'n04540053', 'n04560804', 'n04562935', 'n04596742', 'n04598010', 'n06596364', 'n07056680', 'n07583066', 'n07614500', 'n07615774', 'n07646821', 'n07647870', 'n07657664', 'n07695742', 'n07711569', 'n07715103', 'n07720875', 'n07749582', 'n07753592', 'n07768694', 'n07871810', 'n07873807', 'n07875152', 'n07920052', 'n07975909', 'n08496334', 'n08620881', 'n08742578', 'n09193705', 'n09246464', 'n09256479', 'n09332890', 'n09428293', 'n12267677', 'n12520864', 'n13001041', 'n13652335', 'n13652994', 'n13719102', 'n14991210'
	]
}

def get_class_names(source: str = "tinyimagenet") -> list:
	try:
		ds = load_dataset(source, split='train', download_config=DownloadConfig(max_retries=0))
		names = ds.features['label'].names
		if isinstance(names, list) and len(names) > 1:
			return names
	except Exception:
		return DEFAULT_CLASS_NAMES.get(source.lower(), [str(i) for i in range(100)])

def preprocess_image_array(images):
	if isinstance(images, torch.Tensor):
		images = images.detach().cpu().float().numpy()
	elif not isinstance(images, np.ndarray):
		raise ValueError("Expected torch.Tensor or np.ndarray")
	
	if images.ndim == 2:
		images = images[None, None, ...]
	elif images.ndim == 3:
		images = images[None, ...]
	if images.ndim != 4:
		raise ValueError("Expected 4D image array")
	
	if images.shape[1] in [1, 3] and images.shape[1] != images.shape[2]:
		images = np.transpose(images, (0, 2, 3, 1))
	
	if images.max() <= 1.0:
		images = (images * 255).clip(0, 255)
	
	return [PILImage.fromarray(img.astype(np.uint8)) for img in images]

def load_images(X, image_sort_mode):
	if isinstance(X, str) and os.path.isdir(X):
		image_paths = [
			os.path.join(X, f) for f in os.listdir(X)
			if f.lower().endswith(('.png', '.jpg', '.jpeg'))
		]
	
		if image_sort_mode == "natural":
			def natural_key(f): return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', f)]
			image_paths.sort(key=lambda x: natural_key(os.path.basename(x)))
		elif image_sort_mode == "plain":
			image_paths.sort()
		elif image_sort_mode == "mtime":
			image_paths.sort(key=lambda x: os.path.getmtime(x))
		elif image_sort_mode == "none":
			pass
		elif callable(image_sort_mode):
			image_paths.sort(key=image_sort_mode)
		else:
			raise ValueError(f"Invalid image_sort_mode: {image_sort_mode}")
	
		return [PILImage.open(p).convert('RGB') for p in image_paths]
	
	elif isinstance(X, (torch.Tensor, np.ndarray)):
		return preprocess_image_array(X)
	
	elif isinstance(X, list):
		return [PILImage.fromarray(x) if isinstance(x, np.ndarray) else x for x in X]
	
	else:
		raise ValueError("Unsupported image input type.")

def load_labels(label_source):
	if isinstance(label_source, list):
		return label_source
	elif isinstance(label_source, str) and label_source.endswith(".parquet"):
		return pd.read_parquet(label_source)['label'].tolist()
	else:
		raise ValueError("label_source must be a list or .parquet path")

def push_images(	
	images,
	repo: str,
	token: str,
	config_name: str,
	labels,
	class_names='',
	private=False,
	image_sort_mode="natural"
):
	images = load_images(images, image_sort_mode=image_sort_mode)
	labels = load_labels(labels)
	
	if len(images) != len(labels):
		raise ValueError(f"Mismatch: {len(images)} images vs {len(labels)} labels")
	
	class_names = get_class_names(source=class_names)
	
	features = Features({
		"image": Image(),
		"label": ClassLabel(names=class_names)
	})
	
	dataset = Dataset.from_dict({
		"image": images,
		"label": labels
	}, features=features)
	
	dataset.push_to_hub(repo, config_name=config_name, private=private, token=token)
