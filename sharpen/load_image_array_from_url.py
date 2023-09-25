import urllib.request
import cv2
import numpy as np

def img_from_url(url):
	response = urllib.request.urlopen(url)
	image_data = response.read()
	np_array = np.frombuffer(image_data, np.uint8)
	image_bgr = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
	return image_bgr[:, :, ::-1].copy()
