# sharpen
Don't be afraid of making copies!

## Installation

To install this small tool from source code
```
pip install git+https://github.com/cestwc/sharpen/
```

## How to use

Example
```python
import sharpen
img = sharpen.img_from_url('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRGsRnODOn8Zx5ivrww4p_AR1sAjC3AXo-hyOev1nNTCEbwx7klxq2_ADltxprbOt56T2o&usqp=CAU')
```

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(10, 50),
    nn.ReLU(),
    nn.Linear(50, 1)
)

total, trainable = sharpen.count_parameters(model)
```

```python
from sharpen import generate_serial
generate_serial()
```

```python
from sharpen import view
view(_your_image, normalise = False, max_images = 1, bounding_boxes= None, axis = True).shape
```
