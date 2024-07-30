import numpy as np
from PIL import Image

mri = np.array(Image.open('MRI_of_Human_Brain.jpg').resize((512, 512)))
cubist = np.array(Image.open('zoya').resize((512, 512)))
new_image = np.zeros_like(mri)
new_image[:, :256, :] = cubist[:, :256, :]
new_image[:, 256:, :] = mri[:, 256:, :]
Image.fromarray(new_image.astype(np.uint8)).save('unreal_image.png')
Image.fromarray((0.5 * mri + 0.5 * cubist).astype(np.uint8)).save('unreal_image_blend.png')
