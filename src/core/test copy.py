import sys
from radiomics import featureextractor
import nrrd
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import csv

print(sys.version)

def rgb_to_gray(rgb_array):
    # Apply standard luminance weights using a dot product
    return np.dot(rgb_array[..., :3], [0.2989, 0.5870, 0.1140])


image, header = nrrd.read(r"C:\Users\bil\1.nrrd")
mask, mask_header = nrrd.read(r"C:\Users\bil\1.seg.nrrd")

image = np.squeeze(image)
image = np.transpose(image, (1,2,0))
mask = np.squeeze(mask)

print(image.shape)
print(mask.shape)

gray_image = rgb_to_gray(image)
print(gray_image.shape)

gray_image = np.transpose(gray_image, (1,0))
mask = np.transpose(mask, (1,0))


plt.imshow(gray_image)
plt.show()

plt.imshow(mask)
plt.show()

image_sitk = sitk.GetImageFromArray(gray_image)
mask_sitk = sitk.GetImageFromArray(mask)

extractor = featureextractor.RadiomicsFeatureExtractor()
extractor.loadParams('params.yaml')


result = extractor.execute(image_sitk, mask_sitk)


with open('output.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    
    # Write the header row
    writer.writerow(['Key', 'Value'])
    
    # Loop through the dictionary, print, and save
    for key, val in result.items():
        print("\t%s: %s" % (key, val))
        writer.writerow([key, val])
