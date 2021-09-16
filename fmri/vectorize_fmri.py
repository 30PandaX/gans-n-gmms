from nilearn import plotting, datasets
import nibabel as nib
import numpy as np
import os

vectorized_fmri = []
atlas_img = nib.load("/share/sablab/nfs04/data/fmri_on_celeba/atlas/Yeo2011_7Networks_MNI152_FreeSurferConformed1mm_LiberalMask.3mm.nii.gz")
mask = atlas_img.get_fdata()

for i in range(4): # sub01 to sub04
  i += 1
  directory = f'/share/sablab/nfs04/data/fmri_on_celeba/derivatives/1st_level_analysis/fwhm5/sub-0{i}/'
  for entry in os.scandir(directory):
    if (entry.path.endswith(".nii.gz")) and entry.is_file():
        img = nib.load(entry.path)
        # this is the 3D matrix containing the brain image's data
        vol = img.get_fdata()
        visual_data = vol[mask == 1] # pick out visual network
        vectorized_fmri.append(visual_data)

with open('vectorized_fmri.npy', 'wb') as f:
    np.save(f, vectorized_fmri)

with open('vectorized_fmri.npy', 'rb') as f:
    vectorized_fmri = np.load(f)

print(vectorized_fmri)