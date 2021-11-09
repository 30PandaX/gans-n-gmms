import numpy as np
import pandas as pd
import os


def get_fmri_ids(mode="train"):

    fmri_ids = []
    sliced_len_before = len('/share/sablab/nfs04/data/fmri_on_celeba/derivatives/1st_level_analysis/fwhm5/sub-04/train_run97_')
    sliced_len_after = len('_gt_fix.nii.gz')

    num_sub = 3 if mode == "train" else 4
    print('num_sub = ', num_sub)
    for i in range(num_sub): # sub01 to sub03. exclude sub04
        i += 1
        directory = f'/share/sablab/nfs04/data/fmri_on_celeba/derivatives/1st_level_analysis/fwhm5/sub-0{i}/'
        for entry in os.scandir(directory):
            # TODO ignore anything that starts with test_...
            if entry.path.endswith(".nii.gz") and entry.path.startswith(directory+mode) and entry.is_file():
                # fmri_id  = str(entry.name).split('_')
                fmri_id = str(entry.name)[:-len("_gt_fix.nii.gz")]
                fmri_ids.append(fmri_id)
            # print(f'progress: {i}, {len(fmri_ids)}, id: {fmri_id}')
    
    with open(f'fmri_ids_{mode}.npy', 'wb') as f:
        np.save(f, fmri_ids)

    with open(f'fmri_ids_{mode}.npy', 'rb') as f:
        fmri_ids = np.load(f)
    print(f'Got fmri ids, length is {len(fmri_ids)}')
    return fmri_ids

def get_face_ids():
    img_paths_cleaned = [] #/home/px48/gans-n-gmms/utils/img_paths.npy
    with open('/home/px48/gans-n-gmms/utils/img_paths.npy', 'rb') as f:
        img_paths_cleaned = np.load(f)
    
    print(f'Got img_paths_cleaned, length is {len(img_paths_cleaned)}')
    return img_paths_cleaned

def fmri_celeba_mapping(mode="train"):
    face_ids = get_face_ids()
    fmri_ids = get_fmri_ids(mode)
    fmri_face_map = [] # key is face image

    for face_id in face_ids:
        face_id = face_id.split('/')
        face_id = face_id[2][:-4]
        for index, fmri_id in enumerate(fmri_ids):
            if face_id in fmri_id: 
                fmri_face_map.append(index)

    with open(f'fmri_face_map_{mode}.npy', 'wb') as f:
        np.save(f, fmri_face_map)

    with open(f'fmri_face_map_{mode}.npy', 'rb') as f:
        fmri_face_map = np.load(f)
    print(f'Got fmri_face_map, length is {len(fmri_face_map)}')
    return fmri_face_map
    # find the index of face_id in fmri_ids, return the index

def concatenated_vectors(vectorized_img, vectorized_fmri, fmri_face_map, img_size=32):
    ret = []
    for index, face in enumerate(vectorized_img):
        fmri_index = fmri_face_map[index]
        concatenated_vector = np.concatenate((face, vectorized_fmri[fmri_index]))
        ret.append(concatenated_vector)

    with open(f'concatenated_vectors_{img_size}.npy', 'wb') as f:
        np.save(f, ret)

    with open(f'concatenated_vectors_{img_size}.npy', 'rb') as f:
        ret = np.load(f)
    ret.shape
    return ret

mode = "train"
if os.path.isfile(f'fmri_face_map_{mode}.npy'):
    with open(f'fmri_face_map_{mode}.npy', 'rb') as f:
        fmri_face_map = np.load(f)
else:
    fmri_face_map = fmri_celeba_mapping(mode)

with open('vectorized_fmri.npy', 'rb') as f:
    vectorized_fmri = np.load(f)

img_size = 32
list_file = f'../../braindecoder/vectorized_images_{img_size}.npy'
with open(list_file, 'rb') as f:
    vectorized_img = np.load(f)

# flatten RGB 3D to 1D
vectorized_img = np.asarray(vectorized_img).flatten().reshape(len(vectorized_img), img_size * img_size * 3)
# norm = np.linalg.norm(vectorized_img)
# normal_vectorized_img = vectorized_img/norm
normal_vectorized_img = vectorized_img/255.0
concatenated_vectors(normal_vectorized_img, vectorized_fmri, fmri_face_map, img_size)