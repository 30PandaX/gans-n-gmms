import numpy as np
import pandas as pd
import cv2
import os
import scipy.io as sio
import struct
import pickle
from sklearn.model_selection import train_test_split

class ImageBatchProvider:
    def __init__(self, image_folder=None, output_size=None, flatten=False, float_output=True, mirror=True,
                 batch_size=100, test_set_ratio=0.25, max_test_set_size=2000,
                 list_file=None, dtype=np.float32, crop_bbox=None, class_list=None,
                 read_as_gray=False, file_suffix=None):
        self.image_folder = image_folder
        self.output_size = output_size
        self.crop_bbox = crop_bbox
        self.flatten = flatten
        self.batch_size = batch_size
        self.dtype = dtype
        self.train_images = None
        self.test_images = None
        self.read_as_gray = read_as_gray

        if list_file:
            # Split to train/test according to a list file
            # Reading the np array from file:
            with open(list_file, 'rb') as f:
                X = np.load(f)
            # flatten RGB 3D to 1D
            X = np.asarray(X).flatten().reshape(len(X), 32 * 32 * 3)
            # divide test and train set
            num_train = int(len(X) * 0.7)
            num_test = len(X) - num_train
            self.train_image_list = X[:num_train]
            self.test_image_list = X[num_train:]
        else:
            img_paths_cleaned = []
            non_exist = []
            if not os.path.isfile('img_paths.npy'):
                root = "/share/sablab/nfs04/data/fmri_on_celeba/stimuli"
                txt_path = root + "/ImageNames2Celeba.txt"
                img_paths = pd.read_csv(txt_path, delimiter = '\t', names = ['path', 'img_ID']).drop_duplicates(subset=['img_ID'])[['path']]
                img_paths_list = img_paths.values.flatten()
                for path in img_paths_list:
                    img_path = root + '/' + path
                    if os.path.isfile(img_path):
                        img_paths_cleaned.append(path)
                img_paths_cleaned = np.asarray(img_paths_cleaned)
                with open('img_paths.npy', 'wb') as f:
                    np.save(f, img_paths_cleaned)
            else:
                # Reading the np array from file:
                with open('img_paths.npy', 'rb') as f:
                    img_paths_cleaned = np.load(f)
            self.train_image_list, self.test_image_list = train_test_split(img_paths_cleaned, test_size=0.2, random_state=0)


        # assert len(set(self.test_image_list).intersection(self.train_image_list)) == 0
        self.num_test_images = len(self.test_image_list)
        self.num_train_images = len(self.train_image_list)
        self.mirror = mirror
        self.float_output = float_output
        print('Starting image batch provider for {} - {}/{} (train/test) images.'.format(self.image_folder,
                                                                                         self.num_train_images,
                                                                                         self.num_test_images))
        # assert self.num_train_images >= self.batch_size
        self._shuffle()

    def _load_images_from_mat(self, mat_file, train_test_str):
        mat_data = sio.loadmat(mat_file)
        # Below is specific to SVHN
        # TODO: Implement as base-class and dataset-specific adaptation
        images = mat_data['X']
        images = np.transpose(images, [3, 0, 1, 2])
        image_list = [train_test_str + '_' + str(i) for i in list(np.random.permutation(images.shape[0]))]
        return images, image_list

    def _load_images_from_ubyte(self, ubyte_file, train_test_str):
        # Used for MNIST...
        with open(ubyte_file, 'rb') as fimg:
            magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
            images = np.fromfile(fimg, dtype=np.uint8).reshape(num, rows, cols)
        image_list = [train_test_str + '_' + str(i) for i in list(np.random.permutation(images.shape[0]))]
        return images, image_list

    def _load_cifar_images(self, data_folder, file_prefix, class_list, train_test_str):
        batch_files = [f for f in os.listdir(data_folder) if file_prefix in f]
        images = np.empty((0, 32*32*3), dtype=np.uint8)
        for f in batch_files:
            with open(os.path.join(data_folder, f), 'rb') as fo:
                batch_dict = pickle.load(fo, encoding='bytes')
            batch_labels = np.array(batch_dict[b'labels'])
            batch_images = batch_dict[b'data']
            if class_list:
                for cls in class_list:
                    images = np.vstack((images, batch_images[batch_labels == cls]))
            else:
                images = np.vstack((images, batch_images))
        images = np.transpose(np.reshape(images, [images.shape[0], 3, 32, 32]), axes=[0, 2, 3, 1])
        image_list = [train_test_str + '_' + str(i) for i in list(np.random.permutation(images.shape[0]))]
        return images, image_list

    def _shuffle(self):
        print('@', end='', flush=True)
        self.random_order = np.random.permutation(self.num_train_images)
        self.mb_idx = 0

    def _crop_image(self, img, crop_rect_or_size):
        if len(crop_rect_or_size) == 2:
            # Random cropping to a fix size
            s = crop_rect_or_size
            x = np.random.randint(0, img.shape[1] - s[0])
            y = np.random.randint(0, img.shape[0] - s[1])
            return img[y:y+s[1], x:x+s[0]]
        # Fixed bbox cropping
        b = crop_rect_or_size
        return img[b[1]:b[1]+b[3], b[0]:b[0]+b[2]]

    def _process_image(self, img, mirror_image=False):
        assert not(mirror_image and self.mirror)
        if self.crop_bbox:
            img = self._crop_image(img, self.crop_bbox)
        if mirror_image or (self.mirror and np.random.uniform() > 0.5):
            img = img[:, ::-1]   # mirror
        if self.output_size and not self.output_size == img.shape[1::-1]:
            img = cv2.resize(img, self.output_size)
        if self.float_output:
            img = img.astype(self.dtype) / 255.0
        if self.flatten:
            img = img.flatten()
        return img


    def _collect_batch_data(self, image_list, image_indices):
        m = image_indices.size
        mb_data = None
        root = "/share/sablab/nfs04/data/fmri_on_celeba/stimuli"
        for i in range(m):
            mirror_image = False
            img_name = image_list[image_indices[i]]
            if img_name.startswith('mirror:'):
                img_name = img_name[len('mirror:'):]
                mirror_image = True
            if img_name.startswith('train_'):
                img = self.train_images[int(img_name[len('train_'):]), ...]
            elif img_name.startswith('test_'):
                img = self.test_images[int(img_name[len('test_'):]), ...]
            else:
                assert self.train_images is None and self.test_images is None
                if self.read_as_gray:
                    img = cv2.imread(root + '/' + img_name, cv2.IMREAD_GRAYSCALE)
                else:
                    img = cv2.imread(root + '/' + img_name)
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = self._process_image(img, mirror_image=mirror_image)
            if mb_data is None:
                mb_data = np.zeros([m] + list(img.shape), dtype=img.dtype)
            mb_data[i, ...] = img
        return mb_data

    def get_images_from_list(self, image_list):
        return self._collect_batch_data(image_list, np.arange(0, len(image_list)))

    def get_test_samples(self, num_samples):
        temp_random_order = np.random.permutation(self.num_test_images)
        return self._collect_batch_data(self.test_image_list, temp_random_order[:num_samples])

    def get_random_samples(self, num_samples):
        temp_random_order = np.random.permutation(self.num_train_images)
        return self._collect_batch_data(self.train_image_list, temp_random_order[:num_samples])

    def get_next_minibatch_samples(self):
        if self.mb_idx + self.batch_size >= self.num_train_images:
            self._shuffle()
        # print(self.mb_idx, end='', flush=True)
        batch_data = self._collect_batch_data(self.train_image_list, self.random_order[self.mb_idx:self.mb_idx + self.batch_size])
        self.mb_idx += self.batch_size
        return batch_data

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    image_provider = ImageBatchProvider('../../../../Datasets/CelebA/img_align_celeba', output_size=(52, 64),
                                        list_file='../../../../Datasets/CelebA/list_eval_partition.txt')
    mb_data = image_provider.get_next_minibatch_samples()
    print(mb_data.shape)
    for i in range(mb_data.shape[0]):
        plt.imshow(mb_data[i, ...])
        plt.pause(0.1)
