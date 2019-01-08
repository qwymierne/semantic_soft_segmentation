from scipy.misc import imread
from semantic_soft_segmentation import semantic_soft_segmentation

if __name__ == '__main__':
    img = imread('COCO_train2014_000000362884.jpg', mode='RGB')

    features = img[:, img.shape[1] // 2 + 1:, :]
    image = img[:, :img.shape[1] // 2, :]

    sss = semantic_soft_segmentation(image, features)
