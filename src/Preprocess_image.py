import cv2
import numpy as np

def imread(img_path, target_size):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        img = np.zeros((target_size[1], target_size[0]), dtype=np.uint8)
    u, i = np.unique(img.flatten(), return_inverse=True)
    background_intensity = int(u[np.argmax(np.bincount(i))])
    return img, background_intensity

def preprocess_image(image_path, target_size, augmentation=False):
    image, bg_intensity = imread(image_path, target_size)
    (t_w, t_h, ch) = target_size
    (h, w) = image.shape
    fx = w/t_w
    fy = h/t_h
    f = max(fx, fy)
    newsize = (max(min(t_w, int(w / f)), 1), max(min(t_h, int(h / f)), 1))
    image = cv2.resize(image, newsize)
    (h, w) = image.shape
    background = np.ones((t_h, t_w), dtype=np.uint8) * bg_intensity
    row_freedom = background.shape[0]-image.shape[0]
    col_freedom = background.shape[1]-image.shape[1]
    row_off=0
    col_off=0
    if augmentation:
        if row_freedom:
            row_off = np.random.randint(0, row_freedom)
        if col_freedom:
            col_off = np.random.randint(0, col_freedom)
    else:
        row_off, col_off = row_freedom//2 , col_freedom//2
   
    background[row_off:row_off+h, col_off:col_off+w] = image
   
    image = cv2.transpose(background)
    return image

def augmentation():
    pass

def normalization():
    pass

def preprocess_label(text, maxTextLength):
    cost = 0
    for i in range(len(text)):
        if i != 0 and text[i] == text[i-1]:
            cost += 2
        else:
            cost += 1

        if cost > maxTextLength:
            return text[:i]

    return text
if __name__ == "__main__":
    img1 = cv2.imread("f07-046a-06-05.png", 0)
    import matplotlib.pyplot as plt

    plt.subplot(121)
    plt.imshow(img1, cmap='gray')
    img2 = preprocess_image("f07-046a-06-05.png", (96, 32, 1), True)
    plt.subplot(122)
    plt.imshow(cv2.transpose(img2), cmap='gray')
    plt.show()