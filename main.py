import matplotlib.pyplot as plt
# from matplotlib.colors import LightSource
import numpy as np
from skimage import io
# from scipy.ndimage import gaussian_filter
import cv2



def read_images(set, ext):
    print("Reading images...")
    im_up = io.imread('images/' + set + '/up.' + ext)
    im_right = io.imread('images/' + set + '/right.' + ext)
    im_down = io.imread('images/' + set + '/down.' + ext)
    im_left = io.imread('images/' + set + '/left.' + ext)
    # print(im_up.shape)
    im = np.stack((im_up, im_right, im_down, im_left))
    print(im.shape)
    print("...finished reading images!\n")
    return im

def get_norm_intensities(im):
    print("Converting to grayscale...")
    im_gray = np.mean(im, axis=-1) # 4 grayscale images
    # print(im_int.shape)
    means = np.mean(im_gray, axis=(1,2)).reshape(4) # means of 4 images
    # print(means.shape)
    means = (means / np.mean(means)).reshape((4,1,1))  # get scalar to normali_e
    # print(means.shape)
    im_gray = np.multiply(im_gray, means)
    # print(im_gray.shape)
    print("...finished converting to grayscale!\n")
    return im_gray

def get_no_shadow(im, im_gray):
    print("Calculating no shadow images...")
    im_max_rgb = np.amax(im, axis=0)
    im_max_gray = np.amax(im_gray, axis=0)
    # print(im_max_gray.shape)
    print("...calculated no shadow images!\n")
    return im_max_rgb, im_max_gray

def get_ratio_images(im_gray, im_max_gray):
    print("Calculating ratio images...")
    print(im_gray.shape)
    print(im_max_gray.shape)
    im_ratio = im_gray / im_max_gray
    print(im_ratio.shape)
    print("...calculated ratio images!\n")
    return im_ratio

def get_confidence_map(im_ratio):
    print("Computing confidence map...")
    
    sobel_x = np.array([[1, 0, -1],
                        [2, 0, -2],
                        [1, 0, -1]])

    sobel_y = np.array([[ 1,  2,  1],
                        [ 0,  0,  0],
                        [-1, -2, -1]])


    d_up = cv2.filter2D(im_ratio[0], -1, sobel_y)
    d_right = cv2.filter2D(im_ratio[1], -1, sobel_x)
    d_down = cv2.filter2D(im_ratio[2], -1, sobel_y)
    d_left = cv2.filter2D(im_ratio[3], -1, sobel_x)

    silhouette_up  = np.where(d_up > 0, d_up, 0)
    silhouette_right = np.where(d_right < 0, -d_right, 0)
    silhouette_down = np.where(d_down < 0, -d_down, 0)
    silhouette_left  = np.where(d_left > 0, d_left, 0)
    # silhouette_up  = d_up * (d_up > 0)
    # silhouette_right = abs(d_right * (d_right < 0))
    # silhouette_down = abs(d_down * (d_down < 0))
    # silhouette_left  = d_left * (d_left > 0)
    print(silhouette_up.shape)
    print(silhouette_right.shape)
    print(silhouette_down.shape)
    print(silhouette_left.shape)

    im_silhouette = np.stack((silhouette_up, silhouette_right, silhouette_down, silhouette_left))
    print(im_silhouette.shape)


    im_confidence = np.amax(im_silhouette, axis=0)
    print(im_confidence.shape)

    print("...computed confidence map!\n")
    # return d0, d1, d2, d3, im_confidence
    return silhouette_up, silhouette_right, silhouette_down, silhouette_left, im_confidence
    # return d_up, d_right, d_down, d_left, im_confidence




# def get_confidence_map(silhouette_up, silhouette_right, silhouette_down, silhouette_left):
#     low_thresh = 0.5;
#     hi_thresh = 1.0;
#     edges_up = hysteresis_thresholding(silhouette_up, low_thresh, hi_thresh)
#     edges_right = hysteresis_thresholding(silhouette_right, low_thresh, hi_thresh)
#     edges_down = hysteresis_thresholding(silhouette_down, low_thresh, hi_thresh)
#     edges_left = hysteresis_thresholding(silhouette_left, low_thresh, hi_thresh)

#     edges = edges_up | edges_right | edges_down | edges_left
#     return edges_up, edges_right, edges_down, edges_left, edges





def normalize(I):
    return (I - np.min(I)) / (np.max(I) - np.min(I))

def main():

    print("Initializing variables...\n")

    im = read_images("flower", "bmp")
    # fig = plt.figure()
    # plt.imshow(im[0])
    # fig = plt.figure()
    # plt.imshow(im[1])
    # fig = plt.figure()
    # plt.imshow(im[2])
    # fig = plt.figure()
    # plt.imshow(im[3])

    im_gray = get_norm_intensities(im)
    # fig = plt.figure()
    # plt.imshow(im_gray[0], cmap='gray')
    # fig = plt.figure()
    # plt.imshow(im_gray[1], cmap='gray')
    # fig = plt.figure()
    # plt.imshow(im_gray[2], cmap='gray')
    # fig = plt.figure()
    # plt.imshow(im_gray[3], cmap='gray')

    im_max_rgb, im_max_gray = get_no_shadow(im, im_gray)
    fig = plt.figure()
    plt.imshow(im_max_rgb)
    # fig = plt.figure()
    # plt.imshow(im_max_gray, cmap='gray')
    # color = cv2.stylization(im_max_rgb, sigma_s=60, sigma_r=0.07)
    color = cv2.edgePreservingFilter(im_max_rgb, flags=2, sigma_s=60, sigma_r=0.4)
    fig = plt.figure()
    plt.imshow(color)

    im_ratio = get_ratio_images(im_gray, im_max_gray)
    # fig = plt.figure()
    # plt.imshow(im_ratio[0], cmap='gray')
    # fig = plt.figure()
    # plt.imshow(im_ratio[1], cmap='gray')
    # fig = plt.figure()
    # plt.imshow(im_ratio[2], cmap='gray')
    # fig = plt.figure()
    # plt.imshow(im_ratio[3], cmap='gray')

    d0, d1, d2, d3, im_confidence = get_confidence_map(im_ratio)
    # im_confidence = get_confidence_map(im_ratio)
    # fig = plt.figure()
    # plt.imshow(d0, cmap='gray')
    # fig = plt.figure()
    # plt.imshow(d1, cmap='gray')
    # fig = plt.figure()
    # plt.imshow(d2, cmap='gray')
    # fig = plt.figure()
    # plt.imshow(d3, cmap='gray')
    # fig = plt.figure()
    # plt.imshow(normalize(np.where(im_confidence < 0.5, 0, 1)), cmap='gray')
    fig = plt.figure()
    plt.imshow(im_confidence, cmap='gray')
    fig = plt.figure()
    plt.imshow(1 - im_confidence, cmap='gray')
    fig = plt.figure()
    plt.imshow(color / 255 + np.expand_dims(np.clip(im_confidence, 0.3, np.inf), axis=2))


    # conf = np.expand_dims(np.clip(im_confidence, 0.3, np.inf), axis=2)
    conf = 1 - im_confidence

    new = np.zeros_like(color)
    new1 = np.zeros_like(color)
    new2 = np.zeros_like(color)
    new3 = np.zeros_like(color)
    new4 = np.zeros_like(color)
    for y in range(new.shape[0]):
        for x in range(new.shape[1]):
            if(conf[y,x] >= 0.9):
                new[y,x,:] = color[y,x,:]
            if(conf[y,x] >= 0.7):
                new1[y,x,:] = color[y,x,:]
            if(conf[y,x] >= 0.6):
                new2[y,x,:] = color[y,x,:]
            if(conf[y,x] >= 0.5):
                new3[y,x,:] = color[y,x,:]
            if(conf[y,x] >= 0.95):
                new4[y,x,:] = color[y,x,:]

    fig = plt.figure()
    plt.imshow(new / 255)

    fig = plt.figure()
    plt.imshow(new1 / 255)

    fig = plt.figure()
    plt.imshow(new2 / 255)

    fig = plt.figure()
    plt.imshow(new3 / 255)

    fig = plt.figure()
    plt.imshow(new4 / 255)


    plt.show()


if __name__ == "__main__":
    main()