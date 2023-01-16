import cv2
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import timeit
import os


def calculate_target_size(img_size: int, kernel_size: int) -> int:
    num_pixels = 0
    
    for i in range(img_size):
        added = i + kernel_size
        if added <= img_size:
            num_pixels += 1
            
    return num_pixels


def convolve(img: np.array, kernel: np.array) -> np.array:
    tgt_size = calculate_target_size(
        img_size=img.shape[0],
        kernel_size=kernel.shape[0]
    )
   
    k = kernel.shape[0]
    
    # tablica tylko 2D
    convolved_img = np.zeros(shape=(tgt_size, tgt_size))
    
    for i in range(tgt_size): #wiersze
        for j in range(tgt_size):  # kolumny
            mat = img[i:i+k, j:j+k]
            convolved_img[i, j]=multiply_add(mat, kernel)
    return convolved_img


def multiply_add(A,B):
    if  A.shape[1] == B.shape[0]:
        sum =0
        
        rows = A.shape[0]
        cols = B.shape[1]
        for row in range(rows): 
            for col in range(cols):
                sum += A[row, col] * B[row, col]
        
        return sum
    else:
        pass
    


def rgb_convolve(im1: np.array, kernel: np.array):
    im2_size = calculate_target_size(im1.shape[0],kernel.shape[0])
    im2 = np.zeros((im2_size, im2_size, 3))
    for dim in range(im1.shape[-1]):  # loop over rgb channels
        im2[:, :, dim] = convolve(img=im1[:,:,dim], kernel = kernel)

    im2 = im2.astype(np.uint8)
    return im2


def rgb_cut(im1: np.array):
    img_size = im1.shape[0]
    new_img_size = img_size//2

    new_img = np.zeros((new_img_size,new_img_size,3))

    for dim in range(im1.shape[-1]):
        for id_row in range(0,new_img_size):
            for id_col in range(0,new_img_size):
                img_row = id_row*2+1
                img_col = id_col*2+1
                new_img[id_row,id_col,dim] = im1[img_row,img_col,dim]

    return new_img

def rgb_add(im1:np.array):
    img_size = im1.shape[0]
    new_img_size = img_size*2

    new_img = np.zeros((new_img_size,new_img_size,3))

    for dim in range(im1.shape[-1]):
        for id_row in range(0,img_size):
            for id_col in range(0,img_size):
                img_row = id_row*2+1
                img_col = id_col*2+1
                new_img[img_row,img_col,dim] = im1[id_row,id_col,dim]

    return new_img


def img_pyramids():
    path = './'
    files = os.listdir(path)

    for f in files:
        if f.endswith(".jpg"):
            apple_path = f
        if f.endswith(".jpeg"):
            orange_path = f
        
    apple = cv2.imread(apple_path)

    orange = cv2.imread(orange_path)

    apple = cv2.resize(apple,(512,512))
    orange = cv2.resize(orange,(512,512))

    cv2.imshow('jablko',apple)
    cv2.imshow('pomarancza',orange)


    blur = np.array([
        [1, 4, 6, 4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1, 4, 6, 4, 1]
    ])

    blur = blur*1/256
    blur_laplace = blur*4


    # rozmazane jabluszko
    for iter in range(0,2):
        apple_blured = rgb_convolve(im1=np.array(apple),kernel = blur)
        apple_cut= rgb_cut(apple_blured)
        apple= apple_cut
        
    apple= rgb_convolve(im1=np.array(apple_cut),kernel = blur)

    for iter in range(0,2):
        apple_add = rgb_add(apple)
        apple_laplace = rgb_convolve(im1=np.array(apple_add),kernel = blur_laplace)
        apple = apple_laplace

    cv2.imshow('laplacowe jabluszko', apple)
    filename = 'gaussowe_jabluszko.jpg'
    #cv2.imwrite(filename, apple)

    #rozmazana pomarancza
    for iter in range(0,2):
        orange_blured = rgb_convolve(im1=np.array(orange),kernel = blur)
        orange_cut= rgb_cut(orange_blured)
        orange= orange_cut
        
    orange= rgb_convolve(im1=np.array(orange_cut),kernel = blur)

    for iter in range(0,2):
        orange_add = rgb_add(orange)
        orange_laplace = rgb_convolve(im1=np.array(orange_add),kernel = blur_laplace)
        orange = orange_laplace

    cv2.imshow('laplacowa pomaranczka', orange)
    filename = 'pomaranczka.jpg'

    # pomaranczka zmniejszamy jej wymiary o 100 po czym dodajemy u gory i na dole kolumny w kolorze tla o 50
    blur_size = orange.shape[0]

    #jabukowapomaranczka
    apple_orange = np.hstack((apple[:,:blur_size//2],orange[:,blur_size//2:]))
    cv2.imshow('jabukopomaranczka', apple_orange)


times = timeit.timeit(stmt = 'img_pyramids()', setup = 'from __main__ import img_pyramids',number = 1)
print("Wykonanie kodu zajelo [s]")
print(times) 
cv2.waitKey()

