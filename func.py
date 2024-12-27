import cv2
import numpy as np
from sklearn.cluster import KMeans

def classfy_pixel(img):

    # 중복되는 값을 제거
    unique_pixels = np.unique(img, axis=0)

    # KMeans 알고리즘을 사용하여 픽셀을 x개의 군집으로 분류
    kmeans = KMeans(n_clusters=80)
    kmeans.fit(unique_pixels)
    clusters = kmeans.cluster_centers_
    return clusters

############################################################################################################

def get_color(clusters):
    colors = clusters.round(0)
    return colors

############################################################################################################

def small_img(original, scale_percent):
    # 이미지의 너비와 높이
    width = int(original.shape[1] * scale_percent / 100)
    height = int(original.shape[0] * scale_percent / 100)

    # 새 이미지 크기
    dim = (width, height)

    # 이미지 크기 줄이기
    resized = cv2.resize(original, dim, interpolation = cv2.INTER_AREA)
    
    #1차원 배열로 변환
    resized= np.reshape(resized, (-1, 3))
    
    return resized

############################################################################################################

def create_pixel(original,resized,colors,scale_percent):
    width = int(original.shape[1] * scale_percent / 100)
    height = int(original.shape[0] * scale_percent / 100)
    
    new_image = np.zeros(resized.shape, dtype=np.uint8)
    index = 0
    for i in resized:
        # 픽셀과 가장 가까운 색상을 찾기
        j = np.argmin(np.linalg.norm(i - colors, axis=1)) #각 요소에 대한 유클리드 거리 계산 -> 가장 적은값의 요소 선택
        new_image[index] = colors[j]
        index += 1
    
    #크기조정
    new_image = np.reshape(new_image, (height, width, 3))

    #이미지 확대
    resized = np.zeros((height*6, width*6, 3))

    for i in range(height):
        for j in range(width):
            resized[i*6:i*6+6, j*6:j*6+6] = new_image[i][j]

    pixel = resized.astype(np.uint8)
    return pixel

