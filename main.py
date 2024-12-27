import func
import cv2
import sys
import matplotlib.pyplot as plt
import numpy as np

IMG = cv2.imread("img\\input.png") #이미지 불러오기
if IMG is None:
    print("사진 파일을 점검해주십시오")
    sys.exit()
    
RGB = np.float32(IMG.reshape(-1, 3))
clusters = func.classfy_pixel(RGB) #픽셀 분류

COL = func.get_color(clusters) #색상 추출
COL = COL.astype(int)

while True:
    try:
        scale_percent = int(input("축소될 이미지의 비율을 입력해주세요: "))
        break
    except ValueError:
        print("정수를 입력해주세요.")
        continue

SMALL = func.small_img(IMG, scale_percent) #이미지 크기 조정

SMALL = func.create_pixel(IMG,SMALL,COL,scale_percent) #픽셀 생성

# 이미지 출력
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(IMG, cv2.COLOR_BGR2RGB))
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(SMALL, cv2.COLOR_BGR2RGB))
plt.title('Pixel Image')

plt.show()

# 이미지 저장
cv2.imwrite('img/output.png', cv2.cvtColor(SMALL, None))
