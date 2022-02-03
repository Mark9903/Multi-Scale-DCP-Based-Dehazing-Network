import cv2
import numpy as np

img1ori = './fh_outdoor/OriSGP_Bing_588.png'
img1dc = './fh_outdoor_dcresults/DC1.png'
img1gridgan = './fh_outdoor_gridresults/OriSGP_Bing_588.png'
img1grad = './fh_outdoor_gradimg/gradimg1.png'

img2ori = './fh_outdoor/OriHazyDr_Google_396.jpeg'
img2dc = './fh_outdoor_dcresults/DC2.png'
img2gridgan = './fh_outdoor_gridresults/OriHazyDr_Google_396.jpeg'
img2grad = './fh_outdoor_gradimg/gradimg2.png'

img3ori = './fh_outdoor/OriMLS_Google_585.png'
img3dc = './fh_outdoor_dcresults/DC3.png'
img3gridgan = './fh_outdoor_gridresults/OriMLS_Google_585.png'
img3grad = './fh_outdoor_gradimg/gradimg3.png'

img4ori = './fh_outdoor/OriNW_Google_837.jpeg'
img4dc = './fh_outdoor_dcresults/DC4.png'
img4gridgan = './fh_outdoor_gridresults/OriNW_Google_837.jpeg'
img4grad = './fh_outdoor_gradimg/gradimg4.png'

def GetFinalImg(r, c, gradimg, dc, hardgan):
    finalimg = np.zeros((r, c, 3))
    for i in range(0, r):
        for j in range(0, c):
            for k in range(0, 3):
                finalimg[i][j][k] = float(gradimg[i][j][0]) / 255 * hardgan[i][j][k] + float(255 - gradimg[i][j][0]) / 255 * dc[i][j][k]
    return finalimg

dc = cv2.imread(img1dc)
hardgan = cv2.imread(img1gridgan)
gradimg = cv2.imread(img1grad)
r, c = gradimg.shape[:2]
finalimg = GetFinalImg(r, c, gradimg, dc, hardgan)
cv2.imwrite("./fh_outdoor_results/finalresult1.png", finalimg)     
print("img1 ok")

dc = cv2.imread(img2dc)
hardgan = cv2.imread(img2gridgan)
gradimg = cv2.imread(img2grad)
r, c = gradimg.shape[:2]
finalimg = GetFinalImg(r, c, gradimg, dc, hardgan)
cv2.imwrite("./fh_outdoor_results/finalresult2.png", finalimg)     
print("img2 ok")

dc = cv2.imread(img3dc)
hardgan = cv2.imread(img3gridgan)
gradimg = cv2.imread(img3grad)
r, c = gradimg.shape[:2]
finalimg = GetFinalImg(r, c, gradimg, dc, hardgan)
cv2.imwrite("./fh_outdoor_results/finalresult3.png", finalimg)     
print("img3 ok")

dc = cv2.imread(img4dc)
hardgan = cv2.imread(img4gridgan)
gradimg = cv2.imread(img4grad)
r, c = gradimg.shape[:2]
finalimg = GetFinalImg(r, c, gradimg, dc, hardgan)
cv2.imwrite("./fh_outdoor_results/finalresult4.png", finalimg)     
print("img4 ok")