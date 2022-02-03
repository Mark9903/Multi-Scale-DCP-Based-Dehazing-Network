import cv2
import numpy as np
from adaptACE import adaptContrastEnhancement

img1ori = './fh_outdoor/OriSGP_Bing_588.png'
img1dc = './fh_outdoor_dcresults/DC1.png'
img1gridgan = './fh_outdoor_gridresults/OriSGP_Bing_588.png'

img2ori = './fh_outdoor/OriHazyDr_Google_396.jpeg'
img2dc = './fh_outdoor_dcresults/DC2.png'
img2gridgan = './fh_outdoor_gridresults/OriHazyDr_Google_396.jpeg'

img3ori = './fh_outdoor/OriMLS_Google_585.png'
img3dc = './fh_outdoor_dcresults/DC3.png'
img3gridgan = './fh_outdoor_gridresults/OriMLS_Google_585.png'

img4ori = './fh_outdoor/OriNW_Google_837.jpeg'
img4dc = './fh_outdoor_dcresults/DC4.png'
img4gridgan = './fh_outdoor_gridresults/OriNW_Google_837.jpeg'

def DarkChannel(im,sz):
    b,g,r = cv2.split(im)            # 分割
    dc = cv2.min(cv2.min(r,g),b);    # 取最小值
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(sz,sz)) # 创建矩阵型kernel
    dark = cv2.erode(dc,kernel)      # 用上面创建的kernel进行腐蚀操作 灰度图腐蚀：像素点取卷积核覆盖到范围的最小值
    return dark

def Guidedfilter(im,p,r,eps): # 导向滤波 r = 60 eps = 0.1 im = gray r = estimate transmission map
    mean_I = cv2.boxFilter(im, cv2.CV_64F,(r,r)); # CV_64F等同于CV_64FC1 表示双精度浮点数一通道 方框滤波 模糊作用
    mean_p = cv2.boxFilter(p, cv2.CV_64F,(r,r));
    mean_Ip = cv2.boxFilter(im*p,cv2.CV_64F,(r,r));
    cov_Ip = mean_Ip - mean_I*mean_p;             # cov_Ip 中基本为负数
    
    mean_II = cv2.boxFilter(im*im,cv2.CV_64F,(r,r));
    var_I = mean_II - mean_I*mean_I;

    a = cov_Ip/(var_I + eps);
    b = mean_p - a*mean_I;

    mean_a = cv2.boxFilter(a,cv2.CV_64F,(r,r));
    mean_b = cv2.boxFilter(b,cv2.CV_64F,(r,r));

    q = mean_a*im + mean_b;
    return q;

def GetGradImg(r, c, gradimgSize, phrase, ori):
    gradimg = np.zeros((r, c))
    dataVector = np.zeros(gradimgSize)
    for i in range(0, r):
        for j in range(0, c):
            maxx = 0
            for k in range(0, 3):
                if phrase == 1:
                    localGrad = abs(int(ori[i][j][k]) - int(ori[i + 1][j][k]))
                    
                if phrase == 2:
                    localGrad = abs(int(ori[i][j][k]) - int(ori[i][j + 1][k]))
                    
                if phrase == 3:
                    subEle1 = abs(int(ori[i][j][k]) - int(ori[i + 1][j][k]))
                    subEle2 = abs(int(ori[i][j][k]) - int(ori[i][j + 1][k]))
                    
                    localGrad = (subEle1 ** 2 + subEle2 ** 2) ** 0.5
                    # localGrad = ori[i][j][k]
                # maxx += localGrad
                maxx = max(maxx, localGrad)
            dataVector[i * c + j] = maxx
            
    dataVectorIndex = dataVector.argsort()
    lcur = -1
    for i in range(0, gradimgSize):
        cur = dataVectorIndex[i] # 第i小的梯度的下标是cur
        x = cur // c
        y = cur % c
        
        if lcur != -1:
            if abs(dataVector[cur] - dataVector[lcur]) < 1e-8:
                gradimg[x][y] = gradimg[lcur // c][lcur % c]
            else:
                gradimg[x][y] = float(i) / float(gradimgSize)
                # print(gradimg[x][y])
                lcur = dataVectorIndex[i]
        else:
            gradimg[x][y] = float(i) / float(gradimgSize)
            # print(gradimg[x][y])
            lcur = dataVectorIndex[i]
            
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # gradimg = cv2.dilate(gradimg, kernel, iterations = 1)
    # gradimg = cv2.blur(gradimg, (7, 7))
    # gray = cv2.cvtColor(ori,cv2.COLOR_BGR2GRAY)
    gradimg = cv2.resize(gradimg, (gradimg.shape[1] + 1, gradimg.shape[0] + 1))
    # gray = np.float64(gray)/255;
    # cv2.imshow("gray", gray)
    # cv2.imshow("gradimg", gradimg)
    # r = 120;
    # eps = 0.0001;
    # gradimg = adaptContrastEnhancement(gradimg, 5, 100)
    # gradimg = Guidedfilter(gray, gradimg,r,eps);
    return gradimg

def GetLap(img):
    r = 60
    eps = 0.0001
    dc = DarkChannel(img, 15)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.Laplacian(img, -1)
    img = cv2.equalizeHist(img)
    img = cv2.blur(img, (7, 7))
    nimg = np.full(img.shape, 255, dtype = 'uint8')
    img = nimg - img
    # cv2.imshow("img", img)
    # cv2.imshow("dc", dc)
    img = Guidedfilter(img, dc, r, eps)
    img = nimg - img
    return img


ori = cv2.imread(img1ori)
dc = cv2.imread(img1dc)
gridgan = cv2.imread(img1gridgan)
gradimgSize = (ori.shape[0] - 1) * (ori.shape[1] - 1)
r, c = ori.shape[:2]
r = r - 1
c = c - 1
laplacian = GetLap(ori)
# 横向并排对比显示
# gradimg = GetGradImg(r, c, gradimgSize, 3, ori)
cv2.imwrite("./fh_outdoor_gradimg/gradimg1.png", laplacian)
cv2.waitKey()
print("img1 ok")

ori = cv2.imread(img2ori)
dc = cv2.imread(img2dc)
gridgan = cv2.imread(img2gridgan)
gradimgSize = (ori.shape[0] - 1) * (ori.shape[1] - 1)
r, c = ori.shape[:2]
r = r - 1
c = c - 1
laplacian = GetLap(ori) 
# 横向并排对比显示
# gradimg = GetGradImg(r, c, gradimgSize, 3, ori)
cv2.imwrite("./fh_outdoor_gradimg/gradimg2.png", laplacian)
cv2.waitKey()
print("img2 ok")

ori = cv2.imread(img3ori)
dc = cv2.imread(img3dc)
gridgan = cv2.imread(img3gridgan)
gradimgSize = (ori.shape[0] - 1) * (ori.shape[1] - 1)
r, c = ori.shape[:2]
r = r - 1
c = c - 1
laplacian = GetLap(ori)
# 横向并排对比显示
# gradimg = GetGradImg(r, c, gradimgSize, 3, ori)
cv2.imwrite("./fh_outdoor_gradimg/gradimg3.png", laplacian)
cv2.waitKey()
print("img3 ok")

ori = cv2.imread(img4ori)
dc = cv2.imread(img4dc)
gridgan = cv2.imread(img4gridgan)
gradimgSize = (ori.shape[0] - 1) * (ori.shape[1] - 1)
r, c = ori.shape[:2]
r = r - 1
c = c - 1
laplacian = GetLap(ori)
# 横向并排对比显示
# gradimg = GetGradImg(r, c, gradimgSize, 3, ori)
cv2.imwrite("./fh_outdoor_gradimg/gradimg4.png", laplacian)
cv2.waitKey()
print("img4 ok")