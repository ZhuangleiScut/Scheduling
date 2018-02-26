# -*- coding:UTF-8 -*-
import cv2
import dlib
import math
from imutils import face_utils
import time
import pandas as pd
from pandas import DataFrame
from PIL import Image
from PIL.ExifTags import TAGS

"""setDLib"""
p = 'shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)


def get_pitures(i):
    # 构造文件路径
    filepath = './face/netlab_530-' + str(i) + '.jpg'
    print(filepath)
    image = cv2.imread(filepath)
    img = Image.open(filepath)
    ret = {}
    if hasattr(img,'_getexif'):
        exifinfo = img._getexif()
        if exifinfo != None:

            for tag, value in exifinfo.items():
                decoded = TAGS.get(tag, tag)

                ret[decoded] = value
    return (image,exifinfo)


# 获取脸部特征点（68个）
def get_shape(gray, rect):
    # determine the facial landmarks for the face region, then
    # convert the facial landmark (x, y)-coordinates to a NumPy
    # array
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)
    return shape


# 判断睁眼
def testEye(landmark):
    # 左眼眼角
    point36 = landmark[36]
    # 有眼眼角
    point45 = landmark[45]
    # 计算眼间距
    eyeDistance = math.sqrt(math.pow((point36[0] - point45[0]), 2) + math.pow((point36[1] - point45[1]), 2))

    # 计算眼面积
    pointsRight = [landmark[36], landmark[37], landmark[38], landmark[39], landmark[40], landmark[41]]
    pointsLeft = [landmark[42], landmark[43], landmark[44], landmark[45], landmark[46], landmark[47]]
    areaRight = eyeArea(pointsRight)
    areaLeft = eyeArea(pointsLeft)
    temp = (areaRight + areaLeft) / (math.pow(eyeDistance, 2))
    return temp


# 求眼睛面积
def eyeArea(points):
    point0 = points[0]
    point1 = points[1]
    point2 = points[2]
    point3 = points[3]
    point4 = points[4]
    point5 = points[5]

    tri0 = triangleArea(point0, point1, point5)
    tri1 = triangleArea(point1, point5, point2)
    tri2 = triangleArea(point5, point2, point4)
    tri3 = triangleArea(point2, point4, point3)

    triArea = tri0 + tri1 + tri2 + tri3
    return triArea


# 求三角形面积
def triangleArea(point0, point1, point2):
    # a = 1.0/2.0(x1*(y2-y3)+x2*(y3-y1)+x3*(y1-y2))
    a1 = point0[0] * (point1[1] - point2[1])
    a2 = point1[0] * (point2[1] - point0[1])
    a3 = point2[0] * (point0[1] - point1[1])

    a = abs(0.5 * (a1 + a2 + a3))
    # print(a)
    return a


# 获取专注度
def get_focus(shape):
    result = testEye(shape)
    return result


# 测试每一个人
def test_pre(emotions, focus):
    attention = 0
    get = 0
    if (focus > 0.018):
        attention = 1
    if (len(emotions) > 0):
        anger = emotions['anger']
        contempt = emotions['contempt']
        disgust = emotions['disgust']
        fear = emotions['fear']
        happiness = emotions['happiness']
        neutral = emotions['neutral']
        sadness = emotions['sadness']
        surprise = emotions['surprise']
        e = [anger, contempt, disgust, fear, happiness, neutral, sadness, surprise]
        emotion = e.index(max(e))
        if (attention == 1 and (emotion == 4 or emotion == 5)):
            get = 1
    return (attention, get)


if __name__ == '__main__':
    print('开始检测')

    # 打开Excel文件
    data = pd.read_excel('result.xlsx', sheetname=0)
    temp = 0

    for temp in range(0, 50):
        # 开始获取图片时间戳
        time_get_begin = time.time()
        print('begin:', time_get_begin)

        total = 0
        listen = 0
        understand = 0

        # 获取图片，图片里有时间、教室信息
        image, exif = get_pitures(temp)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print('exif:', exif)
        print(exif[256])
        print(exif[257])
        # resolution = (exif[])

        rects = detector(gray, 1)
        print('识别人脸数:' + str(len(rects)))

        # 总人数total
        total = len(rects)
        # loop over the face detections每张人脸
        for (i, rect) in enumerate(rects):
            # print('检测到人脸:')
            l = rect.left()
            r = rect.right()
            t = rect.top()
            b = rect.bottom()
            # print(l, r, t, b)
            roiImage = image[t:b, l:r]
            # cv2.imwrite('sss.jpg', roiImage)

            # 获取脸部特征点
            shape = get_shape(image, rect)
            # 获取专注度
            focus = get_focus(shape)
            print('focus:' + str(focus))

        # 结束的时候获取时间戳
        time_get_end = time.time()
        print('end:', time_get_end)
        print(time_get_end - time_get_begin)

        # 把记录下来是数据写入Excel文件
        data['编号'][temp] = temp
        data['帧开始时间'][temp] = time_get_begin
        data['帧结束时间'][temp] = time_get_end
        data['帧处理时间'][temp] = time_get_end - time_get_begin
        temp = temp + 1
        # 将更新写到新的Excel中
        DataFrame(data).to_excel('ji.xlsx')
