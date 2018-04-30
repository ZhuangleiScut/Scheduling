import cv2
from PIL import Image
from keras.models import load_model
import numpy as np
import time
import pandas as pd
from pandas import DataFrame
from xlwt import Workbook
import os

from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.inference import load_image
from utils.preprocessor import preprocess_input

# parameters for loading data and images
# image_path = sys.argv[1]
detection_model_path = '../trained_models/detection_models/haarcascade_frontalface_default.xml'
emotion_model_path = '../trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
gender_model_path = '../trained_models/gender_models/simple_CNN.81-0.96.hdf5'
emotion_labels = get_labels('fer2013')
gender_labels = get_labels('imdb')
font = cv2.FONT_HERSHEY_SIMPLEX

# hyper-parameters for bounding boxes shape
gender_offsets = (30, 60)
gender_offsets = (10, 10)
emotion_offsets = (20, 40)
emotion_offsets = (0, 0)

# loading models
face_detection = load_detection_model(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
gender_classifier = load_model(gender_model_path, compile=False)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]
gender_target_size = gender_classifier.input_shape[1:3]


def dealImage(image_path):
    # loading images
    rgb_image = load_image(image_path, grayscale=False)
    gray_image = load_image(image_path, grayscale=True)
    gray_image = np.squeeze(gray_image)
    gray_image = gray_image.astype('uint8')

    # face_area
    face_area = 0

    faces = detect_faces(face_detection, gray_image)
    for face_coordinates in faces:
        x1, x2, y1, y2 = apply_offsets(face_coordinates, gender_offsets)
        rgb_face = rgb_image[y1:y2, x1:x2]

        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]

        try:
            rgb_face = cv2.resize(rgb_face, (gender_target_size))
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            continue

        rgb_face = preprocess_input(rgb_face, False)
        rgb_face = np.expand_dims(rgb_face, 0)
        gender_prediction = gender_classifier.predict(rgb_face)
        gender_label_arg = np.argmax(gender_prediction)
        gender_text = gender_labels[gender_label_arg]

        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_label_arg = np.argmax(emotion_classifier.predict(gray_face))
        emotion_text = emotion_labels[emotion_label_arg]

        if gender_text == gender_labels[0]:
            color = (0, 0, 255)
        else:
            color = (255, 0, 0)

        draw_bounding_box(face_coordinates, rgb_image, color)
        draw_text(face_coordinates, rgb_image, gender_text, color, 0, -20, 1, 2)
        draw_text(face_coordinates, rgb_image, emotion_text, color, 0, -50, 1, 2)

        # print('检测到人脸:')
        l = face_coordinates.left()
        r = face_coordinates.right()
        t = face_coordinates.top()
        b = face_coordinates.bottom()
        # print(l, r, t, b)
        area = abs(r - l) * abs(b - t)
        face_area += area

    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    # cv2.imwrite('../images/predicted_test_image.png', bgr_image)
    return bgr_image, len(faces), face_area



def memory_stat():
    mem = {}
    f = open("/proc/meminfo")
    lines = f.readlines()
    f.close()
    for line in lines:
        if len(line) < 2: continue
        name = line.split(':')[0]
        var = line.split(':')[1].split()[0]
        mem[name] = long(var) * 1024.0
    mem['MemUsed'] = mem['MemTotal'] - mem['MemFree'] - mem['Buffers'] - mem['Cached']
    return mem


# get CPU state
def cpu_stat():
    cpu = []
    cpuinfo = {}
    f = open("/proc/cpuinfo")
    lines = f.readlines()
    f.close()
    for line in lines:
        if line == '\n':
            cpu.append(cpuinfo)
            cpuinfo = {}
        if len(line) < 2: continue
        name = line.split(':')[0].rstrip()
        var = line.split(':')[1]
        cpuinfo[name] = var
    return cpu


# get load state
def load_stat():
    loadavg = {}
    f = open("/proc/loadavg")
    con = f.read().split()
    f.close()
    loadavg['lavg_1'] = con[0]
    loadavg['lavg_5'] = con[1]
    loadavg['lavg_15'] = con[2]
    loadavg['nr'] = con[3]
    loadavg['last_pid'] = con[4]
    return loadavg


# get Uptime
def uptime_stat():
    uptime = {}
    f = open("/proc/uptime")
    con = f.read().split()
    f.close()
    all_sec = float(con[0])
    MINUTE, HOUR, DAY = 60, 3600, 86400
    uptime['day'] = int(all_sec / DAY)
    uptime['hour'] = int((all_sec % DAY) / HOUR)
    uptime['minute'] = int((all_sec % HOUR) / MINUTE)
    uptime['second'] = int(all_sec % MINUTE)
    uptime['Free rate'] = float(con[1]) / float(con[0])
    return uptime


# get net state
def net_stat():
    net = []
    f = open("/proc/net/dev")
    lines = f.readlines()
    f.close()
    for line in lines[2:]:
        con = line.split()
        intf = dict(
            zip(
                ('interface', 'ReceiveBytes', 'ReceivePackets',
                 'ReceiveErrs', 'ReceiveDrop', 'ReceiveFifo',
                 'ReceiveFrames', 'ReceiveCompressed', 'ReceiveMulticast',
                 'TransmitBytes', 'TransmitPackets', 'TransmitErrs',
                 'TransmitDrop', 'TransmitFifo', 'TransmitFrames',
                 'TransmitCompressed', 'TransmitMulticast'),
                (con[0].rstrip(":"), int(con[1]), int(con[2]),
                 int(con[3]), int(con[4]), int(con[5]),
                 int(con[6]), int(con[7]), int(con[8]),
                 int(con[9]), int(con[10]), int(con[11]),
                 int(con[12]), int(con[13]), int(con[14]),
                 int(con[15]), int(con[16]),)
            )
        )

        net.append(intf)
    return net


# get disk state
def disk_stat():
    import os
    hd = {}
    disk = os.statvfs("/")
    hd['available'] = disk.f_bsize * disk.f_bavail
    hd['capacity'] = disk.f_bsize * disk.f_blocks
    hd['used'] = disk.f_bsize * disk.f_bfree
    return hd


if __name__ == '__main__':
    i = 0;
    while os.path.exists(str(i) + '.xls'):
        i += 1
    if not os.path.exists(str(i) + '.xls'):
        # build excel
        book = Workbook(encoding='utf-8')
        sheet1 = book.add_sheet('Sheet 1')
        sheet1.write(0, 0, "id")
        for a in range(300):  # y
            sheet1.write(a + 1, 0, a)
            # print(task_list[a])
        sheet1.write(0, 1, 'task')
        sheet1.write(0, 2, 'image_size')
        sheet1.write(0, 3, 'recognition')
        sheet1.write(0, 4, 'face_num')
        sheet1.write(0, 5, 'face_area')
        sheet1.write(0, 6, 'cpu_core')
        sheet1.write(0, 7, 'mem')
        sheet1.write(0, 8, 'disk')
        sheet1.write(0, 9, 'time')
        # save Excel book.save('path/name.xls')
        book.save(str(i) + '.xls')
        print('builded Excel.')
    # open Excel
    dt = pd.read_excel(str(i) + '.xls')
    # deal image
    for t in range(300):
        image_path = "../face/" + str(t) + '.jpg'
        time_start = time.time()
        bgr_image, face_num, face_area = dealImage(image_path)
        cv2.imwrite('../images/predicted_test_image' + str(t) + '.png', bgr_image)

        # get hardware message
        mem = memory_stat()
        cpu = cpu_stat()
        load = load_stat()
        uptime = uptime_stat()
        net_state = net_stat()
        disk_state = disk_stat()

        time_end = time.time()
        ptime = time_end - time_start
        print(t, ptime)
        dt['task'][t] = t
        dt['time'][t] = ptime

        dt['image_size'][t] = os.path.getsize(image_path)

        img = Image.open(image_path)
        if hasattr(img, '_getexif'):
            exifinfo = img._getexif()
        if exifinfo[256] == 1920 and exifinfo[257] == 1080:
            # 1920,1080分辨率设为1
            dt['recognition'][t] = 1
        if exifinfo[256] == 1280 and exifinfo[257] == 960:
            # 1920,1080分辨率设为1
            dt['recognition'][t] = 2
        if exifinfo[256] == 1280 and exifinfo[257] == 720:
            # 1920,1080分辨率设为1
            dt['recognition'][t] = 3

        dt['face_num'][t] = face_num
        dt['face_area'][t] = face_area
        dt['cpu_core'][t] = float(cpu[0]['cpu cores'])
        dt['mem'][t] = mem['MemTotal']
        dt['disk'][t] = disk_state['available']

    DataFrame(dt).to_excel(str(i) + '.xls')
