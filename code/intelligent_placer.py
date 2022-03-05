import matplotlib.pyplot as plt
import numpy as np
from imageio import imread, imsave

import cv2

'''
TO-DO: требования про библиотеку и тесты в ноутбуке
план:
1) Найти многоугольник(ok)
реультаты в папке code/test
2) Определение объектов по особым точкам(ok + to-do)
реультаты в файле code/test_res.txt
3) Помещается ли в многоугольник (to-do)
- посмотреть есть ли lope и star
- сопоставить размеры площади, диаметра (max расстояние) мб по этим признакам выкинуть ответ
- задача упаковки (генетический алгоритм (tmp)) (*) тут пока непонятно

'''

[UP, DOWN, LEFT, RIGHT] = ["UP","DOWN","LEFT","RIGHT"]
UNKNOWN = 'UNKNOWN'

names = ['black_obj.jpg','divider.jpg','eraser.jpg','loupe.jpg','marker.jpg','pencil.jpg','star.jpg','sticker.jpg','tardis.jpg','Virt.jpg']
[BLACK_OBJ, DIVIDER,ERASER,LOUPE,MARKER,PENCIL,STAR,STIKER,TARDIS,VIRT] = ['black_obj','divider','eraser','loupe','marker','pencil','star','sticker','tardis','Virt']
OBJ = [BLACK_OBJ, DIVIDER,ERASER,LOUPE,MARKER,PENCIL,STAR,STIKER,TARDIS,VIRT]
some_obj = OBJ
some_obj.remove(LOUPE)
some_obj.remove(STAR)


def check_image(path_to_jpg_image_on_local_computer):
    '''
    in process...
    :param path_to_jpg_image_on_local_computer:
    :return: True or False'''
    return True

def find_polygon_paper(img):
    '''
    search polygon and A4 paper
    to-do: add error processing
    :param img:
    :return: contour of polygon, position polygon, contour of paper
    '''
    contours, hierarchy = get_contours(img,[135, 130, 110])
    good_cnt = []
    cx = []
    cy = []
    polygon = None
    pos = None
    area_cnt = []
    # remove contours with small area
    for cnt in contours:
        tmp = cv2.contourArea(cnt)
        if tmp >12700:
            area_cnt.append(tmp)
            good_cnt.append(cnt)
            M = cv2.moments(cnt)
            cx.append(int(M['m10'] / M['m00']))
            cy.append(int(M['m01'] / M['m00']))
    # search paper and polygon
    for i in range(len(cx)-1):
        if np.linalg.norm([cx[i]-cx[i+1], cy[i]-cy[i+1]]) < 20:
            polygon = good_cnt[i]
            good_cnt.pop(i+1)
            good_cnt.pop(i)
            if len(img) > len(img[0]):
                #вертикальный
                if cy[i]>len(img)/2:
                    pos = DOWN
                else:
                    pos = UP
            else:
                #горизонтальный
                if cx[i]>len(img[0])/2:
                    pos = RIGHT
                else:
                    pos = LEFT
            break

    idx = area_cnt.index(max(area_cnt))
    paper = good_cnt[idx]
    good_cnt.pop(idx)
    return polygon,pos, paper

def test_poly_list():
    for i in range(5, 29):
        print("i = ", i)
        img = imread('test/' + str(i) + '.jpg')

        polygon,pos, paper, good_cnt = find_polygon_paper(img)

        cv2.drawContours(img, [paper], 0, (0, 255, 0), 10)
        cv2.drawContours(img, [polygon], 0, (255, 0, 0), 10)

        plt.imshow(img)
        plt.show()

def crop_img(img,paper,pos):
    '''
    crop img
    :param img:
    :param paper: contour A4 paper
    :param pos: polygon pos
    :return: part of picture with objects
    '''
    img2 = img.copy()
    rect = cv2.minAreaRect(paper)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    box = box.transpose()
    x1 = min(box[0])
    x2 = max(box[0])
    y1 = min(box[1])
    y2 = max(box[1])

    bound = 100
    img2 = np.array(img2[y1 + bound:y2 - bound, x1 + bound:x2 - bound])
    if pos == UP:
        img2 = img2[int(len(img2) / 2):-1]
    elif pos == DOWN:
        img2 = img2[0:int(len(img2) / 2)]
    elif pos == LEFT:
        img2 = img2[:, int(len(img2[0]) / 2):-1]
    elif pos == RIGHT:
        img2 = img2[:, 0:int(len(img2[0]) / 2)]
    return img2

def get_contours(img2,min_color):
    '''
    get contours, hierarchy using filters
    :param img2: crop picture
    :return:
    '''
    filter = cv2.inRange(img2, np.array(min_color), np.array([255, 255, 255]))
    st1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5), (-1, -1))
    filter = cv2.morphologyEx(filter, cv2.MORPH_CLOSE, st1)
    st2 = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50), (-1, -1))
    filter = cv2.morphologyEx(filter, cv2.MORPH_OPEN, st2)
    filter = cv2.medianBlur(filter, 7)

    ret, thresh = cv2.threshold(filter, 127, 255, 0)
    return cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

def identify_obj(img, paper, pos):
    '''
    function identify objects
    :param img: input img
    :param paper: paper contour
    :param pos: polygon pos
    :return: list of objects
    to-do: return linear (size) koef
    '''
    obj = []

    img2 = crop_img(img,paper,pos)

    contours, hierarchy = get_contours(img2,[130, 120, 100])

    count = len(contours)-1  # количество объектов
    #проверка на лупу, у неё есть вложенный контур
    check_in = max(np.array(hierarchy[0])[:,-1])
    if check_in>0:
        obj.append(LOUPE)
        count-=2
    if count == 0:
        return obj

    #проверка на звезды, опираясь на её геометрическе совойства
    for cnt in contours:
        (x, y), rad = cv2.minEnclosingCircle(cnt)
        area = cv2.contourArea(cnt)
        err = abs(area/(np.pi*rad ** 2))
        if (err>0.50)and(err<0.53):
            obj.append(STAR)
            count-=1
    if count==0:
        return obj

    # проверка на всё стальное, с помощью sift
    img2 = cv2.cvtColor(img2.copy(), cv2.COLOR_BGR2GRAY)
    for obj1 in some_obj:
        sift = cv2.SIFT_create()
        template = cv2.imread('objects/' + obj1 + '.jpg', 0)

        kp1, des1 = sift.detectAndCompute(template, None)
        picture = img2
        kp2, des2 = sift.detectAndCompute(picture, None)
        # поиск хороших точек
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        # store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)
        #print(len(good))
        if len(good) >=20:
            obj.append(obj1)
            count-=1
            if count == 0:
                return obj
    #если чего не нашли
    while count != 0:
        obj.append(UNKNOWN)
        count-=1
    return obj

