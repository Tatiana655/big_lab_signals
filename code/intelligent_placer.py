import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import cv2

[UP, DOWN, LEFT, RIGHT] = ["UP","DOWN","LEFT","RIGHT"]

OBJ =[BLACK_OBJ, DIVIDER,ERASER,LOUPE,MARKER,PENCIL,STAR,STIKER,TARDIS,VIRT] = ['black_obj','divider','eraser','loupe','marker','pencil','star','sticker','tardis','Virt']
some_obj = OBJ.copy()
some_obj.remove(LOUPE)
some_obj.remove(STAR)
some_obj.remove(STIKER)

IMG_SIZE = [2592,4608]

def check_image(path_to_jpg_image_on_local_computer):
    '''
    Функция проверки размещения объектов внутри контура
    :param path_to_jpg_image_on_local_computer: ходная картинка соответсвующая входным условиям
    :return: True or False'''
    img = cv2.imread(path_to_jpg_image_on_local_computer, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    polygon, pos, paper = find_polygon_paper(img)
    nanes, cnt = identify_obj(img, paper, pos)
    res, cap_opt, arg_opt, phi_opt = knapsack_problem(polygon, nanes, cnt, 50)
    return res

def find_polygon_paper(img):
    '''
    search polygon and A4 paper
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
    board_area = 12700
    # remove contours with small area
    for cnt in contours:
        tmp = cv2.contourArea(cnt)
        if tmp >board_area:
            area_cnt.append(tmp)
            good_cnt.append(cnt)
            M = cv2.moments(cnt)
            cx.append(int(M['m10'] / M['m00']))
            cy.append(int(M['m01'] / M['m00']))

    # search paper and polygon
    for i in range(len(cx)-1):
        if np.linalg.norm([cx[i]-cx[i+1], cy[i]-cy[i+1]]) < 50:
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

    idx = np.argmax(area_cnt)
    paper = good_cnt[idx]
    good_cnt.pop(idx)
    return polygon,pos, paper

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
    :param min_color: минимальний пропускаемый фильтром цвет
    :return: список контуров и их иерархию
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
    Нахождение и идентфикация контуров объектов
    :param img: input img
    :param paper: paper contour/ контур белого листа A4
    :param pos: polygon pos/ позиция многоугольника
    :return:
    :obj: список обнаруженных объектов
    :good_cnt: список контуров для каждого объекта
    '''
    good_cnt =[]
    idx_good_cnt = []
    obj = []
    img2 = crop_img(img,paper,pos)
    contours, hierarchy = get_contours(img2,[130, 120, 100])
    count = len(contours)-1  # количество объектов

    # проверка на лупу, у неё есть вложенный контур.
    check_in = np.where(np.array(hierarchy[0])[:, -1] > 0)[0]

    for id, num in enumerate(check_in):
        cnt = contours[np.array(hierarchy[0])[:, -1][num]]
        tmp = cv2.contourArea(cnt)
        if tmp > 12700:
            obj.append(LOUPE)
            good_cnt.append(contours[np.array(hierarchy[0])[:, -1][num]])
            idx_good_cnt.append(np.array(hierarchy[0])[:, -1][num])
            count -= 2
        else:
            count-=1

    if count == 0:
        return obj, good_cnt

    # проверка на stiker
    filter = cv2.medianBlur(img2, 15)
    filter = cv2.inRange(filter, np.array([160, 145, 42]), np.array([205, 186, 81]))
    yell_cnt, h= cv2.findContours(filter, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    yellow_check = np.where(filter != 0)

    if len(yellow_check[0]) > 10:
        x = [np.argmin(yellow_check[1]), np.argmax(yellow_check[1])]
        y = [np.argmin(yellow_check[0]), np.argmax(yellow_check[0])]
        for idxc, cnt in enumerate(contours):
            s = cv2.contourArea(cnt)
            M = cv2.moments(cnt)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            if x[0]<=cx and cx <= x[1] and y[0]<=cy and cy <= y[1] and s>15700:
                #good_cnt.append(contours[idxc])
                idx_good_cnt.append(idxc)
        good_cnt.append(yell_cnt[0])
        obj.append(STIKER)
        count -= 1
    if count == 0:
        return obj, good_cnt

    #проверка на звезды, опираясь на геометрическе совойства
    for idxc, cnt in enumerate(contours):
        (x1, y1), rad = cv2.minEnclosingCircle(cnt)
        area = cv2.contourArea(cnt)
        err = abs(area/(np.pi*rad ** 2))
        if (err>0.45)and(err<0.53):
            obj.append(STAR)
            good_cnt.append(contours[idxc])
            idx_good_cnt.append(idxc)
            count-=1
    if count==0:
        return obj, good_cnt

    # проверка на всё стальное, с помощью sift
    template = {}
    area = {}
    cx_obj = {}
    cy_obj = {}

    for obj1 in some_obj:
        template[obj1] = cv2.imread('objects/' + obj1 + '.jpg')
        c, h = get_contours(template[obj1] , [130, 120, 100])
        area[obj1] = cv2.contourArea(c[0])
        M = cv2.moments(c[0])
        cx_obj[obj1] = int(M['m10'] / M['m00'])
        cy_obj[obj1] = int(M['m01'] / M['m00'])
        template[obj1] = cv2.imread('objects/' + obj1 + '.jpg',0)

    img2 = cv2.cvtColor(img2.copy(), cv2.COLOR_BGR2GRAY)
    for obj1 in some_obj:
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(template[obj1], None)
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

        if len(good) >=20:
            # поиск нужного контура
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            A = np.array([cx_obj[obj1],cy_obj[obj1]])
            pts1 = np.float32([A]).reshape(-1, 1, 2)
            dst = np.int64(cv2.perspectiveTransform(pts1, M))
            # найти контур с самым близким центром к dist
            mini = 100000
            for idxc,cnt in enumerate(contours):
                M = cv2.moments(cnt)
                x = int(M['m10'] / M['m00'])
                y = int(M['m01'] / M['m00'])
                err = np.linalg.norm(np.array([x,y]) - dst)
                if err < mini:
                    mini = err
                    closest = cnt
                    id_closest = idxc
            if mini < 500:
                obj.append(obj1)
                good_cnt.append(closest)
                idx_good_cnt.append(id_closest)
                count-=1
            if count == 0:
                return obj, good_cnt

    #если чего не нашли, сравниваем по площади
    none_obj = OBJ.copy()
    none_obj.remove(LOUPE)
    none_obj.remove(STAR)
    none_obj.remove(STIKER)
    contours = list(contours)

    # удвление уже определённых контуров
    if len(idx_good_cnt)>0:
        idx_good_cnt.sort(reverse=True)
        for id in idx_good_cnt:
            contours.pop(id)
    # смотрим что по площади сравниваем оставшиеся
    if count != 0:
        m = -1
        new_arr = contours.copy()
        new_arr.reverse()
        for idxc, c in enumerate(new_arr):
            tmp = cv2.contourArea(c)
            if tmp > m:
                m = tmp
            if tmp <= 15700: # удаляем маленькие
                contours.pop(len(contours)-idxc-1)

        for ob in obj:
            if (ob != STAR) and (ob != STIKER) and (ob != LOUPE):
                none_obj.remove(ob)
        for el_cnt in contours:
            mini = 10e10
            for ob in none_obj:
                err = abs(cv2.contourArea(el_cnt)-area[ob])
                if err < mini:
                    mini = err
                    closest = ob
            count-=1
            obj.append(closest)
            good_cnt.append(el_cnt)
            none_obj.remove(closest)

    return obj, good_cnt


def cap(polygon,contours):
    '''
    Функция минимизации. Пересечение контуров и области за многоугольником
    :param polygon: контур многоугольника
    :param contours: контуры объектов
    :return: сумма пикселей в пересечении
    '''
    poly_arr = np.ones((IMG_SIZE[1],IMG_SIZE[0], 3), np.uint8) #np.ones((IMG_SIZE[1],IMG_SIZE[0]))
    cv2.fillPoly(poly_arr,[polygon],(0,0,0))
    for cnt in contours:
        cnt_arr = np.zeros((IMG_SIZE[1],IMG_SIZE[0], 3), np.uint8)
        cv2.fillPoly(cnt_arr, [cnt], (1,1,1))
        poly_arr = cv2.add(poly_arr,cnt_arr)

    filtered = cv2.inRange(poly_arr, np.array([2,2,2]), np.array([255, 255, 255]))
    res = sum(sum(filtered))
    return res


def cart2pol(x, y):
    '''
    перевод кооринат из декартовой в полярную СК
    :param x:
    :param y:
    :return: theta, rho
    '''
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return theta, rho


def pol2cart(theta, rho):
    '''
    перевод кооринат из полярной в декартову
    :param theta: угол в полярной СК
    :param rho: радиус в полярной СК
    :return: x,y в декартовой СК
    '''
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y


def rotate_contour(cnt, angle):
    '''
    contour rotation
    :param cnt: контур, который нужно повернуть
    :param angle: угол в градусах на который нужно повернуть degrees
    :return:
    :cnt_rotation: контур к которому применен поворот
    '''
    M = cv2.moments(cnt)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    cnt_norm = cnt - [cx, cy]

    coordinates = cnt_norm[:, 0, :]
    xs, ys = coordinates[:, 0], coordinates[:, 1]
    thetas, rhos = cart2pol(xs, ys)

    thetas = np.rad2deg(thetas)
    thetas = (thetas + angle) % 360
    thetas = np.deg2rad(thetas)

    xs, ys = pol2cart(thetas, rhos)

    cnt_norm[:, 0, 0] = xs
    cnt_norm[:, 0, 1] = ys

    cnt_rotated = cnt_norm + [cx, cy]
    cnt_rotated = cnt_rotated.astype(np.int32)

    return cnt_rotated

def translate(cnt,cx,cy):
    '''
    Translation
    :param cnt: контур, который надо переместить
    :param cx: перемещение по x
    :param cy: перемещение по y
    :return:
    :cnt_norm: контур к которому применено перемещение
    '''
    cnt_norm = cnt + [cx, cy]
    return cnt_norm

def wrap(cnt,cx,cy,angle):
    '''
    Перемещение и поворот контуров
    :param cnt: контуры
    :param cx: перемещение по x
    :param cy: перемещение по y
    :param angle: угол поворота
    :return:
    :cnt_trans: контур к которому применениы перемещение и поворот
    '''
    cnt_rotate = rotate_contour(cnt,angle)
    cnt_trans = np.array(translate(cnt_rotate,cx,cy),np.int32)
    return cnt_trans

def MonteCarlo(max_iter, N, polygon,contours):
    '''
    Минимизация методом Монте Карло
    :param max_iter: максимальное количесвто итераций
    :param N: Количество точек генерирующихся за одну итерацию
    :param polygon: контур многоугольника
    :param contours: контуры объектов
    :return:
    :res: True/False (помещаются ли объекты)
    :cap_opt: значение функции минимизации пересечения
    :arg_opt: значения кординат [x,y] для каждого контура последовательно в виде одного массива
    :phi_opt: значения оптимального угла поворота для каждого из контуров
    '''
    (x_poly, y_poly), rad_poly = cv2.minEnclosingCircle(polygon)
    #first
    arg_opt = np.zeros(2*len(contours))
    phi_opt = np.zeros(len(contours))
    k = 0
    tmp_cnts = list(contours).copy()
    for i,cnt in enumerate(contours):
        (x, y), rad = cv2.minEnclosingCircle(cnt)
        contours[i] = wrap(cnt, -x, -y, 0)
        tmp_cnts[i] = np.array(wrap(contours[i],x_poly,y_poly,0),np.int32)
        arg_opt[k] = x_poly
        arg_opt[k+1] = y_poly
        k+=2

    cap_opt = cap(polygon, tmp_cnts)
    print("first", cap_opt)
    if ( cap_opt < 1):
        return True, cap_opt, arg_opt, phi_opt

    # next
    r = max_iter
    while (r>0):
        r-=1
        points = stats.multivariate_normal.rvs(arg_opt, 5*rad_poly*np.eye(2*len(contours)), size=N)
        phi_rand = [np.random.randint(360, size=len(contours)) for h in range(N)]
        for i in range(N):
            #передвинуть контуры
            arg_res = points[i]
            phi_res = phi_rand[i]
            new_cnts = []
            k=0
            for idc, cnt in enumerate(contours):
                new_cnts.append(wrap(cnt,arg_res[k],arg_res[k+1],phi_res[idc]))
                k+=2

            cap_res = cap(polygon,new_cnts)
            if cap_res <= cap_opt:
                print(cap_res)
                cap_opt = cap_res
                arg_opt = arg_res
                phi_opt = phi_res


            if cap_res < 1:
                print("iteration", max_iter-r)
                return True, cap_opt, arg_opt, phi_opt

    return False, cap_opt, arg_opt, phi_opt


def knapsack_problem(polygon, names, contours,max_iter):
    '''
    Решение задачи упаковки
    :param polygon: контур многоугольника
    :param names: названия объектов на фотографии
    :param contours: контуры объектов на фотографии
    :param max_iter: максимальное число итераций для метода Монте Карло
    :return:
    :res: True/False (помещаются ли объекты)
    :cap_opt: значение функции минимизации пересечения или (если не прошло по тривиальным условиям) -1
    :arg_opt: значения кординат [x,y] для каждого контура последовательно в виде одного массива или (если не прошло по тривиальным условиям) []
    :phi_opt: значения оптимального угла поворота для каждого из контуров или (если не прошло по тривиальным условиям) []
    '''
    #проверка за звёзды/лупы
    if LOUPE in names:
        if STAR in names:
            id_star = names.index(STAR)
            contours.pop(id_star)
            names.pop(id_star)

    # сначала тривиальная проверка по площади/диамету, потом минимизация
    area_poly = cv2.contourArea(polygon)
    (x_poly,y_poly),rad_poly = cv2.minEnclosingCircle(polygon)
    diam_poly = 2*rad_poly
    area_cnt = []
    x_cnt = []
    y_cnt = []
    diam_cnt = []
    for cnt in contours:
        area_cnt.append(cv2.contourArea(cnt))
        (x, y), rad = cv2.minEnclosingCircle(cnt)
        x_cnt.append(x)
        y_cnt.append(y)
        diam_cnt.append(2*rad)

    cap_opt = -1
    arg_opt = []
    phi_opt = []
    if area_poly < sum(np.array(area_cnt)):
        return False, cap_opt, arg_opt, phi_opt
    if diam_poly < max(np.array(diam_cnt)) :
        return False, cap_opt, arg_opt, phi_opt

    #нетривиальная проверка
    return MonteCarlo(max_iter, len(contours) * 10, polygon, contours)
