#!/usr/bin/env python
# coding: utf-8


import sys
import threading
import asyncio
import time
import socket
import random
from IPython.display import clear_output
import cv2
import numpy as np
from timeit import default_timer as timer
import datetime
import math
import os
import shutil

# data
cwd = os.getcwd()
examples_dir = cwd + "//video//"
bad_video = examples_dir + "bad.wmv"
good_video = examples_dir + "good.wmv"
right_video = examples_dir + "rgood.wmv"
# settings
turn_on_delay = 5

local_run = True

# Average for lamps detector
average_frame_limit = 8
average_images_limit = 40
average_images_l = []
average_images_r = []
index_for_averaging = 0

green_color = (0, 255, 0)
red_color = (0, 0, 255)
yellow_color = (0, 255, 255)
std_width = 1


# VARS
LEFT_FILTER_X = None
LEFT_FILTER_Y = None

LEFT_ANGLE = 1.67
RIGHT_ANGLE = -1.87

LEFT_K = None
LEFT_B = None

RIGHT_FILTER_X = None
RIGHT_FILTER_Y = None

RIGHT_K = None
RIGHT_B = None


class State:
    def __init__(self):
        self.working = False
        self.stable = False
        self.blink = False
        self.detected = False
        self.l_count = 0
        self.l_enabled = False
        self.r_count = 0
        self.r_enabled = False

    def detect_on_left(self):
        if self.detected:
            return
        self.detected = True
        self.l_count = self.l_count + 1
        if self.l_count > 999:
            self.l_count = 0

    def detect_on_right(self):
        if self.detected:
            return
        self.detected = True
        self.r_count = self.r_count + 1
        if self.r_count > 999:
            self.r_count = 0


machine_state = State()


def get_contours(thresh):
    # Find contours
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    return contours


def add_contours(image, contours, color, width):
    img = image.copy()
    contours_count = len(contours)
    global green_color
    global std_width
    [cv2.drawContours(img, [contours[i]], 0, color, std_width)
     for i in range(contours_count)]
    return img


def show_contours(img, thresh, label):
    contours = get_contours(thresh)
    contours_count = len(contours)
    global green_color
    global std_width
    img = add_contours(img, contours, green_color, std_width)
    cv2.imshow(label, img)


def inside_point(point, poly):
    dist = cv2.pointPolygonTest(poly, (point[0], point[1]), True)
    if dist > 0:
        return True
    else:
        return False


def add_line(publish, is_left, image, l_point, l_angle, color, width):
    if l_point[0] == 0:
        l_point = (1, l_point[1])
    if l_point[1] == 0:
        l_point = (l_point[0], 1)

    # y = kx + b
    k = math.tan(l_angle)
    b = l_point[1] - (k*l_point[0])
    x1 = int((720 - b)/k)
    y1 = 720
    x2 = int((0 - b)/k)
    y2 = 0
    cv2.line(image, (x1, y1), (x2, y2), color, width)
    if publish:
        if is_left:
            global LEFT_K
            global LEFT_B
            LEFT_K = k
            LEFT_B = b
        else:
            global RIGHT_K
            global RIGHT_B
            RIGHT_K = k
            RIGHT_B = b


def is_protected_zone(is_left, point):

    global LEFT_K
    global RIGHT_K

    global LEFT_B
    global RIGHT_B

    k = (LEFT_K, RIGHT_K)[not is_left]
    b = (LEFT_B, RIGHT_B)[not is_left]
    if k is not None and b is not None and k != 0:
        xl = point[0]
        xr = (point[1]-b)/k

        if is_left and xl <= xr:
            return True
        if not is_left and xl >= xr:

            return True
    return False


def final(bin_for_borders_detect, diff_for_total_detect, image_for_printing, is_left, is_ready):
    fails_contours = get_contours(diff_for_total_detect)
    rectangles_fails = []
    for fail_cand in fails_contours:
        x, y, w, h = cv2.boundingRect(fail_cand)
        rectangles_fails.append((x, y, w, h))

    border_contours = get_contours(bin_for_borders_detect)
    rectangles_border = []
    for bord_cand in border_contours:
        x, y, w, h = cv2.boundingRect(bord_cand)
        rectangles_border.append((x, y, w, h))

    detected = []
    detected_in_blocked = []

    for c_i in range(len(fails_contours)):
        inside_border = False
        above_border = False

        R = rectangles_fails[c_i]
        (x, y, w, h) = (R[0], R[1], R[2], R[3])
        points_cand = [(x, y), (x+w, y),
                       (x, y+h), (x+w, y+h)]

        protected_check_point = (points_cand[0], points_cand[3])[not is_left]
        if is_protected_zone(is_left, protected_check_point):
            inside_border = True

        for b_i in range(len(border_contours)):
            if inside_border or above_border:
                break
            rect_border = rectangles_border[b_i]

            for i in range(4):
                P = points_cand[i]
                (X, Y) = (P[0], P[1])

                if inside_point(P, border_contours[b_i]):
                    inside_border = True
                    break
                inside = (X > rect_border[0]) and (
                    X < rect_border[0] + rect_border[2])
                above = Y < rect_border[1]
                if inside and above:
                    above_border = True
                    break
            if not inside_border and not above_border:
                detected.append([])
                detected[len(detected)-1] = fails_contours[c_i]
            else:
                detected_in_blocked.append([])
                detected_in_blocked[len(
                    detected_in_blocked)-1] = fails_contours[c_i]

    global red_color
    global yellow_color
    ans = []
    for i in range(len(detected)):
        contour = detected[i]
        ans.append(contour)

    ans2 = []
    for i in range(len(detected_in_blocked)):
        contour = detected_in_blocked[i]
        ans2.append(contour)

    global machine_state
    if is_ready:
        if is_left and len(ans) > 3:
            machine_state.detect_on_left()

        if not is_left and len(ans) > 3:
            machine_state.detect_on_right()

    image_for_printing = add_contours(image_for_printing, ans, red_color, 3)
    image_for_printing = add_contours(
        image_for_printing, ans2, yellow_color, 5)
    image_for_printing = add_contours(
        image_for_printing, border_contours, green_color, 1)

    if is_left:
        global LEFT_FILTER_X
        global LEFT_FILTER_Y
        global LEFT_ANGLE
        if LEFT_FILTER_X is not None:
            add_line(False, is_left,  image_for_printing,
                     (LEFT_FILTER_X, LEFT_FILTER_Y), LEFT_ANGLE, yellow_color, 1)
            add_line(True, is_left, image_for_printing, (LEFT_FILTER_X +
                                                         40, LEFT_FILTER_Y), LEFT_ANGLE, yellow_color, 1)
            add_line(False, is_left,  image_for_printing,
                     (LEFT_FILTER_X-90, LEFT_FILTER_Y), LEFT_ANGLE, yellow_color, 1)
    else:
        global RIGHT_FILTER_X
        global RIGHT_FILTER_Y
        global RIGHT_ANGLE
        if RIGHT_FILTER_X is not None:
            add_line(False,  is_left, image_for_printing, (RIGHT_FILTER_X,
                                                           RIGHT_FILTER_Y), RIGHT_ANGLE, yellow_color, 1)
            add_line(True, is_left, image_for_printing, (RIGHT_FILTER_X -
                                                         40, RIGHT_FILTER_Y), RIGHT_ANGLE, yellow_color, 1)
            add_line(False, is_left,  image_for_printing, (RIGHT_FILTER_X +
                                                           90, RIGHT_FILTER_Y), RIGHT_ANGLE, yellow_color, 1)

    label = ("L Main", "R Main")[not is_left]
    return image_for_printing


# In[3]:


def get_diff_image(images):
    # im_t is the frame of interest; im_tp1 and im_tm1 are, respectively
    # the successive and previous frames.
    dbp = cv2.absdiff(images[1], images[0])
    db0 = cv2.absdiff(images[2], images[0])
    dbm = cv2.absdiff(images[1], images[2])
    return cv2.bitwise_not(cv2.bitwise_and(cv2.bitwise_and(dbp, dbm), cv2.bitwise_not(db0)))


def plot_average_contours(image, contour_source):
    show_contours(image, contour_source, "Contour")


def process_average(frame, draw_image, is_left):
    global average_frame_limit
    global average_images_limit
    global average_images_l
    global average_images_r
    global index_for_averaging

    average_images = (average_images_r, average_images_l)[is_left]
    ret = None

    average_images.append(frame)

    if len(average_images) == average_images_limit:
        dst = average_images[0]
        for i in range(len(average_images)):
            if i == 0:
                pass
            else:
                alpha = 1.0/(i + 1)
                beta = 1.0 - alpha
                dst = cv2.addWeighted(average_images[i], alpha, dst, beta, 0.0)
        ret, binarizedImage = cv2.threshold(dst, 70, 255, cv2.THRESH_TOZERO)

        main_contours = get_contours(binarizedImage)
        LX = 1280
        LY = 0

        RX = 0
        LY = 0

        for cnt in main_contours:
            for pnt in cnt:
                if is_left and LX > pnt[0][0]:
                    LX = pnt[0][0]
                    LY = pnt[0][1]
                if not is_left and RX < pnt[0][0]:
                    RX = pnt[0][0]
                    RY = pnt[0][1]

        if is_left and LX != 1280:
            global LEFT_FILTER_X
            LEFT_FILTER_X = LX
            global LEFT_FILTER_Y
            LEFT_FILTER_Y = LY
        if not is_left and RX != 0:
            global RIGHT_FILTER_X
            RIGHT_FILTER_X = RX
            global RIGHT_FILTER_Y
            RIGHT_FILTER_Y = RY

        kernel = np.ones((8, 8), np.uint8)
        erosion = cv2.dilate(binarizedImage, kernel, iterations=15)
        ret = erosion
    if len(average_images) >= average_images_limit:
        average_images.pop(0)
    index_for_averaging = index_for_averaging + 1
    if index_for_averaging > average_frame_limit:
        index_for_averaging = 0
    return ret


def process_frame(frame, images, is_ready, is_left):
    # Frames #
    gr_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, binarizedImage = cv2.threshold(gr_img, 215, 255, cv2.THRESH_TOZERO)
    color_img = cv2.cvtColor(binarizedImage, cv2.COLOR_GRAY2RGB)

    av_img = process_average(binarizedImage, color_img, is_left)

    images.append(gr_img)
    if len(images) > 3:
        images.pop(0)

    if len(images) == 3:
        # Frames
        diff_image = get_diff_image(images)
        r, binarized_diff = cv2.threshold(
            diff_image, 215, 255, cv2.THRESH_TOZERO_INV)
        # cv2.imshow('Binarized Diff',binarized_diff)
        # label = ("L Color","R Color")[not is_left]
        # show_contours(frame,binarized_diff,label)
        binarizedImage = binarized_diff
        if av_img is not None:
            return final(av_img, binarized_diff, color_img.copy(), is_left, is_ready)

    return None


# In[4]:


ip_addr = 'localhost'
port = 2002
screenFlipper = False
msg = '!{0}"{1}"{2}"{3}"{4}"{5}"{6}"{7}"'

message_ready = False
message_text = ""


def CreateMessage(state):

    global message_ready
    global message_text

    msg_copy = msg
    errorFound = int(state.working)
    isWorking = int(state.stable)
    flipping = int(state.blink)

    state.blink = not state.blink
    detected = int(state.detected)
    state.detected = False
    l_count = str(state.l_count).zfill(3)
    l_working = int(state.l_enabled)
    r_count = str(state.r_count).zfill(3)
    r_working = int(state.r_enabled)

    state.r_detected = int(False)
    state.l_detected = int(False)

    possibility = str(random.randint(0, 100)).zfill(3)
    size = str(random.randint(0, 999)).zfill(3)
    msg_copy = msg_copy.format(
        errorFound, isWorking, flipping, detected, l_count, l_working, r_count, r_working)

    message_text = msg_copy
    message_ready = True

    return msg_copy.encode()


def CreateRandomMessage():
    global screenFlipper
    global msg
    # server

    msg_copy = msg
    errorFound = random.randint(0, 1)
    isWorking = random.randint(0, 1)
    flipping = (0, 1)[screenFlipper]
    screenFlipper = not screenFlipper
    possibility = str(random.randint(0, 100)).zfill(3)
    size = str(random.randint(0, 999)).zfill(3)
    msg_copy = msg_copy.format(
        errorFound, isWorking, flipping, errorFound, possibility, size)
    return msg_copy.encode()


async def handle_client(reader, writer):
    print("New Client")
    global message_ready
    global message_text

    # return
    while True:
        try:
            if message_ready:
                writer.write(message_text.encode('utf8'))
                await writer.drain()
                message_ready = False
            time.sleep(0.1)
        except Exception as e:
            print(e)
            return


def loop_in_thread(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()


def process_video_file(lvideo, rvideo):
    while(True):
        try:
            global machine_state
            CreateMessage(machine_state)
            # start server

            loop = asyncio.get_event_loop()
            loop.create_task(asyncio.start_server(
                handle_client, ip_addr, port))

            thread = threading.Thread(target=loop_in_thread, args=(loop,))
            thread.start()

            ###

            start = timer()

            lcap = cv2.VideoCapture(lvideo)
            rcap = cv2.VideoCapture(rvideo)
            lcap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            lcap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            rcap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            rcap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            if not lcap.isOpened():
                machine_state.working = False
                machine_state.l_enabled = False

            if not rcap.isOpened():
                machine_state.working = False
                machine_state.l_enabled = False

            l_images = []
            r_images = []

            counter = 0
            while(lcap.isOpened() and rcap.isOpened()):
                CreateMessage(machine_state)
                counter = counter + 1
                if counter > 5:
                    counter = 0
                    # SocketHandle(sock)
                l_ret, l_frame = lcap.read()
                r_ret, r_frame = rcap.read()
                if l_ret == False or r_ret == False:
                    machine_state.l_enabled = l_ret
                    machine_state.r_enabled = r_ret
                    machine_state.working = False
                    machine_state.stable = False
                is_ready = timer() - start > turn_on_delay
                machine_state.stable = is_ready

                image_right = process_frame(
                    r_frame, r_images, is_ready, False)
                image_left = process_frame(
                    l_frame, l_images, is_ready, True)

                is_showing = True
                if is_showing:
                    if image_left is not None and image_right is not None:
                        stack = np.hstack((image_left, image_right))
                        cv2.imshow("LR", stack)
                    elif image_left is not None:
                        cv2.imshow("L", image_left)
                    elif image_right is not None:
                        cv2.imshow("R", image_right)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print('exit')
                    # When everything done, release the video capture object
                    lcap.release()
                    rcap.release()
                    cv2.destroyAllWindows()
                    sys.exit()
                    return

            lcap.release()
            rcap.release()
            cv2.destroyAllWindows()
            # sock.close()
            clear_output(wait=True)
            print("Server stopped due to disconnect, restarting")
        except Exception as e:

            time.sleep(1)
            print(e)
            pass


process_video_file(bad_video, right_video)
