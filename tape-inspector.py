#!/usr/bin/env python
# coding: utf-8

from statistics import mean
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

ip_addr = '192.168.0.20'
port = 2002
left_cam_index = 1
right_cam_index = 0
contours_count_limit = 3
plotting = True

conn = None
addr = None

# data
cwd = os.getcwd()
examples_dir = cwd + "//video//"
bad_video = examples_dir + "N1.wmv"
bad_video = examples_dir + "bad_new0.wmv"
good_video = examples_dir + "good.wmv"
right_video = examples_dir + "rgood.wmv"


ip_addr = '192.168.0.2'
left_cam_index = bad_video
right_cam_index = right_video
# plotting = False

# settings
turn_on_delay = 5


# Average for lamps detector
average_frame_limit = 8
average_images_limit = 40
average_images_l = []
average_images_r = []
index_for_averaging = 0

green_color = (0, 255, 0)
red_color = (0, 0, 255)
yellow_color = (0, 255, 255)
purple_color = (255, 0, 255)
orange_color = (0, 165, 255)
blue_color = (255, 0, 0)
std_width = 1


# VARS
LEFT_FILTER_X = None
LEFT_FILTER_Y = None

LEFT_ANGLE = 1.87
RIGHT_ANGLE = -1.87

LEFT_K = None
LEFT_B = None

RIGHT_FILTER_X = None
RIGHT_FILTER_Y = None

RIGHT_K = None
RIGHT_B = None


LFactor = []
RFactor = []
MeanL = 3
MinSize = 10


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
        self.activated_count_l = 0
        self.activated_count_r = 0
        self.activated_max = 3

    def lost_detect(self, is_left):
        if is_left:
            self.activated_count_l = 0
        else:
            self.activated_count_r = 0

    def detect_on_left(self):
        self.activated_count_l = self.activated_count_l + 1
        if self.activated_count_l < self.activated_max:
            return
        self.activated_count_l = 0
        if self.detected:
            return
        self.detected = True
        self.l_count = self.l_count + 1
        if self.l_count > 999:
            self.l_count = 0

    def detect_on_right(self):
        self.activated_count_r = self.activated_count_r + 1
        if self.activated_count_r < self.activated_max:
            return
        self.activated_count_r = 0
        if self.detected:
            return
        self.detected = True
        self.r_count = self.r_count + 1
        if self.r_count > 999:
            self.r_count = 0


machine_state = State()


def get_contours(thresh, reverse_sorted):
    # Find contours
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=reverse_sorted)
    return contours


def add_contours(image, contours, color, width):
    img = image.copy()
    contours_count = len(contours)
    global green_color
    global std_width
    [cv2.drawContours(img, [contours[i]], 0, color, width)
     for i in range(contours_count)]
    return img


def inside_point(point, poly):
    dist = cv2.pointPolygonTest(poly, (point[0], point[1]), True)
    if dist > 0:
        return True
    else:
        return False


def get_k_and_b(l_point, l_angle):
    if l_point[0] == 0:
        l_point = (1, l_point[1])
    if l_point[1] == 0:
        l_point = (l_point[0], 1)

    # y = kx + b
    k = math.tan(l_angle)
    b = l_point[1] - (k*l_point[0])
    return k, b


def publish_line_params(is_left, l_point, l_angle):
    k, b = get_k_and_b(l_point, l_angle)

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


def add_line(image, l_point, l_angle, color, width):
    k, b = get_k_and_b(l_point, l_angle)
    x1, y1 = int((720 - b)/k), 720
    x2, y2 = int((0 - b)/k), 0
    cv2.line(image, (x1, y1), (x2, y2), color, width)


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


def split_points_special(fails_contours, rectangles_fails, border_contours, rectangles_border, is_left):

    global contours_count_limit
    cutoff = 5*contours_count_limit

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

            if inside_border and not above_border:
                detected.append([])
                detected[len(detected)-1] = fails_contours[c_i]

                if len(detected) > cutoff:
                    break
            else:
                detected_in_blocked.append([])
                detected_in_blocked[len(
                    detected_in_blocked)-1] = fails_contours[c_i]

    return detected, detected_in_blocked


def split_points(fails_contours, rectangles_fails, border_contours, rectangles_border, is_left):

    global contours_count_limit
    detected = []
    detected_in_blocked = []

    for c_i in range(len(fails_contours)):
        inside_border = False
        above_border = False
        on_tape = False

        R = rectangles_fails[c_i]
        (x, y, w, h) = (R[0], R[1], R[2], R[3])
        points_cand = [(x, y), (x+w, y),
                       (x, y+h), (x+w, y+h)]

        protected_check_point = (points_cand[0], points_cand[3])[not is_left]
        if is_protected_zone(is_left, protected_check_point):
            on_tape = True

        for b_i in range(len(border_contours)):

            if len(detected) > 5*contours_count_limit:
                break

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

            if not inside_border and not above_border and not on_tape:
                detected.append([])
                detected[len(detected)-1] = fails_contours[c_i]
            else:
                detected_in_blocked.append([])
                detected_in_blocked[len(
                    detected_in_blocked)-1] = fails_contours[c_i]

    return detected, detected_in_blocked


def add_filter_lines(is_left, image_for_printing):

    global blue_color
    global orange_color

    global LEFT_FILTER_X
    global LEFT_FILTER_Y
    global LEFT_ANGLE

    global RIGHT_FILTER_X
    global RIGHT_FILTER_Y
    global RIGHT_ANGLE

    line_filter_x = (LEFT_FILTER_X, RIGHT_FILTER_X)[not is_left]
    line_filter_y = (LEFT_FILTER_Y, RIGHT_FILTER_Y)[not is_left]
    line_ang = (LEFT_ANGLE, RIGHT_ANGLE)[not is_left]

    diff_1 = (40, -40)[not is_left]
    diff_2 = (-90, -90)[not is_left]

    if line_filter_x is None:
        return image_for_printing

    add_line(image_for_printing, (line_filter_x + diff_1,
                                  line_filter_y), line_ang, orange_color, 3)
    add_line(image_for_printing, (line_filter_x + diff_2,
                                  line_filter_y), line_ang, blue_color, 1)
    add_line(image_for_printing, (line_filter_x,
                                  line_filter_y), line_ang, blue_color, 1)

    return image_for_printing


# bin_for_borders_detect - average
# diff_for_total_detect - binarized

arr_len3 = []
arr_len10 = []


def make_decision_and_report_special(image, border_contours, is_left, is_ready, image_for_printing):
    artifacts = get_contours(image, False)
    detected = []
    detected_in_blocked = []

    rectangles_border = []
    for bord_cand in border_contours:
        x, y, w, h = cv2.boundingRect(bord_cand)
        rectangles_border.append((x, y, w, h))

    rectangles_fails = []
    for fail_cand in artifacts:
        x, y, w, h = cv2.boundingRect(fail_cand)
        rectangles_fails.append((x, y, w, h))

    detected, detected_in_blocked = split_points_special(
        artifacts, rectangles_fails, border_contours, rectangles_border, is_left)

    ans = []
    for i in range(len(detected)):
        contour = detected[i]
        ans.append(contour)

    global machine_state
    global contours_count_limit
    pre_res = len(ans) > contours_count_limit
    if not pre_res:
        machine_state.lost_detect(is_left)
    if is_left and pre_res:
        machine_state.detect_on_left()
    if not is_left and pre_res:
        machine_state.detect_on_right()

    global LFactor
    global RFactor
    global MeanL
    global MinSize

    amount = 0.00
    for cnt in detected:
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        if 2*radius > MinSize:
            amount = amount + 1

    if is_left:
        if len(LFactor) >= MeanL:
            LFactor.pop(0)
        LFactor.append(amount)
    else:
        if len(RFactor) >= MeanL:
            RFactor.pop(0)
        RFactor.append(amount)

    global green_color
    global red_color

    avg = 0.000001
    if is_left:
        avg = mean(LFactor)
    else:
        avg = mean(RFactor)

    label = "OK,   "+"{:.3f}".format(avg)
    color = green_color
    if avg > 3:
        color = red_color
        label = "BAD,  "+"{:.3f}".format(avg)

    cv2.circle(image_for_printing, (700, 500), MinSize, color, -1)
    cv2.putText(image_for_printing, label,
                (700, 550), cv2.FONT_HERSHEY_PLAIN, 1.0, color, 2)

    return image_for_printing


def special_final(bin_for_borders_detect, binarized, image_for_printing, is_left, is_ready):

    global purple_color
    global orange_color
    border_contours = get_contours(bin_for_borders_detect, True)
    image = add_contours(image_for_printing, border_contours, purple_color, 5)
    image = add_filter_lines(is_left, image)
    image = make_decision_and_report_special(
        binarized, border_contours, is_left, is_ready, image)
    return image


def final(bin_for_borders_detect, diff_for_total_detect, image_for_printing, is_left, is_ready):

    fails_contours = get_contours(diff_for_total_detect, False)
    rectangles_fails = []
    for fail_cand in fails_contours:
        x, y, w, h = cv2.boundingRect(fail_cand)
        rectangles_fails.append((x, y, w, h))

    border_contours = get_contours(bin_for_borders_detect, True)
    rectangles_border = []
    for bord_cand in border_contours:
        x, y, w, h = cv2.boundingRect(bord_cand)
        rectangles_border.append((x, y, w, h))

    detected = []
    detected_in_blocked = []

    detected, detected_in_blocked = split_points(
        fails_contours, rectangles_fails, border_contours, rectangles_border, is_left)

    ans = []
    for i in range(len(detected)):
        contour = detected[i]
        ans.append(contour)

    ans2 = []
    for i in range(len(detected_in_blocked)):
        contour = detected_in_blocked[i]
        ans2.append(contour)

    if is_ready:
        make_decision_and_report(ans, is_left)

    return calculate_image_for_printing(is_left, image_for_printing, border_contours, ans, ans2)


def calculate_image_for_printing(is_left, image_for_printing, border_contours, ans, ans2):

    global red_color
    global yellow_color

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
            add_line(image_for_printing,
                     (LEFT_FILTER_X, LEFT_FILTER_Y), LEFT_ANGLE, yellow_color, 1)
            add_line(image_for_printing, (LEFT_FILTER_X +
                                          40, LEFT_FILTER_Y), LEFT_ANGLE, yellow_color, 1)

            publish_line_params(
                is_left, (LEFT_FILTER_X + 40, LEFT_FILTER_Y), LEFT_ANGLE)
            add_line(image_for_printing,
                     (LEFT_FILTER_X-90, LEFT_FILTER_Y), LEFT_ANGLE, yellow_color, 1)
    else:
        global RIGHT_FILTER_X
        global RIGHT_FILTER_Y
        global RIGHT_ANGLE
        if RIGHT_FILTER_X is not None:
            add_line(image_for_printing, (RIGHT_FILTER_X,
                                          RIGHT_FILTER_Y), RIGHT_ANGLE, yellow_color, 1)
            add_line(image_for_printing, (RIGHT_FILTER_X -
                                          40, RIGHT_FILTER_Y), RIGHT_ANGLE, yellow_color, 1)

            publish_line_params(
                is_left, (RIGHT_FILTER_X - 40, RIGHT_FILTER_Y), RIGHT_ANGLE)
            add_line(image_for_printing, (RIGHT_FILTER_X +
                                          90, RIGHT_FILTER_Y), RIGHT_ANGLE, yellow_color, 1)

    return image_for_printing


def make_decision_and_report(ans, is_left):
    global machine_state
    global contours_count_limit
    pre_res = len(ans) > contours_count_limit
    if not pre_res:
        machine_state.lost_detect(is_left)
    if is_left and pre_res:
        machine_state.detect_on_left()
    if not is_left and pre_res:
        machine_state.detect_on_right()


def get_diff_image(images):
    # im_t is the frame of interest; im_tp1 and im_tm1 are, respectively
    # the successive and previous frames.
    dbp = cv2.absdiff(images[1], images[0])
    db0 = cv2.absdiff(images[2], images[0])
    dbm = cv2.absdiff(images[1], images[2])
    return cv2.bitwise_not(cv2.bitwise_and(cv2.bitwise_and(dbp, dbm), cv2.bitwise_not(db0)))


def find_incline(main_contours, is_left):
    LX, LY = 1280, 720
    RX, RY = 0, 0

    if is_left:
        for cnt in main_contours:
            for pnt in cnt:
                if LX > pnt[0][0]:
                    LX = pnt[0][0]
                    LY = pnt[0][1]
                elif LX == pnt[0][0]:
                    if pnt[0][1] < LY:
                        LX = pnt[0][0]
                        LY = pnt[0][1]

    if not is_left:
        for cnt in main_contours:
            for pnt in cnt:
                if RX < pnt[0][0]:
                    RX = pnt[0][0]
                    RY = pnt[0][1]
                elif RX == pnt[0][0] and pnt[0][1] < RY:
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

        main_contours = get_contours(binarizedImage, True)

        find_incline(main_contours, is_left)

        kernel = np.ones((8, 8), np.uint8)
        erosion = cv2.dilate(binarizedImage, kernel, iterations=18)
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
        if av_img is not None:
            return special_final(av_img, binarizedImage,
                                 color_img.copy(), is_left, is_ready)

    return None


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

    detected = int(state.detected)
    if not message_ready:
        state.blink = not state.blink
        message_ready = True
    flipping = int(state.blink)
    l_count = str(state.l_count).zfill(3)
    l_working = int(state.l_enabled)
    r_count = str(state.r_count).zfill(3)
    r_working = int(state.r_enabled)

    state.r_detected = int(False)
    state.l_detected = int(False)
    msg_copy = msg_copy.format(
        errorFound, isWorking, flipping, detected, l_count, l_working, r_count, r_working)

    message_text = msg_copy

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


def SocketCycle(conn):
    global message_max_length
    # data = conn.recv(message_max_length)
    global machine_state
    global message_ready
    conn.send(CreateMessage(machine_state))
    machine_state.detected = False
    message_ready = False


def SocketHandle(sock):
    global port
    global conn
    global addr
    if conn is not None and addr is not None:
        try:
            SocketCycle(conn)
        except:
            conn = None
            addr = None
            pass
    else:
        try:
            print('wait')
            sock.listen(1)
            x, y = sock.accept()
            sock.settimeout(0.4)
            print('Server started')
            conn = x
            addr = y
        except:
            conn = None
            addr = None
            pass


def get_captures(lvideo, rvideo, machine_state):
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

    return lcap, rcap


def get_frames(lcap, rcap, machine_state):
    l_ret, l_frame = lcap.read()
    r_ret, r_frame = rcap.read()
    if l_ret == False or r_ret == False:
        machine_state.l_enabled = l_ret
        machine_state.r_enabled = r_ret
        machine_state.working = False
        machine_state.stable = False
    return l_frame, r_frame


def show_image(image_left, image_right):
    if image_left is not None and image_right is not None:
        stack = np.hstack((image_left, image_right))
        # suits for image containing any amount of channels
        w, h = stack.shape[:2]
        #resize_factor = 1.6
        resize_factor = 0.9
        w = int(w / resize_factor)  # one must compute beforehand
        h = int(h / resize_factor)  # and convert to INT
        # use variables defined/computed BEFOREHAND
        stack = cv2.resize(stack, (h, w))
        cv2.imshow("LR", stack)


def release_resources(lcap, rcap):
    lcap.release()
    rcap.release()
    sock.close()
    cv2.destroyAllWindows()


def process_video_file(lvideo, rvideo):
    while(True):
        try:
            global machine_state
            # Prepare for first report
            CreateMessage(machine_state)

            sock = socket.socket()
            sock.bind((ip_addr, port))

            lcap, rcap = get_captures(lvideo, rvideo, machine_state)
            focus_delay_timer = timer()
            l_images, r_images = [], []
            counter = 0
            while(lcap.isOpened() and rcap.isOpened()):
                CreateMessage(machine_state)
                SocketHandle(sock)

                l_frame, r_frame = get_frames(lcap, rcap, machine_state)

                if l_frame is None or r_frame is None:
                    print("Some is none")

                is_ready = timer() - focus_delay_timer > turn_on_delay
                machine_state.stable = is_ready

                image_right = process_frame(
                    r_frame, r_images, is_ready, False)
                image_left = process_frame(
                    l_frame, l_images, is_ready, True)

                global plotting
                if plotting:
                    show_image(image_left, image_right)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print('exit')
                    release_resources(lcap, rcap)
                    sys.exit()
                    return

            release_resources(lcap, rcap)
            clear_output(wait=True)
            print("Server restarting")
        except Exception as e:

            time.sleep(1)
            print(e)
            pass


process_video_file(left_cam_index, right_cam_index)
