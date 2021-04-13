import gym
import vision_arena
import time
import pybullet as p
import pybullet_data

import os
from collections import defaultdict

import numpy as np
import cv2
import cv2.aruco as aruco
import sys
import math
                                                #cropping and reshaping input as per need
env = gym.make("vision_arena-v0")

env.remove_car()
i = env.camera_feed()

print("****** SELECT ROI ******")

R=cv2.selectROI('select',i)
im=i[int(R[1]):int(R[1]+R[3]),int(R[0]):int(R[0]+R[2])]
         # conversion of image to grayscale
im1 = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

         # Noise reduction step
im2 = cv2.GaussianBlur(im1, (1, 1), 0.5)
hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)                          # red
contours, _ = cv2.findContours(im2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
blank=np.zeros_like(im)
for cnt in contours:
    epsilon = 0.01 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    area = cv2.contourArea(approx)
    per = cv2.arcLength(approx, True)
    rang = (per * per) / area
    if (area > 100000):
            cv2.drawContours(blank, [approx], -1, (0, 255, 0), 1)
            x, y, w, h = cv2.boundingRect(approx)
im4=im[int(y):int(y+w),int(x):int(x+h)]
img=cv2.resize(im4,(540,540))
                                            #matrix to be filled
array = np.zeros([9,9])

                                           #shape , color detection and matrix filling starts from here
def push(n,maskD):
    contours, _ = cv2.findContours(maskD, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        array[int(y / 60), int(x / 60)] = n
    # print(array)

def strips(inc,maskF):
    contours, _ = cv2.findContours(maskF, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        array[int(y / 60), int(x / 60)] = array[int(y / 60), int(x / 60)] + inc


def circle(maskfinal):
    contours, _ = cv2.findContours(maskfinal, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cir = np.zeros_like(maskfinal)
    for cnt in contours:
        epsilon = 0.009 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        area = cv2.contourArea(approx)
        per= cv2.arcLength(approx,True)
        rang = (per * per)/area
        if cv2.contourArea(approx) > 100:
            if ((np.shape(approx)[0] > 5) and (10<=rang<=14)):
                cv2.drawContours(cir, [approx], -1, (255, 255, 255), -1)

    return cir


def square(maskfinal):
    contours, _ = cv2.findContours(maskfinal, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    sqr = np.zeros_like(maskfinal)
    for cnt in contours:
        epsilon = 0.008 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        area = cv2.contourArea(approx)
        per= cv2.arcLength(approx,True)
        rang = (per * per)/area
        if (area> 100):
            if (15.5<=rang<=16.8):
                cv2.drawContours(sqr,[approx],-1,(255,255,255),-1)
    return sqr



def triangle(maskfinal):
    contours, _ = cv2.findContours(maskfinal, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    tri = np.zeros_like(maskfinal)
    for cnt in contours:
        epsilon = 0.06 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if (cv2.contourArea(approx) > 100):
            if np.shape(approx)[0] == 3:
                cv2.drawContours(tri, [approx], -1, (255, 255, 255), -1)
    return tri

def transitionstrips(mask):
    shape  = cv2.bitwise_and(shapemask,shapemask,mask = mask)
    return  shape

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)                          # red
upper = np.array([10, 255, 255])
lower = np.array([0, 120, 70])

mask1 = cv2.inRange(hsv, lower, upper)

upper = np.array([180, 255, 255])
lower = np.array([170, 120, 70])

mask2 = cv2.inRange(hsv, lower, upper)

kernel = np.ones((2, 2), np.uint8)

mask_R = mask1 + mask2

closingr = cv2.morphologyEx(mask_R, cv2.MORPH_OPEN, kernel)
mask_R = cv2.morphologyEx(closingr, cv2.MORPH_DILATE, kernel)

uppery = np.array([35, 255, 255])                                   # yellow
lowery = np.array([20, 200, 200])

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
kernely = np.ones((2, 2), np.uint8)
mask_y = cv2.inRange(hsv, lowery, uppery)
mask_y = cv2.morphologyEx(mask_y, cv2.MORPH_CLOSE, kernely)
                                                                 # All shapes
finalmask = mask_y + mask_R
kerneld = np.ones((3, 3), np.uint8)
rawmask = cv2.dilate(finalmask, kerneld, iterations=1)
tri = triangle(rawmask)
sqr = square(rawmask)
cir = circle(rawmask)
shapemask = tri + sqr + cir
                                                                  # Cyan
upper = np.array([100, 255, 255])
lower = np.array([80, 200, 200])

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
kernel = np.ones((40, 40), np.uint8)
mask_C = cv2.inRange(hsv, lower, upper)
mask_C = cv2.morphologyEx(mask_C, cv2.MORPH_CLOSE, kernel)
                                                                 # Green
upper = np.array([80, 255, 255])
lower = np.array([50, 200, 200])

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
kernel = np.ones((40, 40), np.uint8)
mask_G = cv2.inRange(hsv, lower, upper)
mask_G = cv2.morphologyEx(mask_G, cv2.MORPH_CLOSE, kernel)
                                                                 # Pink
upper = np.array([170, 255, 255])
lower = np.array([140, 200, 200])

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
kernel = np.ones((40, 40), np.uint8)
mask_P = cv2.inRange(hsv, lower, upper)
mask_P = cv2.morphologyEx(mask_P, cv2.MORPH_CLOSE, kernel)

                                                                 # Blue
upper = np.array([140, 255, 255])
lower = np.array([100, 200, 200])

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
kernel = np.ones((40, 40), np.uint8)
mask_B = cv2.inRange(hsv, lower, upper)
mask_B = cv2.morphologyEx(mask_B, cv2.MORPH_CLOSE, kernel)

                                                  #transition strip && all the shapes with their respective color
shapes_B = transitionstrips(mask_B)
shapes_C = transitionstrips(mask_C)
shapes_G = transitionstrips(mask_G)
shapes_P = transitionstrips(mask_P)
Y_T = cv2.bitwise_and(mask_y,tri)
Y_S = cv2.bitwise_and(mask_y,sqr)
Y_C = cv2.bitwise_and(mask_y,cir)
R_T = cv2.bitwise_and(mask_R,tri)
R_S = cv2.bitwise_and(mask_R,sqr)
R_C = cv2.bitwise_and(mask_R,cir)


                                                      #pushing values into array
push(1,Y_T)
push(2,Y_S)
push(3,Y_C)
push(4,R_T)
push(5,R_S)
push(6,R_C)
strips(10,shapes_B)
strips(20,shapes_C)
strips(30,shapes_G)
strips(40,shapes_P)
                                                    #pushing value 100 at
array[4,0]=100
array[0,4]=100
array[8,4]=100
array[4,8]=100
array[4,4]=999
array = array.astype(np.int32)

env.respawn_car()

print(array)



#################################### END OF IMAGE PROCESSING ##############################################

def detect_Aruco(img):  # returns the detected aruco list dictionary with id: corners
    aruco_list = {}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(
        aruco.DICT_ARUCO_ORIGINAL)  # creating aruco_dict with 5x5 bits with max 250 ids..so ids ranges from 0-249
    parameters = aruco.DetectorParameters_create()  # refer opencv page for clarification
    # lists of ids and the corners beloning to each id
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    aruco.drawDetectedMarkers(img, corners, borderColor=(0, 0, 255))
    # corners is the list of corners(numpy array) of the detected markers. For each marker, its four corners are returned in their original order (which is clockwise starting with top left). So, the first corner is the top left corner, followed by the top right, bottom right and bottom left.
    # print corners[0]
    # gray = aruco.drawDetectedMarkers(gray, corners,ids)
    # cv2.imshow('frame',gray)
    # print (type(corners[0]))
    if len(corners):  # returns no of arucos
        # print (len(corners))
        # print (len(ids))
        # print(type(corners))
        # print(corners[0][0])
        for k in range(len(corners)):
            temp_1 = corners[k]
            temp_1 = temp_1[0]
            temp_2 = ids[k]
            temp_2 = temp_2[0]
            aruco_list[temp_2] = temp_1
        return aruco_list

def angle_calculate(pt1, pt2, trigger=0):  # function which returns angle between two points in the range of 0-359

    x = pt2[0] - pt1[0]  # unpacking tuple
    y = pt2[1] - pt1[1]
    angle = int(math.degrees(
        math.atan2(y, x)))  # takes 2 points nad give angle with respect to horizontal axis in range(-180,180)

    return int(angle)


def calculate_Robot_State(aruco_list):  # gives the state of the bot (centre(x), centre(y), angle)
    robot_state = []
    key_list=[]
    if aruco_list is not None:
        key_list = aruco_list.keys()

    for key in key_list:
        dict_entry = aruco_list[key]
        pt1, pt2 = tuple(dict_entry[0]), tuple(dict_entry[1])
        centre = dict_entry[0] + dict_entry[1] + dict_entry[2] + dict_entry[3]
        centre[:] = [int(x / 4) for x in centre]
        centre = tuple(centre)
        #print(centre)
        angle = angle_calculate(pt1, pt2)

        robot_state = [centre[0], centre[1],angle]
    #print (robot_state)

    return robot_state


def robopos(img, aruco_list):
    h = img.shape[0]
    cellwidth = h / 9
    centre = calculate_Robot_State(aruco_list)
    row = 0
    column = 0
    if (cellwidth / 2) - 15 < centre[0] < (cellwidth / 2) + 15:
        column = int(int(centre[0]) / cellwidth)
    elif 3 * (cellwidth / 2) - 15 < centre[0] < 3 * (cellwidth / 2) + 15:
        column = int(int(centre[0]) / cellwidth)
    elif 5 * (cellwidth / 2) - 15 < centre[0] < 5 * (cellwidth / 2) + 15:
        column = int(int(centre[0]) / cellwidth)
    elif 7 * (cellwidth / 2) - 15 < centre[0] < 7 * (cellwidth / 2) + 15:
        column = int(int(centre[0]) / cellwidth)
    elif 9 * (cellwidth / 2) - 15 < centre[0] < 9 * (cellwidth / 2) + 15:
        column = int(int(centre[0]) / cellwidth)
    elif 11 * (cellwidth / 2) - 15 < centre[0] < 11 * (cellwidth / 2) + 15:
        column = int(int(centre[0]) / cellwidth)
    elif 13 * (cellwidth / 2) - 15 < centre[0] < 13 * (cellwidth / 2) + 15:
        column = int(int(centre[0]) / cellwidth)
    elif 15 * (cellwidth / 2) - 15 < centre[0] < 15 * (cellwidth / 2) + 15:
        column = int(int(centre[0]) / cellwidth)
    elif 17 * (cellwidth / 2) - 15 < centre[0] < 17 * (cellwidth / 2) + 15:
        column = int(int(centre[0]) / cellwidth)

    if (cellwidth / 2) - 15 < centre[1] < (cellwidth / 2) + 15:
        row = int(int(centre[1]) / cellwidth)
    elif 3 * (cellwidth / 2) - 15 < centre[1] < 3 * (cellwidth / 2) + 15:
        row = int(int(centre[1]) / cellwidth)
    elif 5 * (cellwidth / 2) - 15 < centre[1] < 5 * (cellwidth / 2) + 15:
        row = int(int(centre[1]) / cellwidth)
    elif 7 * (cellwidth / 2) - 15 < centre[1] < 7 * (cellwidth / 2) + 15:
        row = int(int(centre[1]) / cellwidth)
    elif 9 * (cellwidth / 2) - 15 < centre[1] < 9 * (cellwidth / 2) + 15:
        row = int(int(centre[1]) / cellwidth)
    elif 11 * (cellwidth / 2) - 15 < centre[1] < 11 * (cellwidth / 2) + 15:
        row = int(int(centre[1]) / cellwidth)
    elif 13 * (cellwidth / 2) - 15 < centre[1] < 13 * (cellwidth / 2) + 15:
        row = int(int(centre[1]) / cellwidth)
    elif 15 * (cellwidth / 2) - 15 < centre[1] < 15 * (cellwidth / 2) + 15:
        row = int(int(centre[1]) / cellwidth)
    elif 17 * (cellwidth / 2) - 15 < centre[1] < 17 * (cellwidth / 2) + 15:
        row = int(int(centre[1]) / cellwidth)

    return [row, column]




n=9
l = int(n / 2)
arr=np.arange(1,n*n+1).reshape(n,n)
for i in range(1,9-1):
    if i!=4:
        arr[1,i]=0
        arr[i,1]=0
        arr[n-2,i]=0
        arr[i,n-2]=0
arr[3,3]=0
arr[5,5]=0
arr[3,5]=0
arr[5,3]=0
arr[4,4]=0


def make_adjacency(arr,n,start_pos_x,start_pos_y):
    arr = np.pad(arr, pad_width=1, mode='constant', constant_values=0)
    start_pos_x=start_pos_x+1
    start_pos_y=start_pos_y+1

    k = int(n / 2 + 1)


    adjacency = np.zeros((n * n + 1, n * n + 1), dtype=int)



    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if arr[i, j] != 0:
                if (i < k and j < k):
                    if arr[i, j + 1] != 0:
                        adjacency[arr[i, j], arr[i, j + 1]] = 1
                    if arr[i - 1, j] != 0:
                        adjacency[arr[i, j], arr[i - 1, j]] = 1
                elif (i < k and j > k):
                    if arr[i, j + 1] != 0:
                        adjacency[arr[i, j], arr[i, j + 1]] = 1
                    if arr[i + 1, j] != 0:
                        adjacency[arr[i, j], arr[i + 1, j]] = 1
                elif (i > k and j < k):
                    if arr[i, j - 1] != 0:
                        adjacency[arr[i, j], arr[i, j - 1]] = 1
                    if arr[i - 1, j] != 0:
                        adjacency[arr[i, j], arr[i - 1, j]] = 1
                elif (i > k and j > k):
                    if arr[i, j - 1] != 0:
                        adjacency[arr[i, j], arr[i, j - 1]] = 1
                    if arr[i + 1, j] != 0:
                        adjacency[arr[i, j], arr[i + 1, j]] = 1
                elif (i == k and j < k - 1):

                    if arr[i, j - 1] != 0:
                        adjacency[arr[i, j], arr[i, j - 1]] = 1
                        adjacency[arr[i, j - 1], arr[i, j]] = 1
                elif (i == k and j > k + 1):

                    if arr[i, j + 1] != 0:
                        adjacency[arr[i, j], arr[i, j + 1]] = 1
                        adjacency[arr[i, j + 1], arr[i, j]] = 1
                elif (i < k - 1 and j == k):

                    if arr[i - 1, j] != 0:
                        adjacency[arr[i, j], arr[i - 1, j]] = 1
                        adjacency[arr[i - 1, j], arr[i, j]] = 1
                elif (i > k + 1 and j == k):

                    if arr[i + 1, j] != 0:
                        adjacency[arr[i, j], arr[i + 1, j]] = 1
                        adjacency[arr[i + 1, j], arr[i, j]] = 1
                adjacency[arr[k, 1], arr[k - 1, 1]] = 1
                adjacency[arr[k, n], arr[k + 1, n]] = 1
                adjacency[arr[1, k], arr[1, k + 1]] = 1
                adjacency[arr[n, k], arr[n, k - 1]] = 1
                adjacency[arr[k, k - 2], arr[k - 1, k - 2]] = 1
                adjacency[arr[k, k + 2], arr[k + 1, k + 2]] = 1
                adjacency[arr[k - 2, k], arr[k - 2, k + 1]] = 1
                adjacency[arr[k + 2, k], arr[k + 2, k - 1]] = 1
                if(start_pos_x==k and start_pos_y==1):
                    adjacency[arr[start_pos_x, start_pos_y], arr[start_pos_x, start_pos_y + 1]] = 0
                    adjacency[arr[start_pos_x, start_pos_y + 1], arr[start_pos_x, start_pos_y]] = 0
                elif(start_pos_x==k and start_pos_y==9):
                    adjacency[arr[start_pos_x, start_pos_y], arr[start_pos_x, start_pos_y - 1]] = 0
                    adjacency[arr[start_pos_x, start_pos_y - 1], arr[start_pos_x, start_pos_y]] = 0
                elif(start_pos_x==1 and start_pos_y==k):
                    adjacency[arr[start_pos_x, start_pos_y], arr[start_pos_x + 1, start_pos_y ]] = 0
                    adjacency[arr[start_pos_x + 1, start_pos_y ], arr[start_pos_x, start_pos_y]] = 0
                elif(start_pos_x==9 and start_pos_y==k):
                    adjacency[arr[start_pos_x, start_pos_y], arr[start_pos_x - 1, start_pos_y]] = 0
                    adjacency[arr[start_pos_x - 1, start_pos_y], arr[start_pos_x, start_pos_y]] = 0



    return adjacency


class Graph:

    # Constructor
    def __init__(self):

        # default dictionary to store graph
        self.graph = defaultdict(list)

        # function to add an edge to graph

    def addEdge(self, u, v):
        self.graph[u].append(v)

        # Function to print a BFS of graph


def BFS_SP(graph, start, goal,shortest_path):
    explored = []

    # Queue for traversing the
    # graph in the BFS
    queue = [[start]]
    coloured_path=[38,39,14,23,43,44,59,68]

    # If the desired node is
    # reached
    if start == goal:
        print("Same Node")
        return

    # Loop to traverse the graph
    # with the help of the queue
    while queue:
        path = queue.pop(0)
        node = path[-1]

        # Codition to check if the
        # current node is not visited
        if node not in explored:
            neighbours = graph[node]

            # Loop to iterate over the
            # neighbours of the node
            for neighbour in neighbours:
                new_path = list(path)

                new_path.append(neighbour)
                queue.append(new_path)

                # Condition to check if the
                # neighbour node is the goal
                if neighbour == goal:
                    #print("Shortest path = ", *new_path)

                    shortest_path.append(new_path)
                    #print(shortest_path)

                    return shortest_path
            #if node not in coloured_path:
            explored.append(node)


            # Condition when the nodes
    # are not connected
    #print("So sorry, but a connecting path doesn't exist :(")
    return None
# Driver code

# Create a graph given in
# the above diagram








start_pos_x = robopos(env.camera_feed(),detect_Aruco(env.camera_feed()))[0]
start_pos_y = robopos(env.camera_feed(),detect_Aruco(env.camera_feed()))[1]
current_pos_x=start_pos_x
current_pos_y=start_pos_y
adjacency = make_adjacency(arr, n, start_pos_x, start_pos_y)
print(arr)
print(start_pos_x,start_pos_y)

g = Graph()
for i in range(1, 82):
    for j in range(1, 82):
        if adjacency[i, j] == 1:
            g.addEdge(i, j)



times=0
cv2.destroyAllWindows()
final_position_reached=0

prev_x=0
prev_y=0

while 1:
    shortest_path = []
    print("*** GETTING INPUT ***")
    to_find=0
    val=env.roll_dice()
    print("got input {}".format(val))
    if val=='CR':
        to_find=6
    elif val=='SR':
        to_find=5
    elif val=='TR':
        to_find=4
    elif val=='CY':
        to_find=3
    elif val=='SY':
        to_find=2
    elif val=='TY':
        to_find=1


    found_list=[]
    for i in range(9):
        for j in range(9):
            y=array[i,j]
            if y%10==to_find and not(i==current_pos_x and j==current_pos_y):
                found_list.append(9*i+j+1)

    for i in found_list:
        checker=BFS_SP(g.graph,arr[current_pos_x,current_pos_y],i,shortest_path)
        if checker is None:
            continue
        else:
            shortest_path_list=checker


    if shortest_path_list is None:
        continue

    else:
        final_path = min(shortest_path_list, key=lambda x: len(x))
        print("Found Path")
        print(final_path)

        current_pos_x = int((final_path[-1] - 1) / 9)
        current_pos_y = int((final_path[-1] - 1) % 9)



    if times==0:
        times = times + 1


    elif times==1:

        if start_pos_x == l and start_pos_y == 0:
            adjacency[arr[start_pos_x, start_pos_y], arr[start_pos_x, start_pos_y + 1]] = 1
            adjacency[arr[start_pos_x, start_pos_y + 1], arr[start_pos_x, start_pos_y]] = 1
            adjacency[arr[start_pos_x, start_pos_y], arr[start_pos_x - 1, start_pos_y]] = 0
            adjacency[arr[start_pos_x, start_pos_y + 2], arr[start_pos_x - 1, start_pos_y + 2]] = 0
            adjacency[arr[start_pos_x, start_pos_y + 2], arr[start_pos_x, start_pos_y + 3]] = 1
        elif start_pos_x == l and start_pos_y == 8:
            adjacency[arr[start_pos_x, start_pos_y], arr[start_pos_x, start_pos_y - 1]] = 1
            adjacency[arr[start_pos_x, start_pos_y - 1], arr[start_pos_x, start_pos_y]] = 1
            adjacency[arr[start_pos_x, start_pos_y], arr[start_pos_x + 1, start_pos_y]] = 0
            adjacency[arr[start_pos_x, start_pos_y - 2], arr[start_pos_x + 1, start_pos_y - 2]] = 0
            adjacency[arr[start_pos_x, start_pos_y - 2], arr[start_pos_x, start_pos_y - 3]] = 1
        elif start_pos_x == 0 and start_pos_y == l:
            adjacency[arr[start_pos_x, start_pos_y], arr[start_pos_x + 1, start_pos_y]] = 1
            adjacency[arr[start_pos_x + 1, start_pos_y], arr[start_pos_x, start_pos_y]] = 1
            adjacency[arr[start_pos_x, start_pos_y], arr[start_pos_x, start_pos_y + 1]] = 0
            adjacency[arr[start_pos_x + 2, start_pos_y], arr[start_pos_x + 2, start_pos_y + 1]] = 0
            adjacency[arr[start_pos_x + 2, start_pos_y], arr[start_pos_x + 3, start_pos_y]] = 1
        elif start_pos_x == 8 and start_pos_y == l:
            adjacency[arr[start_pos_x, start_pos_y], arr[start_pos_x - 1, start_pos_y]] = 1
            adjacency[arr[start_pos_x - 1, start_pos_y], arr[start_pos_x, start_pos_y]] = 1
            adjacency[arr[start_pos_x, start_pos_y], arr[start_pos_x, start_pos_y - 1]] = 0
            adjacency[arr[start_pos_x - 2, start_pos_y], arr[start_pos_x - 2, start_pos_y - 1]] = 0
            adjacency[arr[start_pos_x - 2, start_pos_y], arr[start_pos_x - 3, start_pos_y]] = 1


        g = Graph()
        for i in range(1, 82):
            for j in range(1, 82):
                if adjacency[i, j] == 1:
                    g.addEdge(i, j)
        times=times+1
    
    e=5


    ##########################MOVEMENT######################################

    if(prev_x!=final_path[-1]):
        prev_x=final_path[-1]
    elif prev_x==final_path[-1]:
        print("given shape is not accessible")
        continue
    print("*********************************************************************")

    for i in final_path[1:]:
        
        to_reach_x=25+26*(2*int((i-1)%9)+1) -(5-int((i-1)%9))
        to_reach_y=25+26*(2*int((i-1)/9)+1) -(5-int((i-1)/9))

        present_x=25+26*(2*int((prev_x-1)%9)+1) -(5-int((i-1)%9))
        present_y=25+26*(2*int((prev_y-1)/9)+1) -(5-int((i-1)/9))
        img = env.camera_feed()

        aruco_list = detect_Aruco(img)
        state=calculate_Robot_State(aruco_list)
        if len(state)>0:



            present_x = state[0]
            present_y = state[1]

        required_angle = 90+angle_calculate( (present_x, present_y),(to_reach_x, to_reach_y))

        count=0

        alpha=10

        while 1:

            if len(state)>0:

                if state[2]>-30 or required_angle<-5:

                    alpha = required_angle - state[2]
                else:
                    alpha=required_angle -(340+state[2])
                if math.fabs(alpha) < 10:
                    break
            else:
                alpha=-alpha


            p.stepSimulation()
            #env.move_husky(0.1,0.1,0.1,0.1)
            env.move_husky(alpha * 0.06, -alpha * 0.04, alpha * 0.06, -alpha * 0.04)

            if count%40==0:

                img = env.camera_feed()
                aruco_list=detect_Aruco(img)


                if len(calculate_Robot_State(aruco_list))>0:
                    state = calculate_Robot_State(aruco_list)

                    present_x=calculate_Robot_State(aruco_list)[0]
                    present_y=calculate_Robot_State(aruco_list)[1]

            count=count+1

        check=math.fabs(to_reach_x-present_x) + math.fabs(to_reach_y-present_y)
        checking=25
        distance_moved=0

        while check>checking:

            if -40<required_angle<40 or 140<required_angle<200 or -180<required_angle<-140:
                check=math.fabs(to_reach_y-present_y)
                checking=10
            else:
                check=math.fabs(to_reach_x - present_x)
                checking=10

            p.stepSimulation()
            env.move_husky(0.7*1.5, 0.75*1.5, 0.65*1.5, 0.71*1.5)
            if count%80==0:

                img=env.camera_feed()
                aruco_list=detect_Aruco(img)
                if len(calculate_Robot_State(aruco_list))>0:
                    present_x=calculate_Robot_State(aruco_list)[0]
                    present_y=calculate_Robot_State(aruco_list)[1]

                else:
                    env.move_husky(0.7*2, -0.75*2, -0.65*2, -0.7*2)
            count=count+1

        env.move_husky(0, 0, 0, 0)
        final_position_reached_list=[40,32,42,50]

        for z in final_position_reached_list:
            if i==z:
                i=41
                to_reach_x = 25 + 26 * (2 * int((i - 1) % 9) + 1) - (5 - int((i - 1) % 9))
                to_reach_y = 25 + 26 * (2 * int((i - 1) / 9) + 1) - (5 - int((i - 1) / 9))

                img = env.camera_feed()

                aruco_list = detect_Aruco(img)
                present_x = calculate_Robot_State(aruco_list)[0]
                present_y = calculate_Robot_State(aruco_list)[1]
                print(present_x, present_y)
                print(to_reach_x, to_reach_y)
                required_angle = 90 + angle_calculate((present_x, present_y), (to_reach_x, to_reach_y))

                print(required_angle)
                print(calculate_Robot_State(aruco_list)[2])

                count = 0

                while 1:
                    alpha = 30
                    robot_state = calculate_Robot_State(aruco_list)
                    if len(robot_state) > 0:
                        if robot_state[2] > -40 or required_angle < -8:
                            alpha = required_angle - robot_state[2]
                        else:
                            alpha = required_angle - (340 + robot_state[2])
                        if math.fabs(alpha) < 10:
                            break



                    p.stepSimulation()
                    env.move_husky(alpha * 0.01, -alpha * 0.01, alpha * 0.01, -alpha * 0.01)


                    if count % 50 == 0:

                        img = env.camera_feed()
                        aruco_list = detect_Aruco(img)

                        if len(calculate_Robot_State(aruco_list)) > 0:

                            present_x = calculate_Robot_State(aruco_list)[0]
                            present_y = calculate_Robot_State(aruco_list)[1]

                    count = count + 1

                check = math.fabs(to_reach_x - present_x) + math.fabs(to_reach_y - present_y)
                checking = 25

                while check > checking:

                    if -40 < required_angle < 40 or 140 < required_angle < 200 or -180 < required_angle < -140:
                        check = math.fabs(to_reach_y - present_y)
                        checking = 15
                    else:
                        check = math.fabs(to_reach_x - present_x)
                        checking = 15


                    p.stepSimulation()
                    env.move_husky(0.7*5, 0.75*5, 0.65*5, 0.7*5)
                    if count % 100 == 0:

                        img = env.camera_feed()
                        aruco_list = detect_Aruco(img)
                        if len(calculate_Robot_State(aruco_list)) > 0:
                            present_x = calculate_Robot_State(aruco_list)[0]
                            present_y = calculate_Robot_State(aruco_list)[1]

                        else:
                            env.move_husky(-0.7*5, -0.75*5, -0.65*5, -0.7*5)
                    count = count + 1
                env.move_husky(0, 0, 0, 0)

                print("#################   FINAL POSITION REACHED   ######################")
                final_position_reached=1
                break


    if final_position_reached==1:
        break
