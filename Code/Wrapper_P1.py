from pprint import pprint
from random import triangular
import cv2
import dlib
import numpy as np
from numpy import linalg as nl, source
import copy
import math
import sys
import argparse


def U(r):
    u = r**2*math.log(r**2)
    return u


def find_index(list, array):
    temp = np.where((list == array).all(axis=1))
    index = None
    for num in temp[0]:
        index = num
        break
    return index


def compute_landmarks(img, path):
    hog_face_detector = dlib.get_frontal_face_detector()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces1 = hog_face_detector(gray)
    p = "Phase1/shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(p)
    lpoints = []
    for face in faces1:
        landmarks = predictor(gray, face)
        for n in range(0, 68):
            X = landmarks.part(n).x
            Y = landmarks.part(n).y
            lpoints.append((X, Y))
    return lpoints


def TPS(p1, p2):

    p = p1.shape[0]
    px = p2[:, 0]
    py = p2[:, 1]
    lamda = 0.1
    P = np.hstack((p1, np.ones([p, 1])))

    K = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            pi = np.array(p1[i, :])
            pj = np.array(p1[j, :])

            K[i, j] = U(nl.norm(pi-pj) +
                        sys.float_info.epsilon)

    L = np.hstack((K, P))

    P_T = P.T

    P_T = np.hstack((P_T, np.zeros((3, 3))))

    L = np.vstack((L, P_T))
    V_x = np.concatenate([px, np.zeros([3, ])])

    V_x.resize(V_x.shape[0], 1)
    V_y = np.concatenate([py, np.zeros([3, ])])
    V_y.resize(V_y.shape[0], 1)

    I = np.eye(p+3)
    W_x = np.matmul(nl.inv(L + lamda*I), V_x)
    W_y = np.matmul(nl.inv(L + lamda*I), V_y)

    return W_x, W_y


def fxy(p1, p2, W):
    K = np.zeros([p2.shape[0], 1])
    for i in range(p2.shape[0]):

        K[i] = U(np.linalg.norm((p2[i]-p1), ord=2) +
                 sys.float_info.epsilon)

    f = W[-1] + W[-3]*p1[0] + \
        W[-2]*p1[1]+np.matmul(K.T, W[0:-3])
    return int(f[0][0])


def make_mask(img, hull):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(gray)

    cv2.fillConvexPoly(mask, hull, 255)
    face = cv2.bitwise_and(img, img, mask=mask)
    return face, mask


def Blend(source, dest, hull):
    _, mask = make_mask(source, hull)
    (x, y, w, h) = cv2.boundingRect(hull)
    center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))
    seamlessclone = cv2.seamlessClone(
        dest, source, mask, center_face2, cv2.NORMAL_CLONE)
    return seamlessclone


def TPS_twoface(img, path):
    source = copy.deepcopy(img)
    dest = copy.deepcopy(img)
    lpoints = compute_landmarks(source, path)
    lpoints1 = lpoints[:68]
    lpoints2 = lpoints[68:]
    if len(lpoints2) < 5:
        return source,False
    points1 = np.array(lpoints1, np.int32)
    points2 = np.array(lpoints2, np.int32)

    convexhull = cv2.convexHull(points1)
    convexhull2 = cv2.convexHull(points2)
    face, _ = make_mask(dest, convexhull)
    face2, _ = make_mask(source, convexhull2)

    u, v, points = TPS_blend(points1, points2)
    u1, v1, points_1 = TPS_blend(points2, points1)

    mask = np.zeros_like(source)

    for a in range(u.shape[0]):

        if np.any(face2[v[a], u[a], :]) > 0:
            mask[points[a, 1], points[a, 0], :] = face2[v[a], u[a], :]
            source[points[a, 1], points[a, 0], :] = face2[v[a], u[a], :]

    for a in range(u1.shape[0]):
        if np.any(face[v1[a], u1[a], :]) > 0:
            mask[points_1[a, 1], points_1[a, 0], :] = face[v1[a], u1[a], :]

            dest[points_1[a, 1], points_1[a, 0], :] = face[v1[a], u1[a], :]
    cv2.imwrite(path+"maskthin.png",mask)
    cv2.imwrite(path+"before.png",source)
    output = Blend(img, source, convexhull)
    output = Blend(output, dest, convexhull2)
    return output,True


def TPS_blend(point1, point2):

    W_x, W_y = TPS(point1, point2)
    hull = cv2.convexHull(point1)
    hull2 = cv2.convexHull(point2)
    x1, y1, w1, h1 = cv2.boundingRect(hull)
    x2, y2, w2, h2 = cv2.boundingRect(hull2)
    p1_min = np.array([x1, y1])
    p1_max = np.array([x1+w1, y1+h1])

    p2_min = np.array([x2, y2])
    p2_max = np.array([x2+w2, y2+h2])
    x = np.arange(p1_min[0], p1_max[0]).astype(int)
    y = np.arange(p1_min[1], p1_max[1]).astype(int)

    X, Y = np.mgrid[x[0]:x[-1], y[0]:y[-1]]
    points = np.vstack((X.ravel(), Y.ravel()))

    points = points.T

    u = np.zeros_like(points[:, 0])
    v = np.zeros_like(points[:, 0])

    for i in range(points.shape[0]):
        u[i] = fxy(points[i, :], point1, W_x)
        v[i] = fxy(points[i, :], point1, W_y)

    u[u < p2_min[0]] = 0
    u[u > p2_max[0]] = 0

    v[v < p2_min[1]] = 0
    v[v > p2_max[1]] = 0

    return u, v, points


def TPS_face(img1, img2, path):

    source = copy.deepcopy(img1)
    dest = copy.deepcopy(img2)

    lpoints1 = compute_landmarks(source, path)
    lpoints2 = compute_landmarks(dest, path)
    if len(lpoints1) < 5 or len(lpoints1) < 5:
        return source, False

    points1 = np.array(lpoints1, np.int32)
    points2 = np.array(lpoints2, np.int32)
    convexhull = cv2.convexHull(points1)
    convexhull2 = cv2.convexHull(points2)

    face, _ = make_mask(dest, convexhull2)
    u, v, points = TPS_blend(points1, points2)
    for a in range(u.shape[0]):
        if np.any(face[v[a], u[a], :]) > 0:
            source[points[a, 1], points[a, 0], :] = face[v[a], u[a], :]

    output = Blend(img1, source, convexhull)

    return output, True

def tri(rect,lpoints1,points1,points2,mask1,dest):
    
    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(lpoints1)
    triangleList1 = subdiv.getTriangleList()
    triangles1 = np.array(triangleList1, np.int32)
    (x1, y1, h1, w1) = rect
    for i in range(triangles1.shape[0]):

        pts1 = []
        for j in range(6):
            if j % 2 == 0:
                pts1.append((triangles1[i][j], triangles1[i][j+1]))
        
        triangle_index = [find_index(points1, p) for p in pts1]
        triangle = [(points2[p][0], points2[p][1]) for p in triangle_index]
        
        
        B = np.array(pts1).T
        B = np.vstack((B, np.array([1, 1, 1])))

        A = np.array(triangle).T
        A = np.vstack((A, np.array([1, 1, 1])))
        B_inv = nl.inv(B)

        alpha_list = []
        point_list = []

        for k in range(x1, x1+w1):
            for j in range(y1, y1+h1):
                alpha = np.matmul(B_inv, np.array([k, j, 1]))
                if np.all(alpha <= 1) and np.all(alpha >= 0):
                    alpha = (alpha[0], alpha[1], alpha[2])
                    point = (k, j)
                    alpha_list.append(alpha)
                    point_list.append(point)

        if len(alpha_list) != 0:
            alpha = np.array(alpha_list).T
            new = np.matmul(A, alpha)

            new_points = [(new[:, k][0]/new[:, k][2], new[:, k]
                           [1]/new[:, k][2]) for k in range(new.shape[1])]
            for p in range(len(point_list)):

                mask1[point_list[p][1], point_list[p][0],
                      :] = interpolation(dest, new_points[p])
    return mask1

def detect_face(img1, img2, path, flag):
    
    source = copy.deepcopy(img1)
    dest = copy.deepcopy(img2)
    

    lpoints1 = compute_landmarks(source, path)

    lpoints2 = compute_landmarks(dest, path)
    if flag == 1:
        lpoints1 = lpoints1[:68]
        lpoints2 = lpoints2[68:]
            
    if  len(lpoints1) < 5 or  len(lpoints2) < 5:
        return source, False

    points1 = np.array(lpoints1, np.int32)
    points2 = np.array(lpoints2, np.int32)

    convexhull = cv2.convexHull(points1)
    convexhull2 = cv2.convexHull(points2)

    rect = cv2.boundingRect(convexhull)
    rect2 = cv2.boundingRect(convexhull2)
    
    mask = np.zeros_like(source)
    
    mask1= tri(rect, lpoints1, points1, points2, mask,dest)
    
    if flag == 1:
        mask2 = tri(rect2, lpoints2, points2, points1, mask, dest)
        
          
    output = Blend(img1, mask1, convexhull)
    if flag == 1:
        output = Blend(output, mask2, convexhull2)

    return output, True


def interpolation(img, p):
        x0,y0 = math.ceil(p[0]),math.ceil(p[1])
        
        dx, dy = x0-p[0], y0 - p[1]

        # 4 Neighour pixels
        pixel = []
        for c in range(3):
            q11 = img[y0, x0,c]
            q21 = img[y0, x0 + 1,c]
            q12 = img[y0 + 1, x0,c]
            q22 = img[y0 + 1, x0 + 1,c]

            btm = q21.T * dx + q11.T * (1 - dx)
            top = q22.T * dx + q12.T * (1 - dx)
            pixel.append(top * dy + btm * (1 - dy))
        return pixel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--InputFilePath', default='my_test_imgs/', help='Folder in which the input files are located')
    parser.add_argument('--InputFileName', default='Test1.mp4', help='Name of the input file')
    parser.add_argument('--RefFileName', default='None', help='Reference image to swap with')
    parser.add_argument('--Method', default='tri', help='Method to use, tri for Triangulation or tps for Thin Plate Spline')
    parser.add_argument('--SaveFileName', default='Data1outputTringle.mp4', help='File in which to save the outputs')

    args = parser.parse_args()
    path = args.InputFilePath
    file_name = args.InputFileName
    ref_file = args.RefFileName
    method = args.Method
    save_file = args.SaveFileName


    cap = cv2.VideoCapture(path + file_name)
    width = int(cap. get(cv2. CAP_PROP_FRAME_WIDTH))
    height = int(cap. get(cv2. CAP_PROP_FRAME_HEIGHT))

    if width+height > 2000:
        width = width/4
        height = height/4

    dim = (int(width), int(height))

    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(path+save_file,
                        cv2.VideoWriter_fourcc(*"avc1"), fps, dim)

    success = True
    dest = cv2.imread(path+ref_file)
    frames = []
    while success:
        success, frame = cap.read()
        if success:
            frames.append(frame)
    cap.release()
    for frame in frames:

        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        sorce = frame
        if dest is not None:
            if method == "tri":
                output, status = detect_face(sorce, dest, path, 0)
            if method == "tps":
                output,status = TPS_face(sorce,dest,path)
        else:
            if method == "tri":
                output, status = detect_face(sorce, sorce, path, 1)
            if method == "tps":
                output,status= TPS_twoface(sorce, path)
        cv2.imshow("output",output)
        cv2.waitKey(1)
        
        out.write(output)
        
    out.release()


if __name__ == "__main__":
    main()
