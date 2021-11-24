import math
import numpy as np
from PIL import Image


def compute_h(p1, p2):
    # TODO ...
    n = p1.shape[0]
    # construct A
    A = [] # (2n X 9)
    for idx in range(n):
        a1 = [p2[idx, 0], p2[idx, 1], 1, 0, 0, 0, -p1[idx,  0]*p2[idx, 0], -p1[idx, 0]*p2[idx, 1], -p1[idx, 0]]
        a2 = [0, 0, 0, p2[idx, 0], p2[idx, 1], 1, -p1[idx, 1]*p2[idx, 0], -p1[idx, 1]*p2[idx, 1], - p1[idx, 1]]
        A.append(a1)
        A.append(a2)
    A = np.array(A)
    # svd
    U, S, V_t = np.linalg.svd(A, compute_uv=True)
    V = V_t.T

    H_flat = V[:, -1] # last column of V
    H = np.reshape(H_flat, (3, 3))
    return H


def normalize_matrix(p):
    '''
    :param p: NX2 numpy array
    :return: NX2 normalized numpy array
    '''
    max_x = p.max(axis=None)
    min_x = p.min(axis=None)

    # nm = normalize matrix
    nm = np.eye(3)
    nm[0, 0] = 1/(max_x-min_x)
    nm[1, 1] = 1/(max_x-min_x)
    nm[0, 2] = -min_x/(max_x-min_x)
    nm[1, 2] = -min_x/(max_x-min_x)
    return nm


def dot_homogenous(p, M):
    '''
    :param p: point  (NX2)
    :param M: Matrix (3X3)
    :return: Mp (NX2)
    '''
    n = p.shape[0]
    p_homo = np.concatenate((p.T, np.ones((1, n))), axis=0)

    result_homo = M.dot(p_homo)
    scale = result_homo[2, :]

    result = (result_homo/scale)[:2].T
    return result


def compute_h_norm(p1, p2):
    # TODO ...
    # construct normalize matrix
    nm1 = normalize_matrix(p1)
    nm1_inv = np.linalg.inv(nm1)

    nm2 = normalize_matrix(p2)

    np1 = dot_homogenous(p1, nm1)
    np2 = dot_homogenous(p2, nm2)

    H = compute_h(np1, np2)
    H = nm1_inv.dot(H.dot(nm2))
    return H


# def warp_p(p, M):
#     '''
#     :param p: point  (NX2)
#     :param M: Matrix (3X3)
#     :return: Mp (NX2)
#     '''
#     n = p.shape[0]
#     p_homo = np.concatenate((p.T, np.ones((1, n))), axis=0)
#
#     result_homo = M.dot(p_homo)
#     scale = result_homo[2, :]
#
#     result = np.rint((result_homo/scale))[:2].astype(int).T
#     return result


def warp_image(igs_in, igs_ref, H):
    # TODO ...
    h_in, w_in = igs_in.shape[:2]
    h_ref, w_ref = igs_ref.shape[:2]
    H_inv = np.linalg.inv(H)    # for inverse warpping

    # igs_warp
    print('igs_warping...')
    igs_warp = np.zeros(igs_ref.shape, dtype=np.uint8)

    points = []
    for j in range(h_ref):
        for i in range(w_ref):
            points.append([i, j])
    points = np.array(points)
    results = dot_homogenous(points, H_inv)

    for j in range(h_ref):
        for i in range(w_ref):
            result = results[j*w_ref+i]
            if 0 <= result[1] < h_in and 0 <= result[0] < w_in:
                n, m = np.floor(result).astype(int)
                b, a = result-np.floor(result)
                if m != h_in-1 and n != w_in-1:
                    igs_warp[j, i] = (1-a)*(1-b)*igs_in[m, n]+a*(1-b)*igs_in[m+1, n]+(1-a)*b*igs_in[m, n+1]+a*b*igs_in[m+1, n+1]

    # igs_merge
    print('igs_merging...')
    tmp = np.rint(dot_homogenous(np.asarray([[0, 0], [w_in - 1, 0], [0, h_in - 1], [w_in - 1, h_in - 1]]), H)).astype(int)
    w_end, h_end = tmp.max(axis=0)
    w_start, h_start = tmp.min(axis=0)

    left = max(0, -w_start)
    right = max(0, w_end - w_ref)
    top = max(0, -h_start)
    bottom = max(0, h_end - h_ref)

    igs_merge = np.zeros((top + h_ref + bottom, left + w_ref + right, 3), dtype=np.uint8)
    igs_merge[top:top + h_ref, left:left + w_ref] = igs_ref

    points = []
    for j in range(h_start, h_end):
        for i in range(w_start, w_end):
            points.append([i, j])

    points = np.array(points)
    results = dot_homogenous(points, H_inv)

    for j in range(h_end - h_start):
        for i in range(w_end - w_start):
            result = results[j * (w_end - w_start) + i]
            if 0 <= result[1] < h_in and 0 <= result[0] < w_in:
                n, m = np.floor(result).astype(int)
                b, a = result - np.floor(result)
                if m != h_in - 1 and n != w_in - 1:
                    igs_merge[j + max(0, h_start), i + max(0, w_start)] = (1 - a) * (1 - b) * igs_in[m, n] + a * (1 - b) * igs_in[m + 1, n] + (1 - a) * b * \
                                     igs_in[m, n + 1] + a * b * igs_in[m + 1, n + 1]


    return igs_warp, igs_merge


def rectify(igs, p1, p2):
    # TODO ...
    H = compute_h_norm(p2, p1)

    igs_ref = np.zeros((1056, 1920, 3), dtype=np.uint8)

    h_in, w_in = igs.shape[:2]
    h_ref, w_ref = igs_ref.shape[:2]
    H_inv = np.linalg.inv(H)  # for inverse warpping

    # igs_rec
    print('igs_rectifying...')
    igs_rec = np.zeros(igs_ref.shape, dtype=np.uint8)

    points = []
    for j in range(h_ref):
        for i in range(w_ref):
            points.append([i, j])
    points = np.array(points)
    results = dot_homogenous(points, H_inv)
    results_floor = np.floor(results).astype(int)
    for j in range(h_ref):
        for i in range(w_ref):
            result = results[j * w_ref + i]
            result_floor = results_floor[j * w_ref + i]
            if 0 <= result[1] < h_in and 0 <= result[0] < w_in:
                n, m = result_floor
                b, a = result - result_floor
                if m != h_in - 1 and n != w_in - 1:
                    igs_rec[j, i] = (1 - a) * (1 - b) * igs[m, n] + a * (1 - b) * igs[m + 1, n] + (1 - a) * b * \
                                     igs[m, n + 1] + a * b * igs[m + 1, n + 1]

    return igs_rec


def set_cor_mosaic():
    # TODO ...

    p_in = np.array([[1439, 492], [1438, 408], [1286, 498.], [1286, 420], [1326, 907], [1281, 947], [1325, 579], [1284, 573]], dtype=int)
    # p_ref = np.array([[673, 501], [672, 426], [539, 501], [537, 426], [575, 892], [534, 934], [575, 582], [536, 574]])
    # p_ref = np.array([[673, 502], [672, 427], [539, 502], [537, 427], [575, 893], [534, 935], [575, 583], [536, 575]])
    p_ref = np.array([[673, 499], [672, 424], [539, 499], [537, 424], [575, 890], [534, 932], [575, 580], [536, 572]], dtype=int)

    return p_in, p_ref


def set_cor_rec():
    # TODO ...
    c_in = np.asarray([[1086, 870], [1100, 155], [1348, 885], [1352, 129]], dtype=int)
    c_ref = np.asarray([[1091, 851], [1091, 139], [1349, 851], [1349, 139]], dtype=int)
    return c_in, c_ref

def main():
    ##############
    # step 1: mosaicing
    ##############

    # read images
    img_in = Image.open('data/porto1.png').convert('RGB')
    img_ref = Image.open('data/porto2.png').convert('RGB')

    # shape of igs_in, igs_ref: [y, x, 3]
    igs_in = np.array(img_in)
    igs_ref = np.array(img_ref)

    # lists of the corresponding points (x,y)
    # shape of p_in, p_ref: [N, 2]
    p_in, p_ref = set_cor_mosaic()

    # p_ref = H * p_in
    H = compute_h_norm(p_ref, p_in)
    igs_warp, igs_merge = warp_image(igs_in, igs_ref, H)

    # plot images
    img_warp = Image.fromarray(igs_warp.astype(np.uint8))
    img_merge = Image.fromarray(igs_merge.astype(np.uint8))

    # save images
    img_warp.save('porto1_warped.png')
    img_merge.save('porto_mergeed.png')

    ##############
    # step 2: rectification
    ##############

    img_rec = Image.open('data/iphone.png').convert('RGB')
    igs_rec = np.array(img_rec)

    c_in, c_ref = set_cor_rec()

    igs_rec = rectify(igs_rec, c_in, c_ref)

    img_rec = Image.fromarray(igs_rec.astype(np.uint8))
    img_rec.save('iphone_rectified.png')


if __name__ == '__main__':
    main()
