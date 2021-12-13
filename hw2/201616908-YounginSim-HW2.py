import math
import glob
import numpy as np
from PIL import Image, ImageDraw
from itertools import combinations

# parameters

datadir = './data'
resultdir = './results'

# you can calibrate these parameters

sigma = 2
highThreshold = 0.3
lowThreshold = 0.05
rhoRes = 1
thetaRes = math.pi / 180

nLines = 20

is_debug = 2


def convolve(input_image, Kernel):
    """
    Args:
        input_image (numpy array): input array
        Kernel (numpy array): kernel shape of (height, width)
    Return:
        convolved image (numpy array)
    """
    shape = input_image.shape
    kernel_shape = Kernel.shape
    if len(kernel_shape) < 2:
        kernel_shape += (1,)

    input_image = replication_padding(input_image, kernel_shape)

    Kernel = np.flip(Kernel)  # flip Kernel

    kernel_x, kernel_y = kernel_shape
    result = []

    for i in range(shape[0]):
        for j in range(shape[1]):
            result.append((input_image[i:i + kernel_x, j:j + kernel_y] * Kernel).sum())
    result = np.array(result).reshape(shape)
    output_image = result

    return output_image


def replication_padding(input_image, size):
    size_x = (size[0] - 1) // 2
    size_y = (size[1] - 1) // 2
    tmp = []
    for arr in input_image:
        left = list([arr[0]]) * size_y
        right = list([arr[-1]]) * size_y
        tmp.append(left + list(arr) + right)

    output_image = np.array([tmp[0]] * size_x + tmp + [tmp[-1]] * size_x)

    return output_image


def ConvFilter(Igs, G):
    # TODO ...
    return convolve(Igs, G)


def connect_edge(Im_nms, Im_th, i, j):
    if not (0 <= i < Im_nms.shape[0] and 0 <= j < Im_nms.shape[1]):
        return
    if Im_th[i, j] > 0:
        return
    if Im_nms[i, j] > lowThreshold:
        Im_th[i, j] = 1
        dxs = [-1, -1, -1, 0, 0, 1, 1, 1]
        dys = [-1, 0, 1, -1, 1, -1, 0, 1]

        for dx in dxs:
            for dy in dys:
                connect_edge(Im_nms, Im_th, i + dx, i + dy)


def EdgeDetection(Igs, sigma, highThreshold, lowThreshold):
    # TODO ...

    # gaussian filtering
    size = (9, 9)

    gaussian_x = np.array(
        [(1.0 / np.sqrt(2 * np.pi * (sigma ** 2))) * np.exp(-((i - (size[0] // 2)) ** 2) / (2 * (sigma ** 2))) for i
         in range(size[0])], np.float32)
    gaussian_x = gaussian_x / gaussian_x.sum()
    gaussian_y = np.array(
        [(1.0 / np.sqrt(2 * np.pi * (sigma ** 2))) * np.exp(-((i - (size[1] // 2)) ** 2) / (2 * (sigma ** 2))) for i
         in range(size[1])], np.float32)
    gaussian_y = gaussian_y / gaussian_y.sum()

    Ig_x = ConvFilter(Igs, gaussian_x.reshape(size[0], 1))
    Ig = ConvFilter(Ig_x, gaussian_y.reshape(1, size[1]))
    #
    # if is_debug == 2:
    #     print('gaussian x')
    #     image = Image.fromarray(Igs * 255)
    #     image.show()
    #     print('gaussion x y')
    #     image = Image.fromarray(Ig * 255)
    #     image.show()

    # sobel filtering
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

    Ix = ConvFilter(Ig, sobel_x)
    Iy = ConvFilter(Ig, sobel_y)

    # if is_debug == 2:
    #     print('sobel x')
    #     image = Image.fromarray(Ix / Ix.max() * 255)
    #     image.show()
    #     print('sobel y')
    #     image = Image.fromarray(Iy * 255)
    #     image.show()

    Im = np.sqrt(Ix * Ix + Iy * Iy)

    # arc tangent of x1/x2
    # return [-pi, pi]
    Io = np.arctan2(Iy, Ix)

    # if is_debug == 2:
    #     image = Image.fromarray(Im * 255)
    #     image.show()
        # image = Image.fromarray(Io * 255)
        # image.show()

    # interpolation nms
    Im_nms = np.zeros(Im.shape)
    for i in range(1, Im.shape[0] - 1):
        for j in range(1, Im.shape[1] - 1):
            theta = Io[i, j]
            param = (0, 0, 0, 0, 0)
            if 0 <= theta < np.pi / 4 or -np.pi <= theta < -np.pi * 3 / 4:
                param = (np.abs(np.tan(theta)), 1, 0, 1)
            elif np.pi / 4 <= theta < np.pi / 2 or -np.pi * 3 / 4 <= theta < -np.pi / 2:
                param = (np.abs(1 / np.tan(theta)), 1, -1, 0)
            elif np.pi / 2 <= theta < np.pi * 3 / 4 or -np.pi / 2 <= theta < -np.pi / 4:
                param = (np.abs(1 / np.tan(theta)), -1, -1, 0)
            else:  # np.pi * 3 / 4 <= theta <= np.pi or -np.pi / 4 <= theta < 0:
                param = (np.abs(np.tan(theta)), -1, 0, -1)

            a = param[0]
            p = a * Im[i - 1, j + param[1]] + (1 - a) * Im[i+param[2], j + param[3]]
            r = a * Im[i + 1, j - param[1]] + (1 - a) * Im[i-param[2], j - param[3]]

            if Im[i, j] > max(p, r):
                Im_nms[i, j] = Im[i, j]
            else:
                Im_nms[i, j] = 0

    # if is_debug == 2:
    #     image = Image.fromarray(Im_nms * 255)
    #     image.show()

    # Double Thresholding : include edge connected
    Im_th = np.zeros(Im_nms.shape)
    for i in range(1, Im_nms.shape[0] - 1):
        for j in range(1, Im_nms.shape[1] - 1):
            if Im_nms[i, j] > highThreshold:
                connect_edge(Im_nms, Im_th, i, j)

    for i in range(1, Im_nms.shape[0] - 1):
        for j in range(1, Im_nms.shape[1] - 1):
            if Im_th[i, j] > 0:
                connect_edge(Im_nms, Im_th, i, j)
    # if is_debug == 3:
    #     image = Image.fromarray(Im_th * 255)
    #     image.show()

    return Im_th, Io, Ix, Iy


def HoughTransform(Im, rhoRes, thetaRes):
    # TODO ...
    size_x, size_y = Im.shape
    diagonal = np.round(np.sqrt(size_x ** 2 + size_y ** 2))  # max of x cos + y sin

    thetas = np.arange(-90, 90, int(thetaRes * 180 / np.pi))
    thetas = np.deg2rad(thetas)

    cos_thetas = np.cos(thetas)
    sin_thetas = np.sin(thetas)

    H = np.zeros((int(2 * diagonal // rhoRes), len(thetas)))

    for i in range(size_x):
        for j in range(size_y):
            if Im[i, j] > 0:
                for theta_idx in range(len(thetas) - 1):
                    # calculate the rho : diagonal is added for a positive index
                    rho = int(round(j * cos_thetas[theta_idx] + i * sin_thetas[theta_idx])) + diagonal
                    rho_idx = int(rho // rhoRes)
                    H[rho_idx, theta_idx] += 1

    # if is_debug == 3:
    #     image = Image.fromarray(H / H.max() * 255)
    #     image.show()

    return H

def range_min_max_coordinates(i, nhood_size_x, size_x):
    # if idx_x is too close to the edges choose appropriate values
    if (i - (nhood_size_x // 2)) < 0:
        min_x = 0
    else:
        min_x = i - (nhood_size_x // 2)

    if (i + (nhood_size_x // 2) + 1) > size_x:
        max_x = size_x
    else:
        max_x = i + (nhood_size_x // 2) + 1
    return min_x, max_x


def HoughLines(H, rhoRes, thetaRes, nLines):
    # TODO ...
    # reference https://gist.github.com/bygreencn/6a900fd2ff5d0101473acbc3783c4d92
    # but I think another idea
    size_x, size_y = H.shape
    H1 = H.copy()
    nhood_size_x, nhood_size_y = 5, 5
    for i in range(size_x):
        for j in range(size_y):

            min_x, max_x = range_min_max_coordinates(i, nhood_size_x, size_x)
            min_y, max_y = range_min_max_coordinates(j, nhood_size_y, size_y)

            is_valid = True
            for x in range(min_x, max_x):
                for y in range(min_y, max_y):
                    if H[i, j] < H[x, y]:
                        is_valid = False
                        break
                if not is_valid:
                    break

            if is_valid:
                H1[i, j] = H[i, j]
            else:
                H1[i, j] = 0

    image = Image.fromarray(H1 / H1.max() * 255)
    # image.show()
    lRho = []
    lTheta = []
    dist_constraint = int(np.sqrt(size_x**2 + size_y**2) * 0.03)
    cnt = 0
    while True:
        idx = np.argmax(H1)  # find argmax in flattened array
        H1_idx = np.unravel_index(idx, H1.shape)  # remap to shape of H
        h_rho, h_theta = H1_idx[0], H1_idx[1]
        a = [(elem[0]-h_rho)**2 + (elem[1]-h_theta)**2 for elem in zip(lRho, lTheta)]
        H1[h_rho, h_theta] = 0
        is_valid = True
        for an in a:
            if an < dist_constraint:
                is_valid = False
        if is_valid:
            lRho.append(h_rho)
            lTheta.append(h_theta)
            cnt += 1
        if cnt >= nLines:
            break
    # for i in range(nLines):
    #     idx = np.argmax(H1)  # find argmax in flattened array
    #     H1_idx = np.unravel_index(idx, H1.shape)  # remap to shape of H
    #     min_x, max_x = range_min_max_coordinates(H1_idx[0], dist_constraint, size_x)
    #     min_y, max_y = range_min_max_coordinates(H1_idx[1], dist_constraint, size_y)
    #     for x in range(min_x, max_x):
    #         for y in range(min_y, max_y):
    #             H1[x, y] = 0
    #     lRho.append(H1_idx[0])
    #     lTheta.append(H1_idx[1])

    lRho = np.array(lRho)
    lTheta = np.array(lTheta)/(math.pi / 180 / thetaRes) - 90
    # print(lRho, lTheta)
    return lRho, lTheta


# dist^2
def distSquare(x1, y1, x2, y2):
    return (x1 - x2) ** 2 + (y1 - y2) ** 2


def HoughLineSegments(lRho, lTheta, Im):
    # TODO ...
    '''
    idea) (rho, theta)에 해당하는 직선을 paper_accumulated 에 담는다. 두 번 이상 지난 점은 교점이라는 가정을 이용하여 교점을 구한다.
    각 직선과 위에서 구한 점과의 교 집합은 한 라인의 교점들이다. 그 교점들 사이의 거리가 가장 긴 segment를 구한다.
    '''
    l = [dict() for x in range(nLines)]
    size_x, size_y = Im.shape
    im_threshold_x, im_threshold_y = 8, 8
    group_threshold_x, group_threshold_y = size_x//60, size_y//60
    for idx in range(nLines):
        paper = Image.fromarray(np.zeros(Im.shape))
        draw = ImageDraw.Draw(paper)

        diagonal = int(np.ceil(np.hypot(size_x, size_y)))
        cos_theta = np.cos(np.deg2rad(lTheta))
        sin_theta = np.sin(np.deg2rad(lTheta))

        if lTheta[idx] != 0:
            x1 = 0
            y1 = int(round(((lRho[idx] * rhoRes) - diagonal - x1 * cos_theta[idx]) / sin_theta[idx]))
            x2 = size_y - 1
            y2 = int(round(((lRho[idx] * rhoRes) - diagonal - x2 * cos_theta[idx]) / sin_theta[idx]))
            draw.line((x1, y1, x2, y2), fill="white", width=1)
        else:  # vertical
            x = int(round((lRho[idx] * rhoRes) - diagonal / cos_theta[idx]))
            draw.line((0, x, size_y, x), fill="white", width=1)

        paper = np.array(paper)
        group_idx = 1
        group = [[]]
        for i in range(size_x):
            for j in range(size_y):
                if paper[i, j] > 0:
                    min_x, max_x = range_min_max_coordinates(i, im_threshold_x, size_x)
                    min_y, max_y = range_min_max_coordinates(j, im_threshold_y, size_y)

                    is_valid = False
                    for x in range(min_x, max_x):
                        for y in range(min_y, max_y):
                            if Im[x, y] > 0:
                                is_valid = True
                                break
                        if is_valid:
                            break

                    if is_valid:
                        min_k, max_k = range_min_max_coordinates(i, group_threshold_x, size_x)
                        min_l, max_l = range_min_max_coordinates(j, group_threshold_y, size_y)
                        is_group = False
                        cur_idxs = set([])
                        for k in range(min_k, max_k):
                            for ll in range(min_l, max_l):
                                for idxx in range(group_idx):
                                    if (k, ll) in group[idxx]:
                                        cur_idxs.add(idxx)
                                        is_group = True

                        if is_group:
                            for cur_idx in cur_idxs:
                                group[cur_idx].append((i, j))
                        else:
                            group.append([])
                            group[group_idx].append((i, j))
                            group_idx += 1

        dist = 0
        l[idx]['start'] = (0, 0)
        l[idx]['end'] = (0, 0)

        for g in group:
            for a, b in combinations(g, 2):
                if dist < distSquare(a[0], a[1], b[0], b[1]):
                    l[idx]['start'] = a
                    l[idx]['end'] = b
                    dist = distSquare(a[0], a[1], b[0], b[1])

    return l


def main():
    # read images
    global sigma, highThreshold, lowThreshold, rhoRes, thetaRes
    for img_path in glob.glob(datadir + '/*2.jpg'):
        # load grayscale image
        num = img_path[-5:-4]
        if num == '1':
            sigma = 2
            highThreshold = 0.2
            lowThreshold = 0.1
            rhoRes = 1.0
            thetaRes = math.pi / 180
        elif num == '2':
            sigma = 2
            highThreshold = 0.18
            lowThreshold = 0.1
            rhoRes = 1.0
            thetaRes = math.pi / 180
        elif num == '3':
            sigma = 2
            highThreshold = 0.2
            lowThreshold = 0.1
            rhoRes = 1.0
            thetaRes = math.pi / 180
        elif num == '4':
            sigma = 2
            highThreshold = 0.13
            lowThreshold = 0.06
            rhoRes = 1.0
            thetaRes = math.pi / 180
        else:
            sigma = 2
            highThreshold = 0.12
            lowThreshold = 0.05
            rhoRes = 1
            thetaRes = math.pi / 180

        img_line = Image.open(img_path)
        img_segment = Image.open(img_path)
        img = Image.open(img_path).convert("L")

        Igs = np.array(img)
        Igs = Igs / 255.

        # Hough function
        Im, Io, Ix, Iy = EdgeDetection(Igs, sigma, highThreshold, lowThreshold)
        # Im
        image = Image.fromarray(Im / Im.max() * 255)
        image.convert('RGB').save('./Im.jpg')
        # image.show()
        H = HoughTransform(Im, rhoRes, thetaRes)
        lRho, lTheta = HoughLines(H, rhoRes, thetaRes, nLines)
        # H
        image = Image.fromarray(H / H.max() * 255)
        image.convert('RGB').save('H.jpg')
        # image.show()
        # plot HoughLines to the original image
        draw = ImageDraw.Draw(img_line)
        size_x, size_y = Im.shape
        for idx in range(nLines):
            diagonal = int(np.ceil(np.hypot(size_x, size_y)))
            cos_theta = np.cos(np.deg2rad(lTheta))
            sin_theta = np.sin(np.deg2rad(lTheta))

            if lTheta[idx] != 0:
                x1 = 0
                y1 = int(round(((lRho[idx] * rhoRes) - diagonal - x1 * cos_theta[idx]) / sin_theta[idx]))
                x2 = size_y - 1
                y2 = int(round(((lRho[idx] * rhoRes) - diagonal - x2 * cos_theta[idx]) / sin_theta[idx]))
                draw.line((x1, y1, x2, y2), fill="red", width=1)
            else:  # vertical
                x = int(round((lRho[idx] * rhoRes) - diagonal / cos_theta[idx]))
                draw.line((0, x, size_y, x), fill="red", width=1)

        img_line.save('houghline.jpg')
        # img_line.show()

        l = HoughLineSegments(lRho, lTheta, Im)
        # plot hough segment to the original image
        draw = ImageDraw.Draw(img_segment)
        for idx in range(nLines):
            draw.line(((l[idx]['start'][1], l[idx]['start'][0]), (l[idx]['end'][1], l[idx]['end'][0])), fill="red",
                      width=2)
        img_segment.save('segment.jpg')
        # img_segment.show()

        # saves the outputs to files
        # Im, H, Im + hough line , Im + hough line segments


if __name__ == '__main__':
    main()
