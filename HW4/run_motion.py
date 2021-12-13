import os
import numpy as np
import cv2
from scipy.interpolate import RectBivariateSpline
from skimage.filters import apply_hysteresis_threshold

IS_DEBUG = True

p = np.zeros(6)

def lucas_kanade_affine(img1, img2, p, Gx, Gy):
    '''
    TODO
    1. Warp I with W(x;p) to I(W(x;p))
    2. compute error image T(x)-I(W(x;p))
    3. Warp gradient of I to compute grad(I)
    4. Evaluate Jacobian dW/dp
    5. compute Hessian H
    6. compute dp
    dp = inv(hessian) * sigma_term
    :param img1: I(t)
    :param img2: I(t+1)
    :param p:
    :param Gx: Ix
    :param Gy: Iy
    :return dp:
    '''
    ### START CODE HERE ###
    # [Caution] From now on, you can only use numpy and 
    # RectBivariateSpline. Never use OpenCV.

    # img1 : T(x),  img2 : I(W(x;p))
    # In this case, the image shape is like this ~ (318, 636).
    h1, w1 = img1.shape
    h2, w2 = img2.shape

    # subsampling(version 1. entire img)
    y_flats = np.repeat(range(h1), w1)
    x_flats = np.tile(range(w1), h1)
    yx_flats = np.concatenate((y_flats.reshape(-1, 1), x_flats.reshape(-1, 1)), axis=1)

    # compute hessian
    G_max = 128

    def compute_hessian_element(yx_flat):
        y, x = yx_flat
        gradI = np.array([Gx[y, x], Gy[y, x]])/G_max
        jacobian = np.array([[x, 0, y, 0, 1, 0], [0, x, 0, y, 0, 1]])
        gradI_jacobian = np.asarray([gradI.dot(jacobian)])
        return (gradI_jacobian.T).dot(gradI_jacobian)

    # hessian (6X6)
    hessian = np.sum(np.apply_along_axis(compute_hessian_element, 1, yx_flats), axis=0)
    # interpolate the img2
    img2_smooth = RectBivariateSpline(np.arange(h2), np.arange(w2), img2)

    def sigma_term_element(yx_flat):
        # gradI * jacobian
        y, x = yx_flat
        gradI = np.array([Gx[y, x], Gy[y, x]]) / G_max
        jacobian = np.array([[x, 0, y, 0, 1, 0], [0, x, 0, y, 0, 1]])
        gradI_jacobian = np.asarray([gradI.dot(jacobian)])

        # T(x)
        t_x = img1[y, x]

        # I(W(x;p))
        p1, p2, p3, p4, p5, p6 = p
        w_xp = np.asarray([p2*x+(1+p4)*y+p6,(1+p1)*x+p3*y+p5])
        img2_w_xp = img2_smooth(w_xp[0],w_xp[1])[0,0]

        return (gradI_jacobian.T).dot(t_x-img2_w_xp)

    sigma_term = np.sum(np.apply_along_axis(sigma_term_element, 1, yx_flats), axis=0)

    dp = np.linalg.inv(hessian).dot(sigma_term).reshape(6)
    ### END CODE HERE ###
    return dp

def subtract_dominant_motion(img1, img2):
    global p

    Gx = cv2.Sobel(I, cv2.CV_64F, 1, 0, ksize=5)
    Gy = cv2.Sobel(I, cv2.CV_64F, 0, 1, ksize=5)

    ### START CODE HERE ###
    # [Caution] From now on, you can only use numpy and
    # RectBivariateSpline. Never use OpenCV.

    h1, w1 = img1.shape
    h2, w2 = img2.shape

    moving_image = np.zeros_like(img2)
    img1_smooth = RectBivariateSpline(np.arange(h1), np.arange(w1), img1)

    # subsampling(version 1. entire img)
    y_flats = np.repeat(range(h1), w1)
    x_flats = np.tile(range(w1), h1)
    yx_flats = np.concatenate((y_flats.reshape(-1, 1), x_flats.reshape(-1, 1)), axis=1)

    dp = lucas_kanade_affine(img1, img2, p, Gx, Gy)
    p += dp

    def subtract_dominant_motion_element(yx_flat):
        y, x = yx_flat
        p1, p2, p3, p4, p5, p6 = p

        # t is transition matrix
        t = np.asarray([[1 + p1, p3, p5], [p2, 1 + p4, p6], [0, 0, 1]])
        t_inv = np.linalg.inv(t)
        w_xp_inv = np.asarray([t_inv[1, 0]*x + t_inv[1, 1]*y + t_inv[1, 2],
                               t_inv[0, 0]*x + t_inv[0, 1]*y + t_inv[0, 2]])

        # subtract
        moving_image[y, x] = np.abs(img2[y, x] - img1_smooth(w_xp_inv[0], w_xp_inv[1])[0, 0]).astype(np.uint8)

    np.apply_along_axis(subtract_dominant_motion_element, 1, yx_flats)

    th_hi = 0.2 * 256  # you can modify this
    th_lo = 0.15 * 256  # you can modify this
    ### END CODE HERE ###

    hyst = apply_hysteresis_threshold(moving_image, th_lo, th_hi)
    return hyst

if __name__ == "__main__":
    # colab mode
    # data_dir = '/content/drive/MyDrive/2021-2/컴퓨터비전/HW/HW4/data'
    # video_path = '/content/drive/MyDrive/2021-2/컴퓨터비전/HW/HW4/motion1.mp4'
    # local mode
    data_dir = 'data'
    video_path = 'motion.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 150/20, (636, 318))
    tmp_path = os.path.join(data_dir, "organized-{}.jpg".format(0))
    T = cv2.cvtColor(cv2.imread(tmp_path), cv2.COLOR_BGR2GRAY)
    # real loop
    for i in range(0, 50):
    # test loop
    # for i in range(0, 3):
        if IS_DEBUG:
            print(f'img #{i} is running :)')
        img_path = os.path.join(data_dir, "organized-{}.jpg".format(i))
        I = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
        clone = I.copy()
        moving_img = subtract_dominant_motion(T, I)
        clone = cv2.cvtColor(clone, cv2.COLOR_GRAY2BGR)
        clone[moving_img, 2] = 522
        out.write(clone)
        T = I
    out.release()