import numpy as np
import cv2
import imgproc
import os
import Polygon as plg


def crop_image_by_bbox(image, box):
    w = (int)(np.linalg.norm(box[0] - box[1]))
    h = (int)(np.linalg.norm(box[0] - box[3]))
    width = w
    height = h
    if h > w * 1.5:
        width = h
        height = w
        M = cv2.getPerspectiveTransform(np.float32(box),
                                        np.float32(np.array([[width, 0], [width, height], [0, height], [0, 0]])))
    else:
        M = cv2.getPerspectiveTransform(np.float32(box),
                                        np.float32(np.array([[0, 0], [width, 0], [width, height], [0, height]])))

    warped = cv2.warpPerspective(image, M, (width, height))
    return warped, M

def watershed(oriimage, image, viz=False):
    # viz = True
    boxes = []
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    ret, binary = cv2.threshold(gray, 0.2 * np.max(gray), 255, cv2.THRESH_BINARY)
    # 形态学操作，进一步消除图像中噪点
    kernel = np.ones((3, 3), np.uint8)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mb = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)  # iterations连续两次开操作
    sure_bg = cv2.dilate(mb, kernel, iterations=3)  # 3次膨胀,可以获取到大部分都是背景的区域
    sure_bg = mb
    ret, sure_fg = cv2.threshold(gray, 0.6 * gray.max(), 255, cv2.THRESH_BINARY)
    surface_fg = np.uint8(sure_fg)  # 保持色彩空间一致才能进行运算，现在是背景空间为整型空间，前景为浮点型空间，所以进行转换
    unknown = cv2.subtract(sure_bg, surface_fg)
    ret, markers = cv2.connectedComponents(surface_fg)

    nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(surface_fg,
                                                                         connectivity=4)
    markers = labels.copy() + 1
    # markers = markers+1
    markers[unknown == 255] = 0

    markers = cv2.watershed(oriimage, markers=markers)
    oriimage[markers == -1] = [0, 0, 255]

    for i in range(2, np.max(markers) + 1):
        np_contours = np.roll(np.array(np.where(markers == i)), 1, axis=0).transpose().reshape(-1, 2)
        rectangle = cv2.minAreaRect(np_contours)
        box = cv2.boxPoints(rectangle)

        startidx = box.sum(axis=1).argmin()
        box = np.roll(box, 4 - startidx, 0)
        poly = plg.Polygon(box)
        area = poly.area()
        if area < 10:
            continue
        box = np.array(box)
        boxes.append(box)
    return np.array(boxes)

def saveImage(imagename, image, bboxes, affinity_bboxes, region_scores, affinity_scores, confidence_mask):
        output_image = np.uint8(image.copy())
        output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
        if len(bboxes) > 0:
            affinity_bboxes = np.int32(affinity_bboxes)
            for i in range(affinity_bboxes.shape[0]):
                cv2.polylines(output_image, [np.reshape(affinity_bboxes[i], (-1, 1, 2))], True, (255, 0, 0))
            for i in range(len(bboxes)):
                _bboxes = np.int32(bboxes[i])
                for j in range(_bboxes.shape[0]):
                    cv2.polylines(output_image, [np.reshape(_bboxes[j], (-1, 1, 2))], True, (0, 0, 255))

        target_gaussian_heatmap_color = imgproc.cvt2HeatmapImg(region_scores / 255)
        target_gaussian_affinity_heatmap_color = imgproc.cvt2HeatmapImg(affinity_scores / 255)
        heat_map = np.concatenate([target_gaussian_heatmap_color, target_gaussian_affinity_heatmap_color], axis=1)
        confidence_mask_gray = imgproc.cvt2HeatmapImg(confidence_mask)
        output = np.concatenate([output_image, heat_map, confidence_mask_gray], axis=1)
        outpath = os.path.join('./output', imagename)

        if not os.path.exists(os.path.dirname(outpath)):
            os.mkdir(os.path.dirname(outpath))
        cv2.imwrite(outpath, output)

def get_confidence(real_len, pursedo_len):
    if pursedo_len == 0:
        return 0.
    return (real_len - min(real_len, abs(real_len - pursedo_len))) / real_len

def inference_character_box(detector, image, word, word_bbox):
    word_image, MM = crop_image_by_bbox(image, word_bbox)
    real_word_without_space = word.replace(' ', '').strip()
    real_char_nums = len(real_word_without_space)
    input = word_image.copy()

    wordbox_scale = 64.0 / input.shape[0]
    input = cv2.resize(input, None, fx=wordbox_scale, fy=wordbox_scale)
    input2 = imgproc.normalizeMeanVariance(input, mean=(0.485, 0.456, 0.406),
                                           variance=(0.229, 0.224, 0.225))
    scores = detector.model.predict(input2[np.newaxis])
    region_scores = scores[0, :, :, 0]
    region_scores = np.uint8(np.clip(region_scores, 0, 1) * 255)
    bgr_region_scores = cv2.resize(region_scores, (input.shape[1], input.shape[0]))
    bgr_region_scores = cv2.cvtColor(bgr_region_scores, cv2.COLOR_GRAY2BGR)
    pursedo_bboxes = watershed(input, bgr_region_scores, False)
    _tmp = []
    for i in range(pursedo_bboxes.shape[0]):
        if np.mean(pursedo_bboxes[i].ravel()) > 2:
            _tmp.append(pursedo_bboxes[i])
        else:
            print("filter bboxes", pursedo_bboxes[i])
    pursedo_bboxes = np.array(_tmp, np.float32)
    if pursedo_bboxes.shape[0] > 1:
        index = np.argsort(pursedo_bboxes[:, 0, 0])
        pursedo_bboxes = pursedo_bboxes[index]
    confidence = get_confidence(real_char_nums, len(pursedo_bboxes))
    bboxes = []
    if confidence <= 0.5:
        width = input.shape[1]
        height = input.shape[0]

        width_per_char = width / len(word)
        for i, char in enumerate(word):
            if char == ' ':
                continue
            left = i * width_per_char
            right = (i + 1) * width_per_char
            word_bbox = np.array([[left, 0], [right, 0], [right, height],
                             [left, height]])
            bboxes.append(word_bbox)

        bboxes = np.array(bboxes, np.float32)
        confidence = 0.5

    else:
        bboxes = pursedo_bboxes

    bboxes /= wordbox_scale
    try:
        for j in range(len(bboxes)):
            ones = np.ones((4, 1))
            tmp = np.concatenate([bboxes[j], ones], axis=-1)
            I = np.matrix(MM).I
            ori = np.matmul(I, tmp.transpose(1, 0)).transpose(1, 0)
            bboxes[j] = ori[:, :2]
    except Exception as e:
        print(e)

    bboxes[:, :, 1] = np.clip(bboxes[:, :, 1], 0., image.shape[0] - 1)
    bboxes[:, :, 0] = np.clip(bboxes[:, :, 0], 0., image.shape[1] - 1)
    return bboxes, region_scores, confidence