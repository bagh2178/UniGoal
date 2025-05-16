import colorsys
from PIL import Image, ImageDraw, ImageFont
import cv2
import skimage
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import colors


def get_contour_points(pos, origin, size=20):
    x, y, o = pos
    pt1 = (int(x) + origin[0],
           int(y) + origin[1])
    pt2 = (int(x + size / 1.5 * np.cos(o + np.pi * 4 / 3)) + origin[0],
           int(y + size / 1.5 * np.sin(o + np.pi * 4 / 3)) + origin[1])
    pt3 = (int(x + size * np.cos(o)) + origin[0],
           int(y + size * np.sin(o)) + origin[1])
    pt4 = (int(x + size / 1.5 * np.cos(o - np.pi * 4 / 3)) + origin[0],
           int(y + size / 1.5 * np.sin(o - np.pi * 4 / 3)) + origin[1])

    return np.array([pt1, pt2, pt3, pt4])


def draw_line(start, end, mat, steps=25, w=1):
    for i in range(steps + 1):
        x = int(np.rint(start[0] + (end[0] - start[0]) * i / steps))
        y = int(np.rint(start[1] + (end[1] - start[1]) * i / steps))
        mat[x - w:x + w, y - w:y + w] = 1
    return mat


def init_vis_image(goal_name, episode_info, args):
    vis_image = np.ones((655-100, 1380-200-25, 3)).astype(np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (20, 20, 20)  # BGR
    thickness = 2

    # text = f"{goal_name}"
    # textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
    # textX = (215 - textsize[0]) // 2 + 25
    # textY = (50 + textsize[1]) // 2
    # vis_image = cv2.putText(vis_image, text, (textX, textY),
    #                         font, fontScale, color, thickness,
    #                         cv2.LINE_AA)

    if args.goal_type == 'ins-image ':
        text = f"Goal Image"
    elif args.goal_type == 'text':
        text = f"Goal Text"
    elif args.goal_type == 'object':
        text = f"Goal Object"
    else:
        text = f"Goal"
    textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
    textX = (215 - textsize[0]) // 2 + 25
    if args.environment == 'habitat':
        textY = (50 + textsize[1]) // 2
    vis_image = cv2.putText(vis_image, text, (textX, textY),
                            font, fontScale, color, thickness,
                            cv2.LINE_AA)

    if args.environment == 'habitat':
        text = f"Goal Graph"
        textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
        textX = (215 - textsize[0]) // 2 + 25
        textY = (50 + textsize[1]) // 2 + 265
        vis_image = cv2.putText(vis_image, text, (textX, textY),
                                font, fontScale, color, thickness,
                                cv2.LINE_AA)

    text = "Observation RGB"
    textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
    textX = 215 + (360 - textsize[0]) // 2 + 40
    textY = (50 + textsize[1]) // 2
    vis_image = cv2.putText(vis_image, text, (textX, textY),
                            font, fontScale, color, thickness,
                            cv2.LINE_AA)
    
    text = "Occupancy Map"
    textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
    textX = 840 + (480 - textsize[0]) // 2 - 190
    textY = (50 + textsize[1]) // 2
    vis_image = cv2.putText(vis_image, text, (textX, textY),
                            font, fontScale, color, thickness,
                            cv2.LINE_AA)

    text = "Curiousity Map"
    textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
    textX = 1320 + (360 - textsize[0]) // 2 + 60
    textY = (50 + textsize[1]) // 2
    vis_image = cv2.putText(vis_image, text, (textX, textY),
                            font, fontScale, color, thickness,
                            cv2.LINE_AA)
    
    return vis_image
