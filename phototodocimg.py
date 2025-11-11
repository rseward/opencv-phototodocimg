#!/usr/bin/env python

"""
Simple Document Page Scanner Project based on OpenCV

The project assumes a photo of a single A4 page in a photo mostly orientated with the top of the page up. 

It will find the page and extract it into an image of it's own.

You could use this image for further processing such as OCR feeding it to an AI or whatever.

Author: rseward@bluestone-consulting.com
Date: 2025-11-11
"""

import cv2
import numpy as np
import sys
import click

A4_HEIGHT_TO_WIDTH_RATIO = (297 / 210)
debug = False

def estimate_dims(perimeter):
    """Estimate the dimensions of the document from the perimeter.
       Here the assumptions that the page is a A4 page is most important as it estimates the
       dimensions of the page based on the perimeter and this assumption.
    """

    # Estimate the dimensions of the document
    # p = 2*w + 2*h
    # h = p/2 - w
    # w = A4_HEIGHT_TO_WIDTH_RATIO * h
    # h = p/2 - A4_HEIGHT_TO_WIDTH_RATIO * h
    # h + A4_HEIGHT_TO_WIDTH_RATIO * h = p/2
    # h * (1 + A4_HEIGHT_TO_WIDTH_RATIO) = p/2
    # h = p / (2 * (1 + A4_HEIGHT_TO_WIDTH_RATIO))
    h = perimeter / (2 * (1 + A4_HEIGHT_TO_WIDTH_RATIO))
    w = h / A4_HEIGHT_TO_WIDTH_RATIO 
    return (w, h)


def order_points(pts):
    """Re-order the page points to be in the order of nw, sw, se, ne. This ordering assumes
       the page in the image is orientated with top up mostly in the photo."""
    # pts is a numpy array of 4 points
    # Initialize a list of coordinates that will be ordered
    # such that the first entry is the top-left,
    # the second is the top-right, the third is the bottom-right,
    # and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    
    # The top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    nw = pts[np.argmin(s)]  # nw - top-left
    se = pts[np.argmax(s)]  # se - bottom-right
    
    # The top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    ne = pts[np.argmin(diff)]  # ne - top-right
    sw = pts[np.argmax(diff)]  # sw - bottom-left
    
    rect = np.array([nw, sw, se, ne], np.int32)
    return rect


def find_page_rect_in_image(img):
    """Assume a single page in the image at this point. This function returns the approximate 
       geometric rectangle of the page within the the image."""

    global debug

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Binarize the image using OTSU's method
    binimg = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    if debug:
        cv2.imshow('Binary', binimg)
        cv2.waitKey(0)

    # Use findContour to fine the contours of the document
    contours, _ = cv2.findContours(binimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    if len(contours) == 0:
        raise Exception("❌ No page contours found! :-(")

    # Get the largest contour
    contour = contours[0]

    # Use approxPolyDP to convert the contour to a rectangle
    #pts = np.array([nw, sw, se, ne], np.int32)
    peri = cv2.arcLength(contour, True)
    epsilon = 0.02 * peri
    approx = cv2.approxPolyDP(contour, epsilon, True)

    if debug:
        output = img.copy()
        approx_pts = [np.array(approx, np.int32)]
        print(f"approx_pts: {approx_pts}")
        cv2.polylines(output, approx_pts, True, (0, 255, 0), 2)
        cv2.imshow('Image with page outline', output)
        cv2.waitKey(0)

    assert approx.shape[0] == 4, "❌ Could not find a 4-point polygon! :-(" 

    approx_rect = order_points(approx.reshape(4, 2))

    if debug:
        output = img.copy()
        cv2.polylines(output, [approx_rect.astype(np.int32)], True, (0, 255, 0), 2)
        cv2.imshow('Image with page outline', output)
        cv2.waitKey(0)

    print(f"approx_rect: {approx_rect}")

    return approx_rect, peri


def process_image(img):
    approx, peri = find_page_rect_in_image(img)

    (w, h) = estimate_dims(peri)
    print(f"w: {w}, h: {h}")

    # Calculate the homography using 4 point correspondences.
    src_pts = approx.reshape(4,1,2).astype(np.float32)
    dst_pts = np.array([[0, 0], [0, h], [w, h], [w, 0]], dtype=np.float32)

    homography = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # Apply the homography
    output = cv2.warpPerspective(img, homography, (int(w), int(h)))

    # Display the image
    cv2.imshow('Image', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite('page.jpg', output)

    print("✅ Done! One page found!")


@click.command()
@click.option('--image', default='scanned-form.jpg', required=False, help='Input image path')
@click.option('--debug/--no-debug', 'ab_debug', default=False, required=False, help='Debug mode')
def main(image, ab_debug):
    global debug
    debug = ab_debug
    # Load the image
    img = cv2.imread(image)

    process_image(img)

"""
Test points for debugging: scanned-form.jpg

nw = (38, 298)
sw = (171, 1280)
se = (947, 1118)
ne = (742, 196)

cv2.circle(output, nw, 5, (0, 255, 0), 2)
cv2.circle(output, sw, 5, (0, 255, 0), 2)
cv2.circle(output, se, 5, (0, 255, 0), 2)
cv2.circle(output, ne, 5, (0, 255, 0), 2)

cv2.imshow('Image', output)
cv2.waitKey(0)
#cv2.destroyAllWindows()
#sys.exit(1)
"""


if __name__ == '__main__':
    main()

