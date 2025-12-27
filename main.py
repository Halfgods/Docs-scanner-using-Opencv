import cv2
import numpy as np
import argparse
import imutils

# --- 1. THE PERSPECTIVE TRANSFORM MATH (Standard) ---
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # Top-left
    rect[2] = pts[np.argmax(s)] # Bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # Top-right
    rect[3] = pts[np.argmax(diff)] # Bottom-left
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

# --- 2. AUTO-DETECTION (Green Logic) ---
def get_initial_corners(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(cnts) > 0:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        largest = cnts[0]
        peri = cv2.arcLength(largest, True)
        
        # Try to find 4 corners automatically
        for eps in np.linspace(0.01, 0.10, 10):
            approx = cv2.approxPolyDP(largest, eps * peri, True)
            if len(approx) == 4:
                return approx.reshape(4, 2)
                
    # Fallback: If auto-detection fails, return corners of the image with margin
    h, w = img.shape[:2]
    return np.array([[50, 50], [w-50, 50], [w-50, h-50], [50, h-50]])

# --- 3. MOUSE INTERACTION LOGIC ---
# Global variables to track state
doc_pts = None # Will store the 4 corners
selected_point_idx = -1 # Which point are we dragging? (-1 means none)

def mouse_callback(event, x, y, flags, param):
    global doc_pts, selected_point_idx

    # A. MOUSE DOWN: Did we click near a corner?
    if event == cv2.EVENT_LBUTTONDOWN:
        for i, (px, py) in enumerate(doc_pts):
            # Calculate distance from mouse to this corner
            dist = np.sqrt((x - px)**2 + (y - py)**2)
            if dist < 20: # Hit radius (if click is within 20px)
                selected_point_idx = i
                break

    # B. MOUSE MOVE: If dragging, update the coordinate
    elif event == cv2.EVENT_MOUSEMOVE:
        if selected_point_idx != -1:
            doc_pts[selected_point_idx] = [x, y]

    # C. MOUSE UP: Stop dragging
    elif event == cv2.EVENT_LBUTTONUP:
        selected_point_idx = -1

# --- MAIN LOOP ---
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", default="./docs/hello.png", help="Path to image")
args = vars(ap.parse_args())

# Load and resize image
img_original = cv2.imread(args["image"])

# Resize for easier screen fit (keep aspect ratio logic for real apps usually)
img = imutils.resize(img_original, height=600) 
img = imutils.rotate_bound(img , +90)
# 1. Get Initial Points (Auto)
doc_pts = get_initial_corners(img)

# 2. Setup Window and Callback
cv2.namedWindow("Adjust Corners")
cv2.setMouseCallback("Adjust Corners", mouse_callback)

print("Instructions:")
print(" - Drag the GREEN corners to adjust.")
print(" - Press 's' to Scan/Crop.")
print(" - Press 'r' to Reset.")
print(" - Press 'q' to Quit.")

while True:
    # Always draw on a fresh copy so lines don't smear
    display_img = img.copy()

    # Draw the Polygon (Connecting lines)
    # Convert points to shape (1, 4, 2) for polyLines
    poly_pts = doc_pts.reshape((-1, 1, 2)) 
    cv2.polylines(display_img, [poly_pts], True, (0, 255, 0), 2)

    # Draw the Corners (Circles)
    for i, point in enumerate(doc_pts):
        color = (0, 0, 255) if i == selected_point_idx else (0, 255, 0)
        cv2.circle(display_img, tuple(point), 10, color, -1)

    cv2.imshow("Adjust Corners", display_img)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"): # Scan
        # We need float32 for the transform function
        warped = four_point_transform(img, doc_pts.astype("float32"))
        cv2.imshow("Scanned Result", warped)
    
    elif key == ord("r"): # Reset
        doc_pts = get_initial_corners(img)

    elif key == ord("q"):
        break

cv2.destroyAllWindows()