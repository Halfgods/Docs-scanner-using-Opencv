import cv2 as cv
import numpy as np
import argparse
import imutils
from imutils import build_montages
from imutils import paths

def get_four_corners_iterative(contour):
    peri = cv.arcLength(contour, True)
    for eps in np.linspace(0.01, 0.10, 10):
        approx = cv.approxPolyDP(contour, eps * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2)
    return None

def get_min_area_rect(contour):
    rect = cv.minAreaRect(contour)
    box = cv.boxPoints(rect)
    # CRITICAL FIX: Convert float -> int
    return np.int32(box)

def get_convex_hull_approx(contour):
    hull = cv.convexHull(contour)
    peri = cv.arcLength(hull, True)
    approx = cv.approxPolyDP(hull, 0.04 * peri, True)
    if len(approx) == 4:
        return approx.reshape(4, 2)
    return None # Return None if not 4, to stay safe

def get_top_4_farthest(contour):
    M = cv.moments(contour)
    if M["m00"] == 0: return None
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    
    pts = contour.reshape(-1, 2)
    
    # Safety: If contour has fewer than 4 points, return all of them
    if len(pts) < 4:
        return pts
        
    distances = np.sum((pts - np.array([cX, cY]))**2, axis=1)
    top_indices = np.argsort(distances)[::-1][:4]
    return pts[top_indices]

# --- MAIN LOOP ---

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", default="./docs/", help="Path to the folder of images")
args = vars(ap.parse_args())

imagePaths = list(paths.list_images(args["images"]))
processed_images = []

print(f"Processing {len(imagePaths)} images...")

for path in imagePaths:
    try:
        # 1. Read & Resize
        img = cv.imread(path)
        if img is None: continue
        img = imutils.resize(img, height=600)
        final_viz = img.copy()

        # 2. Preprocessing
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(gray, (5,5), 0)
        ret, thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

        # 3. Find Contours
        cnts, _ = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        if len(cnts) > 0:
            cnts = sorted(cnts, key=cv.contourArea, reverse=True)
            largest = cnts[0]

            # --- BLUE: MinAreaRect (Robust Fix) ---
            try:
                box_rect = get_min_area_rect(largest)
                cv.drawContours(final_viz, [box_rect], -1, (255, 0, 0), 2)
            except Exception as e:
                print(f"Blue failed on {path}: {e}")

            # --- RED: Convex Hull ---
            try:
                hull_pts = get_convex_hull_approx(largest)
                if hull_pts is not None:
                    cv.drawContours(final_viz, [hull_pts], -1, (0, 0, 255), 7)
            except Exception as e:
                print(f"Red failed on {path}: {e}")

            # --- YELLOW: Top 4 (Robust Fix) ---
            try:
                farthest_pts = get_top_4_farthest(largest)
                if farthest_pts is not None:
                    for pt in farthest_pts:
                        # CRITICAL FIX: Cast to standard Python int for drawing
                        pt_tuple = (int(pt[0]), int(pt[1]))
                        cv.circle(final_viz, pt_tuple, 8, (0, 255, 255), -1)
            except Exception as e:
                print(f"Yellow failed on {path}: {e}")

            # --- GREEN: Iterative ---
            try:
                iter_pts = get_four_corners_iterative(largest)
                if iter_pts is not None:
                    cv.drawContours(final_viz, [iter_pts], -1, (0, 255, 0), 2)
            except Exception as e:
                print(f"Green failed on {path}: {e}")

        processed_images.append(final_viz)
    
    except Exception as e:
        print(f"CRITICAL ERROR processing {path}: {e}")

# --- BUILD MONTAGE ---
if len(processed_images) > 0:
    montages = build_montages(processed_images, (300 , 300), (5, 5))
    for i, montage in enumerate(montages):
        cv.imshow(f"Montage {i+1}", montage)
    cv.waitKey(0)
    cv.destroyAllWindows()
else:
    print("No images processed!")