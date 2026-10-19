import cv2
import numpy as np
import sys


def load_halcon_points(filename):
    pts = []
    with open(filename, "r") as f:
        for line in f:
            cols = line.strip().split()
            if len(cols) != 3:
                continue
            try:
                pts.append([float(cols[0]), float(cols[1]), 0.0])
            except ValueError:
                pass
    return np.array(pts, dtype=np.float32)

def find_board(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest)
    return np.float32(cv2.boxPoints(rect))

def order_points(pts):
    pts = np.array(pts, dtype=np.float32)
    pts = pts[np.argsort(pts[:, 1])]
    rows = np.array_split(pts, 7)
    ordered = []
    for row in rows:
        row = row[np.argsort(row[:, 0])]
        ordered.extend(row)
    return np.array(ordered, dtype=np.float32)

def find_grid_points(img):
    corners = find_board(img)
    dst = np.float32([[0, 0], [999, 0], [999, 799], [0, 799]])
    M = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(img, M, (1000, 800))
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    pts = []
    for c in contours:
        area = cv2.contourArea(c)
        if not 300 < area < 3000:
            continue
        m = cv2.moments(c)
        if m["m00"] == 0:
            continue
        pts.append([m["m10"] / m["m00"], m["m01"] / m["m00"]])
    if len(pts) != 49:
        return None, None
    pts = order_points(pts)
    Minv = np.linalg.inv(M)
    pts = np.array([pts], dtype=np.float32)
    pts = cv2.perspectiveTransform(pts, Minv)[0]
    return pts.reshape(-1, 1, 2), warped

def undistort_image(img, K, dist):
    h, w = img.shape[:2]
    newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
    return cv2.undistort(img, K, dist, None, newK)

def draw_axes(img, K, dist, rvec, tvec, length=0.05):
    axis = np.float32([[0, 0, 0], [length, 0, 0], [0, length, 0], [0, 0, -length]])
    imgpts, _ = cv2.projectPoints(axis, rvec, tvec, K, dist)
    p0 = tuple(np.int32(imgpts[0][0]))
    px = tuple(np.int32(imgpts[1][0]))
    py = tuple(np.int32(imgpts[2][0]))
    pz = tuple(np.int32(imgpts[3][0]))
    cv2.line(img, p0, px, (0, 0, 255), 2)
    cv2.line(img, p0, py, (0, 255, 0), 2)
    cv2.line(img, p0, pz, (255, 0, 0), 2)

def main():
    image_file = sys.argv[1] if len(sys.argv) > 1 else "real_image.jpg"
    data = np.load("camera.npz")
    K = data["camera_matrix"]
    dist = data["dist_coeffs"]
    obj_points = load_halcon_points("01_2014_257.txt")
    img = cv2.imread(image_file)
    if img is None:
        raise RuntimeError(image_file)
    image_points, warped = find_grid_points(img)
    if image_points is None:
        raise RuntimeError("49 points not found")
    ok, rvec, tvec = cv2.solvePnP(obj_points, image_points, K, dist)
    if not ok:
        raise RuntimeError("solvePnP failed")
    z=float(tvec[2,0])
    undist = undistort_image(img, K, dist)
    draw_axes(undist, K, dist, rvec, tvec)
    cv2.putText(        undist,        f"Distance Z = {z:.3f} m",        (20, 40),        cv2.FONT_HERSHEY_SIMPLEX,        1,        (0, 255, 0),        2,    )
    cv2.imwrite("undistorted.png", undist)
    preview = undist.copy()
    h, w = preview.shape[:2]
    scale = min(1600 / w, 900 / h)
    preview = cv2.resize(preview, None, fx=scale, fy=scale)
    print()
    print(f"Distance Z = {z:.3f} m")
    print()
    print("Position:")
    print(tvec.flatten())
    print()
    print("Rotation:")
    print(rvec.flatten())
    print()
    print("Saved: undistorted.png")

    cv2.imshow("measure", preview)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
