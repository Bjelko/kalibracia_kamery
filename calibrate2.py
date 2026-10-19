import cv2
import numpy as np
import glob


def load_halcon_points(filename):
    """Load 49 image points from a HALCON point file."""
    points=[]
    with open(filename,"r") as f:
        for line in f:
            cols=line.strip().split()
            if len(cols)!=3: continue
            try:
                points.append([float(cols[0]),float(cols[1]),0.0])
            except ValueError:
                    pass
        points=np.array(points,dtype=np.float32)
        print(f"HALCON points: {len(points)}")
    return points


def order_corners(pts):
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]

    return rect


def find_board(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, bw = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest = max(contours, key=cv2.contourArea)

    rect = cv2.minAreaRect(largest)

    return order_corners(np.float32(cv2.boxPoints(rect)))


def warp_board(img, corners):
    w, h = 1000, 800

    dst = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])

    M = cv2.getPerspectiveTransform(corners, dst)

    return cv2.warpPerspective(img, M, (w, h))


def find_grid_points(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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

    pts = np.array(pts, dtype=np.float32)

    if len(pts) != 49:
        return None

    pts = pts[np.argsort(pts[:, 1])]
    rows = np.array_split(pts, 7)

    ordered = []

    for r in rows:
        r = r[np.argsort(r[:, 0])]
        ordered.extend(r)

    return np.array(ordered, dtype=np.float32).reshape(-1, 1, 2)


def main():

    objp = load_halcon_points("01_2014_257.txt")

    objpoints = []
    imgpoints = []

    files = sorted(glob.glob("calib*.jpg"))

    if len(files) < 4:
        raise RuntimeError("Add at least 4 calibration images")

    img_size = None

    for file in files:

        img = cv2.imread(file)

        try:
            corners = find_board(img)
            warped = warp_board(img, corners)
            pts = find_grid_points(warped)

            if pts is None:
                print(f"SKIP {file}")
                continue

            objpoints.append(objp)
            imgpoints.append(pts)

            img_size = (warped.shape[1], warped.shape[0])

            print(f"OK {file}")

        except Exception as e:
            print(f"ERR {file} : {e}")

    print()
    print(f"Images used: {len(imgpoints)}")

    if len(imgpoints) < 5:
        raise RuntimeError("Not enough valid images")

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera( objpoints, imgpoints, img_size, None, None    )

    np.savez("camera.npz", camera_matrix=K, dist_coeffs=dist, rms=rms)

    print()
    print("RMS =", rms)
    print()
    print("Camera matrix")
    print(K)
    print()
    print("Distortion")
    print(dist)

    np.savetxt("camera_matrix.txt", K, fmt="%.10f")

    np.savetxt("distortion.txt", dist, fmt="%.10f")


if __name__ == "__main__":
    main()
