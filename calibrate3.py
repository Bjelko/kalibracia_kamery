import cv2
import numpy as np
import glob

def load_halcon_points(filename):
    pts=[]
    with open(filename,"r") as f:
        for line in f:
            cols=line.strip().split()
            if len(cols)!=3: continue
            try:
                pts.append([float(cols[0]),float(cols[1]),0.0])
            except ValueError:
                pass
    pts=np.array(pts,dtype=np.float32)
    print(f"HALCON points: {len(pts)}")
    return pts

def find_grid_points(img):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    params=cv2.SimpleBlobDetector_Params()
    params.filterByColor=True
    params.blobColor=0
    params.filterByArea=True
    params.minArea=100
    params.maxArea=10000

    detector=cv2.SimpleBlobDetector_create(params)

    found,centers=cv2.findCirclesGrid(
        gray,
        (7,7),
        flags=cv2.CALIB_CB_SYMMETRIC_GRID,
        blobDetector=detector
    )

    if not found:
        return None

    preview=img.copy()

    for i,(x,y) in enumerate(centers.reshape(-1,2)):
        cv2.circle(preview,(int(x),int(y)),5,(0,0,255),-1)
        cv2.putText(preview,str(i),(int(x)+5,int(y)-5),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,255,0),1)

    return centers,preview

def main():

    objp=load_halcon_points("01_2014_257.txt")

    objpoints=[]
    imgpoints=[]

    files=sorted(glob.glob("calib*.jpg"))

    if len(files)<5:
        raise RuntimeError("Need at least 5 images")

    img_size=None

    for file in files:

        img=cv2.imread(file)

        if img is None:
            continue

        result=find_grid_points(img)

        if result is None:
            print(f"SKIP {file}")
            continue

        pts,preview=result

        cv2.imwrite(
            file.replace(".jpg","_points.png"),
            preview
        )

        objpoints.append(objp)
        imgpoints.append(pts)

        img_size=(img.shape[1],img.shape[0])

        print(f"OK {file}")

    print()
    print(f"Images used: {len(imgpoints)}")

    if len(imgpoints)<5:
        raise RuntimeError("Not enough valid images")

    rms,K,dist,rvecs,tvecs=cv2.calibrateCamera(
        objpoints,
        imgpoints,
        img_size,
        None,
        None
    )

    np.savez(
        "camera.npz",
        camera_matrix=K,
        dist_coeffs=dist,
        rms=rms
    )

    np.savetxt("camera_matrix.txt",K,fmt="%.10f")
    np.savetxt("distortion.txt",dist,fmt="%.10f")

    print("\nRMS =",rms)
    print("\nCamera matrix")
    print(K)
    print("\nDistortion")
    print(dist)

if __name__=="__main__":
    main()