import cv2
import numpy as np

def load_halcon_points(filename):
    points=[]
    with open(filename,"r") as f:
        for line in f:
            cols=line.strip().split()
            if len(cols)!=3: continue
            try:
                points.append([float(cols[0]),float(cols[1])])
            except ValueError:
                pass
    points=np.array(points,dtype=np.float32)
    print(f"HALCON points: {len(points)}")
    return points

def order_corners(pts):
    rect=np.zeros((4,2),dtype=np.float32)
    s=pts.sum(axis=1)
    d=np.diff(pts,axis=1)
    rect[0]=pts[np.argmin(s)]
    rect[2]=pts[np.argmax(s)]
    rect[1]=pts[np.argmin(d)]
    rect[3]=pts[np.argmax(d)]
    return rect

def find_board(img):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _,bw=cv2.threshold(gray,100,255,cv2.THRESH_BINARY_INV)
    contours,_=cv2.findContours(bw,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    largest=max(contours,key=cv2.contourArea)
    rect=cv2.minAreaRect(largest)
    box=cv2.boxPoints(rect)
    return order_corners(np.float32(box))

def warp_board(img,corners):
    w,h=1000,800
    dst=np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])
    M=cv2.getPerspectiveTransform(corners,dst)
    warped=cv2.warpPerspective(img,M,(w,h))
    return warped,M

def find_grid_points(warped):
    gray=cv2.cvtColor(warped,cv2.COLOR_BGR2GRAY)
    _,bw=cv2.threshold(gray,120,255,cv2.THRESH_BINARY_INV)
    contours,_=cv2.findContours(bw,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    centers=[]
    for c in contours:
        area=cv2.contourArea(c)
        if not 300<area<3000: continue
        m=cv2.moments(c)
        if m["m00"]==0: continue
        centers.append([
            m["m10"]/m["m00"],
            m["m01"]/m["m00"]
        ])
    centers=np.array(centers,dtype=np.float32)
    print(f"Detected blobs: {len(centers)}")
    if len(centers)!=49:
        raise RuntimeError(f"Expected 49 points, found {len(centers)}")
    centers=centers[np.argsort(centers[:,1])]
    rows=np.array_split(centers,7)
    ordered=[]
    for row in rows:
        row=row[np.argsort(row[:,0])]
        ordered.extend(row)
    return np.array(ordered,dtype=np.float32)

def save_calibration(H,M,filename):
    np.savez(
        filename,
        H=H,
        Hinv=np.linalg.inv(H),
        M=M,
        Minv=np.linalg.inv(M)
    )

def main():
    image_file="calib_image.jpg"
    halcon_file="01_2014_257.txt"
    world_points=load_halcon_points(halcon_file)
    img=cv2.imread(image_file)
    if img is None:
        raise RuntimeError(image_file)
    corners=find_board(img)
    warped,M=warp_board(img,corners)
    image_points=find_grid_points(warped)
    H,_=cv2.findHomography(image_points,world_points)
    save_calibration(H,M,"korekcia.npz")
    np.savetxt("homography.txt",H,fmt="%.10f")
    with open("korekcia.txt","w") as f:
        f.write("Homography\n")
        for row in H:
            f.write(" ".join(f"{v:.10f}" for v in row)+"\n")
        f.write("\nPerspectiveMatrix\n")
        for row in M:
            f.write(" ".join(f"{v:.10f}" for v in row)+"\n")
    preview=warped.copy()
    for i,(x,y) in enumerate(image_points):
        cv2.circle(preview,(int(x),int(y)),5,(0,0,255),-1)
        cv2.putText(
            preview,
            str(i),
            (int(x)+5,int(y)-5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0,255,0),
            1
        )

    cv2.imwrite("preview_points.png",preview)
    print("Saved:")
    print("  korekcia.npz")
    print("  homography.txt")
    print("  korekcia.txt")
    print("  preview_points.png")

if __name__=="__main__":
    main()