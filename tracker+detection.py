
import dlib
import cv2
import argparse as ap
import get_points

vehicle_cascade = cv2.CascadeClassifier('vehicles.xml')
fullbody_cascade = cv2.CascadeClassifier('fullbody.xml')
def run(source=0, dispLoc=False):
    
    cam = cv2.VideoCapture(source)
    cam.set(3, 320)

    cam.set(4, 240)

    
    if not cam.isOpened():
        print ("Video device or file couldn't be opened")
        exit()
    
    print ("Press key `p` to pause the video to start tracking")
    while True:
        
        retval, img = cam.read()
        if not retval:
            print ("Cannot capture frame device")
            exit()
        if(cv2.waitKey(10)==ord('p')):
            break
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.imshow("Image", img)
    cv2.destroyWindow("Image")

    
    points = get_points.run(img) 
    print(points)
    if not points:
        print ("ERROR: No object to be tracked.")
        exit()
    
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Image", img)

    tracker = dlib.correlation_tracker()
    
    tracker.start_track(img, dlib.rectangle(*points[0]))

    while True:
        retval, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        fullbody = fullbody_cascade.detectMultiScale(gray, 1.3, 3)
        vehicles = vehicle_cascade.detectMultiScale(gray, 1.3, 3)
        font = cv2.FONT_HERSHEY_SIMPLEX
        for (vx, vy, vw, vh) in vehicles:
            cv2.putText(img, 'Vehicle', (vx, vy-3), font, 0.5, (0,0,255), 2, cv2.LINE_AA)
            cv2.rectangle(img, (vx,vy), (vx+vw, vy+vh), (0,0,255), 2)
        
        for(x,y,w,h) in fullbody:
            cv2.putText(img, 'Human', (x, y-3), font, 0.5, (255,0,0), 2, cv2.LINE_AA)
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
                   
        cv2.imshow('Image', img)        
        
        if not retval:
            print ("Cannot capture frame device | CODE TERMINATING :(")
            exit()
        
        tracker.update(img)
        
        rect = tracker.get_position()
        pt1 = (int(rect.left()), int(rect.top()))
        pt2 = (int(rect.right()), int(rect.bottom()))
        cv2.rectangle(img, pt1, pt2, (255, 255, 255), 3)
        print ("Object tracked at [{}, {}] \r").format(pt1, pt2),
        if dispLoc:
            loc = (int(rect.left()), int(rect.top()-20))
            txt = "Object tracked at [{}, {}]".format(pt1, pt2)
            cv2.putText(img, txt, loc , cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 1)
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.imshow("Image", img)
        
        if cv2.waitKey(1) == 27:
            break

   
    cam.release()

if __name__ == "__main__":
    
    parser = ap.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-d', "--deviceID", help="Device ID")
    group.add_argument('-v', "--videoFile", help="Path to Video File")
    parser.add_argument('-l', "--dispLoc", dest="dispLoc", action="store_true")
    args = vars(parser.parse_args())

    if args["videoFile"]:
        source = args["videoFile"]
    else:
        source = int(args["deviceID"])
    run(source, args["dispLoc"])
