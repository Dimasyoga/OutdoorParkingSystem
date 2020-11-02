import numpy as np
import cv2
from numpy.linalg import lstsq

# ============================================================================

FINAL_LINE_COLOR = (0, 255, 0)
WORKING_LINE_COLOR = (255, 0, 0)

# ============================================================================


class Preprocessing(object):
    def __init__(self):
        self.ori_img = None

        self.done = False # Flag signalling we're done
        self.current = (0, 0) # Current position, so we can draw the line-in-progress
        self.points = [] # List of points defining our polygon
        self.maskBox = []
        self.clickCount = 0

    def on_mouse(self, event, x, y, buttons, user_param):
        # Mouse callback that gets called for every mouse event (i.e. moving, clicking, etc.)

        if event == cv2.EVENT_MOUSEMOVE:
            # We want to be able to draw the line-in-progress, so update current mouse position
            self.current = (x, y)

        elif event == cv2.EVENT_LBUTTONDOWN:
            # Left click means adding a point at current position to the list of points
            print("Adding point #%d with position(%d,%d)" % (len(self.points), x, y))
            # self.points.append((x, y))
            if (self.clickCount == 0):
                self.points.append((x, y))
                self.clickCount += 1
            elif (self.clickCount == 1):
                self.points.append((x, y))
                self.maskBox.append(self.points)
                self.points = []
                self.clickCount = 0

    def addMask(self):
        # Let's create our working window and set a mouse callback to handle events
        cv2.namedWindow("Editing MaskBox", flags=cv2.WINDOW_AUTOSIZE)
        clone = self.ori_img.copy()
        cv2.waitKey(1)
        cv2.setMouseCallback("Editing MaskBox", self.on_mouse)

        if (len(self.maskBox) > 0):
            for box in self.maskBox:
                cv2.rectangle(clone, box[0], box[1], FINAL_LINE_COLOR, 1)

        while(True):
            # This is our drawing loop, we just continuously draw new images
            # and show them in the named window
            print("maskBox: ", self.maskBox)
            print("position", self.points)
            
            if (len(self.points) > 0):
                clone = self.ori_img.copy()
                if (len(self.maskBox) > 0):
                    for box in self.maskBox:
                        cv2.rectangle(clone, box[0], box[1], FINAL_LINE_COLOR, 1)    

                cv2.rectangle(clone, self.points[0], self.current, WORKING_LINE_COLOR, 1)

            # Update the window
            cv2.imshow("Editing MaskBox", clone)
            # And wait 50ms before next iteration (this will pump window messages meanwhile)
            # if cv2.waitKey(50) == 27: # ESC hit
            #     self.done = True

            key = cv2.waitKey(1)
            if (key == ord("r")):
                clone = self.ori_img.copy()
                if (len(self.maskBox) > 0):
                    self.maskBox.pop()
                
                for box in self.maskBox:
                    cv2.rectangle(clone, box[0], box[1], FINAL_LINE_COLOR, 1)
            
            elif (key == ord("c")):
                break
            elif (key == -1):
                continue

        cv2.destroyWindow("Editing MaskBox")
    
    def removeMask(self):
        cv2.namedWindow("Removing MaskBox", flags=cv2.WINDOW_AUTOSIZE)
        cv2.waitKey(1)
        active = None

        while (True):
            
            clone = self.ori_img.copy()

            if (len(self.maskBox) > 0) or not (active == None):
                
                for box in self.maskBox:
                    cv2.rectangle(clone, box[0], box[1], FINAL_LINE_COLOR, 1)
                
                if not (active == None):
                    cv2.rectangle(clone, active[0], active[1], WORKING_LINE_COLOR, 1)

                key = cv2.waitKey(1)

                if (key == ord("s")):
                    if (active == None):
                        active = self.maskBox.pop(0)
                    else:
                        self.maskBox.append(active)
                        active = self.maskBox.pop(0)

                elif (key == ord("r")) and not (active == None):
                    active = None
                elif (key == ord("c")):
                    if not (active == None):
                        self.maskBox.append(active)
                    break
            else:
                print("MaskBox is empty")
                break
            
            cv2.imshow("Removing MaskBox", clone)
            # print("Mask: ", self.maskBox)
        
        cv2.destroyWindow("Removing MaskBox")

    def getCrop(self):
        crop = []
        if (len(self.maskBox) > 0):
            for box in self.maskBox:
                cropped = self.ori_img[box[0][1]:box[1][1], box[0][0]:box[1][0]]
                crop.append(cropped)
        
        return crop
    
    def setCloudMask(self, message):

        self.maskBox = []

        if message["path"] == '/':

            # messageSorted = sorted(message["data"])
            for value in message["data"].values():
                mask = []
                points = value.split(';')
                for pt in points:
                    point = pt.split(',')
                    mask.append((int(point[0]), int(point[1])))
                
                self.maskBox.append(mask)
        
        # print(self.maskBox)

    def getCloudMask(self):
        mask = {}

        for idx, box in enumerate(self.maskBox):
            mask['slot_'+str(idx)] = str(box[0][0])+','+str(box[0][1])+';'+str(box[1][0])+','+str(box[1][1])+';'+str(box[2][0])+','+str(box[2][1])+';'+str(box[3][0])+','+str(box[3][1])
        
        return mask

    def getMask(self):
        return self.maskBox
    
    def setMask(self, mask):
        self.maskBox = mask

    def setImage(self, img):
        self.ori_img = img


# if __name__ == "__main__":
#     image = cv2.imread('C:/Users/Dim/Pictures/Prambanan.jpg')
#     image = cv2.resize(image, (1024, 680))
#     prepro = Preprocessing()
#     prepro.setImage(image)
#     prepro.addMask()
#     prepro.removeMask()
#     img_crop = prepro.getCrop()
#     for idx, img in enumerate(img_crop):
#         cv2.imshow(str(idx), img)
    
#     cv2.waitKey()
#     cv2.destroyAllWindows()