import numpy as np
import cv2
from numpy.linalg import lstsq

# ============================================================================

FINAL_LINE_COLOR = (0, 255, 0)
WORKING_LINE_COLOR = (255, 0, 0)
font = cv2.FONT_HERSHEY_SIMPLEX

# ============================================================================


class Preprocessing(object):
    """
    A class used to process image and masking

    Attributes
    ----------
    ori_img : cv2 mat
        a cv2 image for reference image
    done : bool
        a flag for editing process
    current : tuple
        the current position of mouse
    points : list of tuple
        list of point defining rectangular
    maskBox : list of rectangular
        list of rectangular defining our masking parameter
    clickCount: int
        mouse click count

    """
    def __init__(self):
        self.ori_img = None

        self.done = False # Flag signalling we're done
        self.current = (0, 0) # Current position, so we can draw the line-in-progress
        self.points = [] # List of points defining our polygon
        self.maskBox = []
        self.clickCount = 0

    def on_mouse(self, event, x, y):
        """Mouse callback that gets called for every mouse event (i.e. moving, clicking, etc.)

        Parameters
        ----------
        event : cv2 event
        x : mouse x coordinate
        y : mouse y coordinate

        """

        if event == cv2.EVENT_MOUSEMOVE:
            # We want to be able to draw the line-in-progress, so update current mouse position
            self.current = (x, y)

        elif event == cv2.EVENT_LBUTTONDOWN:
            # Left click means adding a point at current position to the list of points
            print("Adding point #%d with position(%d,%d)" % (len(self.points), x, y))
            if (self.clickCount == 0):
                self.points.append((x, y))
                self.clickCount += 1
            elif (self.clickCount == 1):
                self.points.append((x, y))
                self.maskBox.append(self.points)
                self.points = []
                self.clickCount = 0

    def addMask(self):
        """Create working windows to draw mask with image that already set before,
            setImage() need to be called first.
        """

        # create working window and set a mouse callback to handle events
        cv2.namedWindow("Editing MaskBox", flags=cv2.WINDOW_AUTOSIZE)
        clone = self.ori_img.copy()
        cv2.waitKey(1)
        cv2.setMouseCallback("Editing MaskBox", self.on_mouse)

        if (len(self.maskBox) > 0):
            for box in self.maskBox:
                cv2.rectangle(clone, box[0], box[1], FINAL_LINE_COLOR, 1)

        while(True):
            # drawing loop
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
        """Create working windows to remove mask that has been set"""

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
        
        cv2.destroyWindow("Removing MaskBox")

    def clahe(img):
        """Apply clahe (Contrast Limited Adaptive histogram equalization) to enhance image contrast 

        Parameters
        ----------
        img : cv2 mat
            The image to be processed

        Returns
        -------
        img : cv2 mat
            The image result
        """

        #-----Converting image to LAB Color model----------------------------------- 
        lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        #-----Splitting the LAB image to different channels-------------------------
        l, a, b = cv2.split(lab)

        #-----Applying CLAHE to L-channel-------------------------------------------
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)

        #-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
        limg = cv2.merge((cl,a,b))

        #-----Converting image from LAB Color model to RGB model--------------------
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        return final

    def getCrop(self):
        """Get list of cropped image from masking parameter and set image

        Returns
        -------
        crop : list of cv2 mat
            The list of cropped image
        """

        crop = []
        if (len(self.maskBox) > 0):
            for box in self.maskBox:
                try:
                    cropped = self.ori_img[box[0][1]:box[1][1], box[0][0]:box[1][0]]
                    cropped = self.clahe(cropped)
                except:
                    cropped = np.ones((150, 150, 3), dtype="uint8")
                crop.append(cropped)
        
        return crop

    def getMask(self):
        """Get masking parameter

        Returns
        -------
        maskBox list : list of tupple
            The list of rectangular coordinate of masking box
        """
        return self.maskBox
    
    def setMask(self, mask):
        """Set masking parameter

        Parameters
        ----------
        mask : list of masking box 
            List of rectangular coordinate of masking box
        """
        self.maskBox = mask

    def setImage(self, img):
        """Set reference image

        Parameters
        ----------
        img : cv2 mat
            The image to be processed
        """
        self.ori_img = img

    def getImage_masked(self, pred):
        """Get reference image with masking box label

        Parameters
        ----------
        pred : List of prediction from inference model
            The list order must match with masking parameter

        Returns
        -------
        frame : cv2 mat
            The image with masking box label
        """
        frame = self.ori_img.copy()

        if (len(self.maskBox) > 0) and (len(pred) > 0):
            for idx, box in enumerate(self.maskBox):
                cv2.rectangle(frame, box[0], box[1], FINAL_LINE_COLOR, 1)
            
                if pred[idx]==0:
                    cv2.putText(frame,'Kosong', box[0], font, 0.5,(180, 0, 0),2)
                    cv2.putText(frame,'Kosong', box[0], font, 0.5,(255, 0, 0),1)
                elif pred[idx]==1:
                    cv2.putText(frame,'Isi', box[0], font, 0.5,(180, 0, 0),2)
                    cv2.putText(frame,'Isi', box[0], font, 0.5,(255, 0, 0),1)
                else:
                    cv2.putText(frame,'Dipesan', box[0], font, 0.5,(180, 0, 0),2)
                    cv2.putText(frame,'Dipesan', box[0], font, 0.5,(255, 0, 0),1)

        return frame

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