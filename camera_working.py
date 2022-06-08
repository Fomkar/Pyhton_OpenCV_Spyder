#%%
# importing libraries
from pypylon import pylon
import cv2
import numpy as np

# Pypylon get camera by serial number
serial_number = '40038474'
info = None
for i in pylon.TlFactory.GetInstance().EnumerateDevices():
    if i.GetSerialNumber() == serial_number:
        info = i
        break
else:
    print('Camera with {} serial number not found'.format(serial_number))

# VERY IMPORTANT STEP! To use Basler PyPylon OpenCV viewer you have to call .Open() method on you camera
if info is not None:
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(info))
    camera.Open()
    

    
# Grabing Continusely (video) with minimal delay
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly) 
camera.AcquisitionFrameRateAbs.SetValue(True)
# camera.AcquisitionFrameRateEnable.SetValue = True
camera.AcquisitionFrameRateAbs.SetValue(30.0)
# camera.AcquisitionFrameRateAbs.SetValue = 5.0
# camera.Width.SetValue(720)
# camera.Height.SetValue = 540.0
# camera.Width.SetValue = 720.0
# camera.Width = 720
converter = pylon.ImageFormatConverter()

# converting to opencv bgr format
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned


images = np.zeros((1000, 1080, 1440, 3), dtype=int)
# images = np.zeros((100, 540, 720, 3), dtype=int)

counter = 0
while camera.IsGrabbing():
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
    
    
    
    if grabResult.GrabSucceeded():
        # Access the image data
        image = converter.Convert(grabResult)
        img = image.GetArray()
        # Get grabbed image
        
        if counter < 1000:
            images[counter] = img
        else:
            break
        
        counter += 1
        
        # if counter <= 100:
        #     cv2.imwrite('C:/Users/Hasan/Desktop/test/Spyder/temp_' + str(counter) + '.jpg', img)
        cv2.namedWindow('title', cv2.WINDOW_NORMAL)
        cv2.imshow('title', img)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    grabResult.Release()
    
# Releasing the resource    
camera.StopGrabbing()
# camera.Release()
camera.Close()
cv2.destroyAllWindows()


#%%
for i in range(1000):
    
    # Get grabbed image
    img = images[i]
    cv2.imwrite('C:/Users/path...' + str(i) + '.jpg', img)
    


