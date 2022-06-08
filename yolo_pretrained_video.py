import cv2
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# from imutils.video import FPS

from pypylon import pylon

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
camera.AcquisitionFrameRateAbs.SetValue(10.0)
# camera.AcquisitionFrameRateAbs.SetValue = 5.0
# camera.Width.SetValue(720)
# camera.Height.SetValue = 540.0
# camera.Width.SetValue = 720.0
# camera.Width = 720
converter = pylon.ImageFormatConverter()

# converting to opencv bgr format
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned



# fps = FPS().start()

#cap=cv2.VideoCapture("images/video.mp4")
# cap=cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH,600)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT,600)

cfg="yolov3.cfg"
weights="yolov3.weights"
model=cv2.dnn.readNetFromDarknet(cfg,weights)

model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

layers=model.getLayerNames()
output_layer=[layers[layer-1]  for layer in model.getUnconnectedOutLayers()] # Modelde ki çıktı katmanlarını aldık.



while True:
    # ret,frame=cap.read()

    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
    
    if grabResult.GrabSucceeded():
        # Access the image data
        image = converter.Convert(grabResult)
        img = image.GetArray()
        # Get grabbed image

    img_width=img.shape[1]
    img_height=img.shape[0]

    frame_blob = cv2.dnn.blobFromImage(img,1/255,(416,416),swapRB=True,crop=False)  # Görüntüyü 4 boyutlu tensöre çevirme işlemi.

    labels = ["insan","bisiklet","araba","motorcycle","airplane","bus","train","truck","boat",
              "trafficlight","firehydrant","stopsign","parkingmeter","bench","bird","cat",
              "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
              "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sportsball",
              "kite","baseballbat","baseballglove","skateboard","surfboard","tennisracket",
              "su_sisesi","wineglass","bardak","fork","knife","spoon","bowl","banana","apple",
              "sandwich","orange","broccoli","carrot","hotdog","pizza","donut","cake","chair",
              "sofa","pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse",
              "remote","keyboard","cellphone","microwave","oven","toaster","sink","refrigerator",
              "kitap","clock","vase","makas","teddybear","hairdrier","toothbrush"]


    colors=["0,255,255","255,0,0","255,255,0","0,255,0","0,0,0","255,255,255"]
    colors=[np.array(color.split(",")).astype("int") for color in colors]
    colors=np.array(colors) # Tek bir array de tuttuk.
    colors=np.tile(colors,(18,1)) # Büyütme işlemi yapıyoruz.

    ##

    model.setInput(frame_blob)

    detection_layers=model.forward(output_layer)

    #----------- Non Maximum Supression Operation-1 ----------
    ids_list=[]
    boxes_list=[]
    confidence_list=[]
    #------------ End Of Opertation 1 -------------

    for detection_layer in detection_layers:
        for object_detection in detection_layer:
            scores=object_detection[5:]
            predicted_id=np.argmax(scores)
            confidence=scores[predicted_id]
            if confidence > 0.30:
                label=labels[predicted_id]
                # if label == "insan":
                #     os.system(file)
                #     cv2.waitKey(1000);
                bounding_box=object_detection[0:4] * np.array([img_width,img_height,img_width,img_height])
                (box_center_x,box_center_y,box_width,box_height)=bounding_box.astype("int")

                start_x=int(box_center_x-(box_width/2))
                start_y =int(box_center_y - (box_height / 2))

                # ----------- Non Maximum Supression Operation-2 ----------
                ids_list.append(predicted_id)
                confidence_list.append(float(confidence))
                boxes_list.append([start_x,start_y,int(box_width),int(box_height)])
                # ------------ End Of Opertation 2 -------------

    # ----------- Non Maximum Supression Operation-3 ----------
    max_ids=cv2.dnn.NMSBoxes(boxes_list,confidence_list,0.5,0.4)

    for max_id in max_ids:
        max_class_id = max_id
        box = boxes_list[max_class_id]

        start_x = box[0]
        start_y = box[1]
        box_width = box[2]
        box_height = box[3]

        predicted_id = ids_list[max_class_id]
        label = labels[predicted_id]
        confidence = confidence_list[max_class_id]
        # ------------ End Of Opertation 3 -------------

        end_x = start_x + box_width
        end_y = start_y + box_height

        box_color = colors[predicted_id]
        box_color = [int(each) for each in box_color]

        label = "{}: {:.2f}%".format(label, confidence * 100)
        print("Predicted_object: ", label)

        cv2.rectangle(img, (start_x, start_y), (end_x, end_y), box_color, 3)
        cv2.putText(img, label, (start_x, start_y - 10), cv2.FONT_ITALIC, 0.6, box_color, 2)

    t, _ = model.getPerfProfile()
    text = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
    cv2.putText(img, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    print(text)
    cv2.imshow("Detection",img)
    # fps.update()

    if cv2.waitKey(1) & 0xff == ord("q"):
        break

#cap.release()
camera.Close()
#fps.stop()
camera.StopGrabbing()
# print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
# print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()