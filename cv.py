
import cv2
import numpy as np
from imutils.video import FPS
import time

frame_count_out = 0
frame_count = 0

# 'path to input image/video'
VIDEO='/Users/shruti/Desktop/videos/v.mp4'

# 'path to yolo config file' 
CONFIG='/Users/shruti/Desktop/project/yolov3-obj.cfg'

# 'path to text file containing class names'
CLASSES='/Users/shruti/Desktop/project/yolov3_helmet.txt'

# 'path to yolo pre-trained weights' 
WEIGHTS='/Users/shruti/Desktop/project/yolov3-obj_2400.weights'


import os  
print(os.path.exists(CLASSES))
print(os.path.exists(CONFIG))
print(os.path.exists(WEIGHTS))
print(os.path.exists(VIDEO))


classes = None
with open(CLASSES, 'r') as f:
     classes = [line.strip() for line in f.readlines()]
        
scale = 0.00392
conf_threshold = 0.5
nms_threshold = 0.4

# generate different colors for different classes 
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))


# function to get the output layer names i.e layers with unconnected outputs
# in the architecture
def get_output_layers(net): 
    layer_names = net.getLayerNames() #get all layers names of the network
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    #print(output_layers)
    return output_layers

# function to draw bounding box on the detected object with class name
def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

	global frame_count_out
	frame_count = 0

	##Display the label at the top of the bounding box
	label = str(classes[class_id])
	color = COLORS[class_id]
	cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 3)
	labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
	y = max(y, labelSize[1])

	if classes:
		assert(class_id < len(classes))
		label = '%s:%s' % (classes[class_id], label)
	label_name,label_conf = label.split(':')    #spliting into class & confidence. will compare it with person.
	if label_name == 'Helmet':
		cv2.rectangle(img, (x, y - round(1.5*labelSize[1])), (x + round(1.5*labelSize[0]), y + baseLine), (255, 255, 255), cv2.FILLED)
		cv2.putText(img, label_name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
		frame_count+=1

	if(frame_count> 0):
		return frame_count


    #cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def processImage(image,index):

    # read pre-trained model and config file
    net = cv2.dnn.readNetFromDarknet(CONFIG, WEIGHTS)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # create 4D input blob 
    blob = cv2.dnn.blobFromImage(image, scale, (416,416), [0,0,0], 1, crop=False)
    # set input blob for the network
    net.setInput(blob)

    # run inference through the network
    # and gather predictions from output layers
    outs = net.forward(get_output_layers(net))

    global frame_count_out
    frame_count_out=0
    count_person = 0


    Width = image.shape[1]
    Height = image.shape[0]

    # initialization
    class_ids = []
    confidences = []
    boxes = []
    # for each detetion from each output layer 
    # get the confidence, class id, bounding box params
    # and ignore weak detections (confidence < 0.5)
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id] #The highest score for a box is also called its confidence.
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
            
    # apply non-max suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    # go through the detections remaining after nms and draw bounding box
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
    
        frame_count_out = draw_bounding_box(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
        # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
        t, _ = net.getPerfProfile()
        label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
        cv2.putText(image, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
        my_class='Helmet'                   
        unknown_class = classes[class_id]
        if my_class == unknown_class:
        	count_person += 1
        print(frame_count_out)
        if(count_person >=0):
        	vid_writer.write(image.astype(np.uint8))
        	cv2.waitKey(10)
        	#out_image_name = "frame"+str(index)
        	#cv2.imwrite("out/"+out_image_name+".jpg", image)
        	#cv2.imshow(out_image_name, image)
        	#
	        
time.sleep(2.0)


# open the video file
cap = cv2.VideoCapture(VIDEO)
fps = FPS().start()
outputFile = VIDEO[:-4]+'_yolo_out.avi'
vid_writer = cv2.VideoWriter(outputFile, cv2.VideoWriter_fourcc('M','J','P','G'), 30, (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))


index = 0
#cap.isOpened()
while( True ):
    ret, frame = cap.read()
    fps.update()
    if not ret:
    	print("Done processing !!!")
    	print("Output file is stored as ", outputFile)
    	
    	break

    processImage(frame,index)
    index = index +1
    key = cv2.waitKey(1) & 0xFF
    if(key == ord("q")):
    	break


fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))    
# release resources
cv2.destroyAllWindows()



