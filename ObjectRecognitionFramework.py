import os
import sys
import random
import math
import numpy as np
import cv2
import json
# Root directory of the project
ROOT_DIR = os.path.abspath("./")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "Mask_RCNN-master/samples/coco/"))  # To find local version
from samples.coco import coco



def random_colors(N):
    np.random.seed(1)
    colors = [tuple(255 * np.random.rand(3)) for _ in range(N)]
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    """apply mask to image"""
    for n, c in enumerate(color):
        image[:, :, n] = np.where(
            mask == 1,
            image[:, :, n] * (1 - alpha) + alpha * c,
            image[:, :, n]
        )
    return image
    
    
def display_instances(image, boxes, masks, ids, names, scores):
    """
        take the image and results and apply the mask, box, and Label
    """
    n_instances = boxes.shape[0]
    colors = random_colors(n_instances)

    if not n_instances:
        print('NO INSTANCES TO DISPLAY')
    else:
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]

    for i, color in enumerate(colors):
        if not np.any(boxes[i]):
            continue

        y1, x1, y2, x2 = boxes[i]
        label = names[ids[i]]
        score = scores[i] if scores is not None else None
        caption = '{} {:.2f}'.format(label, score) if score else label
        mask = masks[:, :, i]

        image = apply_mask(image, mask, color)
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        image = cv2.putText(
            image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2
        )

    return image
    
class ObjectRecognitionFramework:

    def __init__(self, config_path="config.json"):
        '''# Root directory of the project
        ROOT_DIR = os.path.abspath("../")

        # Import Mask RCNN
        sys.path.append(ROOT_DIR)  # To find local version of the library
        from mrcnn import utils
        import mrcnn.model as modellib
        from mrcnn import visualize
        # Import COCO config
        sys.path.append(os.path.join(ROOT_DIR, "Mask_RCNN-master/samples/coco/"))  # To find local version
        import coco
        '''
        
        config_file = json.loads(open(config_path).read())
        
        # Directory to save logs and trained model
        MODEL_DIR = config_file.get("logs_path", "logs")

        # Local path to trained weights file
        COCO_MODEL_PATH = config_file.get("weights", "mask_rcnn_coco.h5")
        
        # Directory of images to run detection on
        #IMAGE_DIR = os.path.join(ROOT_DIR, "Mask_RCNN-master/images")

        class InferenceConfig(coco.CocoConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        
        self.config = InferenceConfig()
        self.model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=self.config)
        
        # Load weights trained on MS-COCO
        self.model.load_weights(COCO_MODEL_PATH, by_name=True)
        self.class_colors = random_colors(81)
        # COCO Class names
        # Index of the class in the list is its ID. For example, to get ID of
        # the teddy bear class, use: class_names.index('teddy bear')
        self.class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

    def input_image(self,image_dir):

        # Load a random image from the images folder
        file_names = next(os.walk(image_dir))[2]
        image = cv2.imread(os.path.join(image_dir, random.choice(file_names)))

        # Run detection
        results = self.model.detect([image], verbose=1)

        # Visualize results
        r = results[0]
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            self.class_names, r['scores'])


    def input_single_image(self,image_dir, image_name):
        image_dir+="/"+image_name
        # Load a random image from the images folder
        #file_names = next(os.walk(IMAGE_DIR))[2]
        image = cv2.imread(os.path.join(image_dir))

        # Run detection
        results = self.model.detect([image], verbose=1)

        # Visualize results
        r = results[0]
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            self.class_names, r['scores'])

    def input_video(self, video_path):
        cam = cv2.VideoCapture(video_path) 
        if (cam.isOpened()== False): 
            print("Error opening video stream or file") 
        try: 
            # creating a folder named data 
            if not os.path.exists('video'): 
                os.makedirs('video') 
  
            # if not created then raise error 
        except OSError: 
            print ('Error: Creating directory of data') 
  
        # frame 
        currentframe = 0
  
        while(True): 
      
            # reading from frame 
            ret,frame = cam.read() 
            if ret: 
                # if video is still left continue creating images 
                root_vid=ROOT_DIR+"/video"
                name = root_vid+"/frame" + str(currentframe) + ".jpg"
                print ('Creating...' + name) 
                
                frame = display_instances(frame, r['rois'], r['masks'], r['class_ids'], 
                            self.class_names, r['scores'], self.class_colors)
                            
                # writing the extracted images  self.class_colors
                cv2.imwrite(name, frame) 
  
                # increasing counter so that it will 
                # show how many frames are created 
                currentframe += 1
            else: 
                break
  
        # Release all space and windows once done 
        cam.release() 
        cv2.destroyAllWindows()
        
        
    def input_move(self,VIDEO_DIR):

        capture = cv2.VideoCapture(VIDEO_DIR) 
        # Check if camera opened successfully
        if (capture.isOpened()== False): 
            print("Error opening video stream or file")   
        VIDEO_SAVE_DIR=ROOT_DIR+"/"
        
        width = int(capture.get(3))
        height = int(capture.get(4))
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'MPEG')
        out = cv2.VideoWriter('output.avi',fourcc, 20.0, (width,height))

        while(capture.isOpened()):
            ret, frame = capture.read()
            if ret==True:
                #frame = cv2.flip(frame,0)
                results = self.model.detect([frame], verbose=1)
                # Visualize results
                r = results[0]
                #frame = (frame, r['rois'], r['masks'], r['class_ids'], self.class_names, r['scores'])
                display_instances(frame, r['rois'], r['masks'], r['class_ids'], 
                            self.class_names, r['scores'], self.class_colors)
                # write the frame
                #out.write(frame)
    
                #cv2.imshow('frame',frame)
                #if cv2.waitKey(1) & 0xFF == ord('q'):
                #    break
            else:
                break
    
         # Release everything if job is finished
        capture.release()
        out.release()
        cv2.destroyAllWindows()