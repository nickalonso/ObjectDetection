import sys 
import os
import io
sys.path.append('./research')
sys.path.append('./research/slim')
sys.path.append('./research/object_detection')
sys.path.append('./research/object_detection/utils')
sys.path.append('/Users/nickalonso/Downloads/gisfederl-ctnr-bbox-sfei/SFEI/gisfederal-bbox-sfei/research')
sys.path.append('/Users/nickalonso/Downloads/gisfederl-ctnr-bbox-sfei/SFEI/gisfederal-bbox-sfei/research/slim')
sys.path.append('/Users/nickalonso/Downloads/gisfederl-ctnr-bbox-sfei/SFEI/gisfederal-bbox-sfei/research/object_detection')
sys.path.append('/Users/nickalonso/Downloads/gisfederl-ctnr-bbox-sfei/SFEI/gisfederal-bbox-sfei/research/object_detection/utils')
sys.path.append('/Users/nickalonso/Downloads/kinetica_scripts-and-raw-model/research/slim')
sys.path.append('/Users/nickalonso/Downloads/kinetica_scripts-and-raw-model/research/object_detection')
sys.path.append('/Users/nickalonso/Downloads/kinetica_scripts-and-raw-model/research/object_detection/utils')

import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import numpy as np
import pandas as pd
from PIL import Image
from visualization_utils import visualize_boxes_and_labels_on_image_array
from object_detection.utils import label_map_util
from PIL import Image

MODEL_PATH = './'
NUM_CLASSES = 3

class TrashClassifier():
    def __init__(self, model_path):
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()

            # Works up to here.
            with tf.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            self.d_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            self.d_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.d_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_d = self.detection_graph.get_tensor_by_name('num_detections:0')
        self.sess = tf.Session(graph=self.detection_graph)

    def get_classification(self, img):
        # Bounding Box Detection.
        with self.detection_graph.as_default():
            # Expand dimension since the model expects image to have shape [1, None, None, 3].
            img_expanded = np.expand_dims(img, axis=0)  
            (boxes, scores, classes, num) = self.sess.run(
                [self.d_boxes, self.d_scores, self.d_classes, self.num_d],
                feed_dict={self.image_tensor: img_expanded})
        return boxes, scores, classes, num

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

def gen_classified_image(trash_classifier, category_index, img, min_score_thresh):
    # Open image then convert to numpy array
    img_np = load_image_into_numpy_array(img)

    # Get bounding boxes
    (boxes, scores, classes, num) = trash_classifier.get_classification(img_np)

    # Draw bounding boxes on image
    img_out_np = visualize_boxes_and_labels_on_image_array(
        image = img_np,
        boxes = np.squeeze(boxes),
        classes = np.squeeze(classes),
        scores = np.squeeze(scores),
        category_index = category_index,
        use_normalized_coordinates = True,
        skip_labels = False,
        agnostic_mode = False,
        min_score_thresh=min_score_thresh
    )

    # Save image with bounding boxes
    img_out = Image.fromarray(img_out_np)

    byteio = io.BytesIO()
    img_out.save(byteio, format='jpeg')
    imbytes = byteio.getvalue()
    imageBytesEnc = bytearray(imbytes)
    score_vals = scores[0]
    detection_count = len(score_vals[score_vals > min_score_thresh])
    return imageBytesEnc, detection_count, num[0]

def detect(inMap):
    imageBytes = inMap["input"]
    imgInput = Image.open(io.BytesIO(imageBytes))
    # Minimum threshold
    min_score_thresh = 0.5

    # Initialize classifier
    model_path = os.path.join(MODEL_PATH,'frozen_inference_graph.pb')
    trash_classifier = TrashClassifier(model_path)
    detection_counts = pd.DataFrame(columns=['File','Detections > {}'.format(min_score_thresh),'Total Detections'])
    
    # Obtain label map
    label_map = label_map_util.load_labelmap(os.path.join(MODEL_PATH,'pascal_label_map.pbtxt'))
    categories = label_map_util.convert_label_map_to_categories(
        label_map,max_num_classes=NUM_CLASSES,use_display_name=True)

    category_index = label_map_util.create_category_index(categories)
    imageBytesEnc, detections, totalDetections = \
    gen_classified_image(
            trash_classifier, 
            category_index, 
            imgInput,
            min_score_thresh)

    # Outmap sent back to AAW
    return {
            'output': imageBytesEnc,
            'detection': int(detections),
            'totalDetection':int(totalDetections)
    }








