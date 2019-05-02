from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse
from django.http import JsonResponse
import tensorflow as tf
import base64
from PIL import Image
import io
import cv2
import numpy as np
import json

def load_model():
    global graph
    global tensor_dict
    global input_tensor

    export_dir = "/home/elifildes/token_getter/instgram_crop/models"
    with tf.Session() as sess:
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], export_dir)
        graph = tf.get_default_graph()

    tensor_dict = {}
    for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes']:
            tensor_name = key + ':0'
            tensor_dict[key] = graph.get_tensor_by_name(tensor_name)

    input_tensor = graph.get_tensor_by_name('image_tensor:0')


def run_detection(image):
    with tf.Session(graph=graph) as sess:
        output_dict = sess.run(tensor_dict, feed_dict={
                               input_tensor: image})

    return output_dict

def prepare_image(image_string):
    """Convert the comimg binary string image to normal image, do some preprocessing
        and return it as 64-encoded binary string
    """
    imgdata = base64.b64decode(image_string)
    image = Image.open(io.BytesIO(imgdata))
        
    if image.mode != "RGB":
 	    image = image.convert("RGB")
    
    resized_image = image.resize((300, 300), Image.ANTIALIAS)
    image_array = np.array(image)[:, :, 0:3]

    return image_array

@csrf_exempt
def predict_crop(request):
    body_unicode = request.body.decode("utf-8")
    body = json.loads(body_unicode)
    image_string = body["b64"]
    image_array =  prepare_image(image_string)
    
    if (image_array == np.zeros(1)).all() == True:
        results = {}
        return JsonResponse(results)
    else:
        results = run_detection([image_array]) 
        boxes   = results["detection_boxes"].tolist()
        classes = results["detection_classes"].tolist()
        scores  = results["detection_scores"].tolist()
	
        results_dict= {
        "detection_boxes" : boxes,
        "detection_classes" : classes,
        "detection_scores" : scores
        }
      
        return JsonResponse(results_dict) 



# def prepare_image(image_string):
#     """Convert the comimg binary string image to normal image, do some preprocessing
#        and return it as 64-encoded binary string
#     """
#     #Resize the coming image
#     image_binary = base64.decode(str(image_string))
#     image = Image.open(io.BytesIO)

#     if image.mode != "RGB":
# 	    image = image.convert("RGB")

#     resized_image = image.resize((300, 300), Image.ANTIALIAS)

#     #Recovert the image to b64 string
#     binary_io = io.BytesIO()
#     resized_image.save(binary_io, "JPEG")

#     bytes_string_image = base64.b64encode(binary_io.getvalue()).decode("utf-8")

#     return image_string


# @csrf_exempt
# def predict(request, image_string):
#     preprocessed_image_string = prepare_image(image_string)
    





#image_data = tf.gfile.FastGFile("/home/drmaelk/presignia/test_image.jpg", 'rb').read()
#image = open("/home/drmaelk/presignia/test_image.jpg", "rb")
#img = Image.open("/home/drmaelk/presignia/datasets/font_postion_instgram/4000/1500-new/JPEGImages/0ca01492-63ef-49a0-a5bb-56869e2c49a9___63.jpg")    
#resized_img = img.resize((300, 300), Image.ANTIALIAS)

#binary_io = io.BytesIO()
#resized_img.save(binary_io, "JPEG")
    
#image = Image.open("/home/drmaelk/presignia/test_image.jpg")
#binary_io = io.BytesIO()
#image.save(binary_io, "JPEG")
#bytes_string_image = base64.b64encode(binary_io.getvalue())



#image = cv2.imread("/home/drmaelk/presignia/test_image.jpg")



# image = Image.open("/home/drmaelk/presignia/test_image.jpg")
# image_array = np.array(image)[:, :, 0:3]

#load_model()
#print(predict_crop("", bytes_string_image))
#print(run_detection([image_array])["num_detections"])
