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
    global crop_graph
    global tensor_dict
    global input_tensor  #crop

    export_dir = "/home/elifildes/token_getter/instgram_two_models/model"   #crop model

    crop_graph = tf.Graph()
    crop_sess =  tf.Session(graph=crop_graph)

    with crop_sess.as_default():
        with crop_graph.as_default():
            tf.saved_model.loader.load(crop_sess, [tf.saved_model.tag_constants.SERVING], export_dir)
    
    tensor_dict = {}
    for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes']:
            tensor_name = key + ':0'
            tensor_dict[key] = crop_graph.get_tensor_by_name(tensor_name)

    input_tensor = crop_graph.get_tensor_by_name('image_tensor:0')

def load_model2():
    global header_graph
    global tensor_dict2
    global input_tensor2

    export_dir2 = "/home/elifildes/token_getter/instgram_two_models/model2"      #header model

    header_graph = tf.Graph()
    header_sess = tf.Session(graph=header_graph)

    with header_sess.as_default():
        with header_graph.as_default():
            tf.saved_model.loader.load(header_sess, [tf.saved_model.tag_constants.SERVING], export_dir2)

    tensor_dict2 = {}
    for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes']:
            tensor_name = key + ':0'
            tensor_dict2[key] = header_graph.get_tensor_by_name(tensor_name)
    input_tensor2 = header_graph.get_tensor_by_name('image_tensor:0')


def get_crop(image):
    crop_sess =  tf.Session(graph=crop_graph)
    with crop_sess.as_default():
        output_dict = crop_sess.run(tensor_dict, feed_dict={
                               input_tensor: image})

    return output_dict

def get_header(image):
    header_sess = tf.Session(graph=header_graph)
    with header_sess.as_default():
        output_dict = header_sess.run(tensor_dict2, feed_dict={
                               input_tensor2: image})

    return output_dict

def prepare_image(image_string):
    """Convert the comimg binary string image to normal image, do some preprocessing
        and return it as binary array and opencv image
    """
    imgdata = base64.b64decode(image_string)
    image = Image.open(io.BytesIO(imgdata))
    
    if image.mode != "RGB":
 	    image = image.convert("RGB")
    
    resized_image = image.resize((300, 300), Image.ANTIALIAS)
    
    image_array = np.array(image)[:, :, 0:3]
    
    image_opencv = np.array(image) 
    #Convert RGB to BGR 
    image_opencv = image_opencv[:, :, ::-1].copy() 
    

    return image_array, image_opencv

def get_maximum_results(results, class_id):
    boxes   = results["detection_boxes"].tolist()
    classes = results["detection_classes"].tolist()
    scores  = results["detection_scores"].tolist()

    max_score = 0
    maxes_indeces = [0]
    for i in range(0, len(scores)):
      if classes[i] == class_id:                             
        if scores[i] > max_score:
          max_score = scores[i]
          maxes_indeces[0] = i

    
    max_boundries = boxes[0][maxes_indeces[0]]
    return max_boundries


def crop_image(image_opencv, max_crop_boundries):
    height, width, _ = image_opencv.shape
    cropped_ymin  = height * max_crop_boundries[0]
    cropped_xmin  = width * max_crop_boundries[1]
    cropped_ymax  = height * max_crop_boundries[2]
    cropped_xmax  = width * max_crop_boundries[3]
    
    #Crop the image with comming cordinates
    cropped_image = image_opencv[
            height - round(cropped_ymax) : round(float(cropped_ymax)) +  round(float(cropped_ymax)) -  round(float(cropped_ymin)),
            round(float(cropped_xmin)) : round(float(cropped_xmin)) +  round(float(cropped_xmax)) -  round(float(cropped_xmin))
    ]

    #covert the cropped image to rgb 
#    cropped_image = cv2.cvtColor(cropped_image)
    cv2.imwrite("image.jpg" ,cropped_image)
    return cropped_image


@csrf_exempt
def predict_crop(request):
    body_unicode = request.body.decode("utf-8")
    body = json.loads(body_unicode)
    image_string = body["b64"]
    image_array, opencv_image =  prepare_image(image_string)
    
    if (image_array == np.zeros(1)).all() == True:
        results = {}
        return JsonResponse(results)
    
    else:
        #First send the image to the crop model and get the crop information
        crop_results = get_crop([image_array]) 
        print(crop_results)
        #get the maximum crop
        maximum_crop = get_maximum_results(crop_results, 1)

        #crop the image with max crop, then return it
        cropped_image = crop_image(opencv_image, maximum_crop)

        #feed the cropped image into the header model
        header_results = get_header([cropped_image])
        print(header_results)
        #get the maximum haeder results
        maximum_header = get_maximum_results(header_results, 1)

        results_dict= {
        "crop" : maximum_crop,
        "header" : maximum_header,
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
