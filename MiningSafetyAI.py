from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from matplotlib import pyplot as plt
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateBatch, ImageFileCreateEntry, Region
from msrest.authentication import ApiKeyCredentials
import os, time, uuid
import cv2
import os
from PIL import Image, ImageDraw
import numpy as np
import io

# Replace with valid values
ENDPOINT = os.environ["VISION_TRAINING_ENDPOINT"]
training_key = os.environ["VISION_TRAINING_KEY"]
prediction_key = os.environ["VISION_PREDICTION_KEY"]
prediction_resource_id = os.environ["VISION_PREDICTION_RESOURCE_ID"]


#Authenticate a client for the training API
credentials = ApiKeyCredentials(in_headers={"Training-key": training_key})
trainer = CustomVisionTrainingClient(ENDPOINT, credentials)
prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
predictor = CustomVisionPredictionClient(ENDPOINT, prediction_credentials)
        

#load image and get hieght, width and channels
image_file = 'image.jpeg'
print('Detecting Objects in', image_file)
image = Image.open(image_file)
h, w, ch = np.array(image).shape

#Detect Objects in the test image
with open (image_file,mode="rb") as image_data:
    results = predictor.detect_image(
        project_id='f21684cd-4664-4317-be15-58ddf506038d',
        published_name='Iteration1',
        image_data=image_file)
        
    #create a figure for the results
    fig = plt.figure(figsize=(8,8))
    plt.axis('off')

    #Displat the image with box
    draw = ImageDraw.Draw(image)
    lineWidth = int(w/100)
    color = 'magenta'
    for predition in results.predictor:
            #Only show objects with a >50% probability
            if (predition.proability*100) > 50:
                #Box coordinates and dimensions are proportional - convert to
                left = predition.bounding_box.left * w
                top = predition.bounding_box.top * h
                height = predition.bounding_box.height * h
                width = predition.bounding_box * w

                #Draw th box
                points = ((left,top),(left+width+top),(left+width,top+height))
                draw.line(points, fill=color, width=lineWidth)
                #Add the tagname and probability
                plt.annotate(predition.tag_name + ": {0:.2f}%".format(predition))
    plt.imshow(image)
    outputfile = "output.jpg"
    fig.savefig(outputfile)
    print ('Results saved in', outputfile)




            