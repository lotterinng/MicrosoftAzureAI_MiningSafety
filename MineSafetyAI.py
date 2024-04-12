import cv2
import os
from PIL import Image, ImageDraw
import numpy as np
import io
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
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
from azure.core.credentials import AzureKeyCredential

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

# Open the video file
video = cv2.VideoCapture('c:/Users/Lottering/Downloads/WhatsApp Video 2024-04-09 at 22.54.17.mp4')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))  # Replace (640, 480) with the dimensions of your frames

while True:

#while video.isOpened():
    # Read the next frame
    ret, frame = video.read()

    if ret:
        # Convert the frame to a byte stream
        is_success, im_buf_arr = cv2.imencode(".jpg", frame)
        h, w, ch = np.array(frame).shape
        byte_im = im_buf_arr.tobytes()


        # Create a byte stream from the byte data
        byte_stream = io.BytesIO(byte_im)

        #Convert the byte stream to a PIL Image object
        pil_image = Image.open(byte_stream)

        # Create a drawable image
        #Display the image with box
        imagedraw = ImageDraw.Draw(pil_image)
        lineWidth = int(w/100)
        color = 'magenta'
        #imagedraw = ImageDraw.Draw(pil_image)
        
        # Reset the position of the byte stream
        byte_stream.seek(0)
        # Analyze the frame
        result = predictor.detect_image(
            project_id='9b458444-ec77-4883-8a72-d6c4629dc76f',
            published_name='Iteration1',
            image_data =byte_stream
            
        )

        # Process the result here
        if result.predictions is not None:
            for prediction in result.predictions:
                if (prediction.probability*100) > 90:

                    left = prediction.bounding_box.left * w
                    top = prediction.bounding_box.top * h
                    height = prediction.bounding_box.height * h
                    width = prediction.bounding_box.width * w

                    #Draw th box
                    imagedraw.rectangle([left, top, left+width, top+height], outline=color, width=lineWidth)
                    #points = ((left,top),(left+width+top),(left+width,top+height))
                    #imagedraw.line(points, fill=color, width=lineWidth)

                    text = f'{prediction.tag_name}'
                    imagedraw.text((left + 5, top + height - 30),text, (225,0,0))
                    print('')
        # Convert the PIL Image object back to a numpy.ndarray object
        frame = np.array(pil_image)
        out.write(frame)   
        # Display the frame
        cv2.imshow('Frame', frame)
       

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release the video capture object
video.release()
out.release()

# Close all OpenCV windows
cv2.destroyAllWindows()