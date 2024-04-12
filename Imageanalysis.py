import os
from PIL import Image, ImageDraw
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential

# Set the values of your computer vision endpoint and computer vision key
# as environment variables:
try:
    endpoint = os.environ["VISION_ENDPOINT"]
    key = os.environ["VISION_KEY"]
except KeyError:
    print("Missing environment variable 'VISION_ENDPOINT' or 'VISION_KEY'")
    print("Set them before running this sample.")
    exit()

# Create an Image Analysis client
client = ImageAnalysisClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(key)
)
#file_path = "c:/Users/Lottering/Downloads/WhatsApp Video 2024-04-09 at 22.56.38.mp4"
#image = Image.open(file_path)
#image_draw = ImageDraw.Draw(image)
import cv2

# Open the video file
file_path = "c:/Users/Lottering/Downloads/WhatsApp Video 2024-04-09 at 22.56.38.mp4"
video = cv2.VideoCapture(file_path)
#image_draw = ImageDraw.Draw(video)
# Loop over each frame in the video
while video.isOpened():
    # Read the next frame
    ret, frame = video.read()
   

    if ret:
        pil_image = Image.fromarray(frame)

         # Create a drawable image
        image_draw = ImageDraw.Draw(pil_image)
        
        # Analyze the image
        result = client.analyze(
            frame,
            visual_features=[VisualFeatures.OBJECTS])
        
        if result.objects is not None:
            for object in result.objects.list:
                left = object.bounding_box.x
                right = object.bounding_box.y
                width = object.bounding_box.width
                heigth = object.bounding_box.height

                shape = [(left, right),(left + width, right + heigth)]
                image_draw.rectangle(shape, outline="red",width=3)
                print('')
        # Convert the PIL Image object back to a numpy.ndarray object
        frame = np.array(pil_image)
        # Display the frame
        cv2.imshow()

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release the video file and close the windows
video.release()
cv2.destroyAllWindows()

# Open the image file in binary mode
with open(file_path, "rb") as image_file:
    image_data = image_file.read()



# Analyze the image
result = client.analyze(
    image_data,
    visual_features=[VisualFeatures.OBJECTS],
    gender_neutral_caption=True,  # Optional (default is False)
)

print("Image analysis results:")
# Print caption results to the console
print(" Objects:")
if result.objects is not None:
    for object in result.objects.list:
       left = object.bounding_box.x
       right = object.bounding_box.y
       width = object.bounding_box.width
       heigth = object.bounding_box.height

       shape = [(left, right),(left + width, right + heigth)]
       image_draw.rectangle(shape, outline="red",width=3)
       print('')
       
