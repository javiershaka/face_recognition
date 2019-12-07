from PIL import Image
import face_recognition
import os

image = face_recognition.load_image_file('herligerman.jpg')
face_locations = face_recognition.face_locations(image)

 
# Create target directory & all intermediate directories if don't exists
dirName = "tempDir2/temp2/temp"
if not os.path.exists(dirName):
    
    os.makedirs(dirName)    
    print("Directory " , dirName ,  " Created ")
else:    
    print("Directory " , dirName ,  " already exists")
for face_location in face_locations:
    top, right, bottom, left = face_location
    face_image = image[top:bottom, left:right]
    pil_image = Image.fromarray(face_image)
    pil_image.show()
    pil_image.save(f"{top}.jpg")