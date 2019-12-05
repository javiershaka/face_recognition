import face_recognition
from PIL import Image, ImageDraw

image_of_javier = face_recognition.load_image_file("196.jpg")
javier_face_encoding = face_recognition.face_encodings(image_of_javier)[0]

unknown_image = face_recognition.load_image_file("387.jpg")
unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]

# Creaate array of enconding and names

known_face_encodings = [
    javier_face_encoding,
    unknown_face_encoding
]

known_face_names = [
    "javiershaka",
    "el otro wey"
]

#  Load test image to find faces in 
test_image = face_recognition.load_image_file("javier.jpg")

# find faces in test image


face_locations = face_recognition.face_locations(test_image)
face_encodings = face_recognition.face_encodings(test_image, face_locations)

# convert to Pil Format

pil_image = Image.fromarray(test_image)

# create imageDraw instance
draw = ImageDraw.Draw(pil_image)

#  lopp throug faces in test image
for(top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    name = "Unknown Person"

    # if match true
    if True in matches:
        first_match_index = matches.index(True)
        name = known_face_names[first_match_index]

# draw Box
draw.rectangle(((left, top), (right, bottom)), outline=(0,0,0))

text_width, text_height = draw.textsize(name)
draw.rectangle(((left, bottom - text_height -10), (right, bottom)), fill=(0,0,0), outline=(0,0,0))
draw.text((left + 6, bottom - text_height -5), name, fill=(255,255,255,255))

del draw

# display
pil_image.show()