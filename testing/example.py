import face_recognition

image = face_recognition.load_image_file("javier.jpg")
face_locations = face_recognition.face_locations(image)

print(face_locations)

# print(f'There are {̣len(face_locations)} people in this image')


