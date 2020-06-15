import cv2,time
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from tensorflow.keras.preprocessing import image
import tensorflow as tf

def crop_image(image, x, y, width, height):
    return image[y:y + height, x:x + width]

print("Place your hand in the green model")
print("press q to capture ")
video=cv2.VideoCapture(0)
start_point=(180,150)
end_point=(385,355)
color=(0,255,0)
thickness=2
while True:
	check,frame=video.read()

	# cv2.imshow("capturing",frame)
	# print(type(frame))
	img=cv2.rectangle(frame,start_point,end_point,color,thickness)
	cv2.imshow("Image",img)

	key=cv2.waitKey(1)

	if key==ord("q"):
		break

cv2.imwrite('imp.png',frame)
video.release()
cv2.destroyAllWindows
im=Image.open("imp.png")
width,height=im.size
left = 182
top = 152
right = 382
bottom = 353
im1 = im.crop((left, top, right, bottom)) 
im1=im1.save("imp.png")
#to load image for the model
img=image.load_img('imp.png',target_size=(28,28),grayscale=True)
# img.show()
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
images = np.vstack([x])
model=tf.keras.models.load_model('my_model.h5')

classes=model.predict(images, batch_size=10)

li=list(classes[0])

k=65+li.index(1)
print(chr(k))