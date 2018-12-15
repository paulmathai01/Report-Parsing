import skimage.segmentation
import scipy
import numpy as np
from collections import namedtuple
import cv2
from random import randint
from skimage import draw
import tkinter
from PIL import ImageTk, Image
from tkinter.filedialog import askopenfilename
import glob
import pandas as pd
from pandas import ExcelWriter
import keras.preprocessing.image
from keras.models import load_model
import os

global name


def filename():
    filename.filename = askopenfilename()


# name = filename


main = tkinter.Tk()
main.title('Report Parser')
canvas = tkinter.Canvas(main, width=300, height=250)
img = ImageTk.PhotoImage(Image.open("and.jpeg"))
canvas.create_image(150, 100, image=img)
canvas.grid(row=0, column=0)
button = tkinter.Button(main, text='Browse', width=5, command=filename)
button.grid(row=1, column=0)
button2 = tkinter.Button(main, text='Start', width=5, command=main.destroy)
button2.grid(row=2, column=0)
# filename = askopenfilename()
main.mainloop()
print("python felz.py --image " + filename.filename)

abc = filename.filename


class Rectangle(namedtuple('Rectangle', 'x0 y0 x1 y1')):
    def pixels(_, shape):
        return draw.polygon([_.x0, _.x0, _.x1, _.x1], [_.y0, _.y1, _.y1, _.y0], shape)

def aabb(region):
    (x0, y0) = np.min(region, axis=0)
    (x1, y1) = np.max(region, axis=0)
    return Rectangle(x0, y0, x1, y1)


img2 = scipy.misc.imread(filename.filename)


###
gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV +cv2.THRESH_OTSU)
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 5, 5)

thresh = cv2.bilateralFilter(thresh, 1, 75, 75)
kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
thresh1 = cv2.filter2D(thresh, -1, kernel)
thresh1 = cv2.Canny(thresh1, 50, 150, apertureSize=3)
# cv2.imshow("a1", thresh1)
# cv2.waitKey(0)

lines = cv2.HoughLines(thresh1, 1, np.pi / 180, 200, 1)
for rho, theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))

    cv2.line(thresh1, (x1, y1), (x2, y2), (0, 0, 0), 5)

# cv2.imshow("a1", thresh1)
# cv2.waitKey(0)

###
# segment_mask1 = skimage.segmentation.felzenszwalb(img2, scale=100)
segment_mask2 = skimage.segmentation.felzenszwalb(thresh1, scale=1000)
segments = segment_mask2
"""
fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.imshow(segment_mask1); ax1.set_xlabel("k=100")
ax2.imshow(segment_mask2); ax2.set_xlabel("k=1000")
fig.suptitle("Felsenszwalb's efficient graph based image segmentation")
plt.tight_layout()
plt.show()
"""
numRegions = segments.max()
rectangles = []
y = []

for regionTag in range(numRegions):
    selectedRegion = segments == regionTag
    regionPixelIndices = np.transpose(np.nonzero(selectedRegion))
    (x0, y0) = np.min(regionPixelIndices, axis=0)
    (x1, y1) = np.max(regionPixelIndices, axis=0)
    rectangles.append(((x0, y0, x1, y1)))
    x = aabb(regionPixelIndices)
    y.append(x)

# print(rectangles)
a = rectangles[5]
print(len(rectangles))
"""
im1 = img2[a[0]:a[2],a[1]:a[3]]
cv2.imshow("a", im1)
cv2.waitKey(0)
"""
os.mkdir("parse_imgs")
g = 0
for i in range(len(rectangles)):
    a = rectangles[i]
    try:
        im1 = img2[a[0] - 5:a[2] + 5, a[1] - 5:a[3] + 5]
        height = a[3] - a[1]
        length = a[2] - a[0]
        if (height > length + 10 or length > height + 10 or length < 10 or height < 10):
            continue
        g = g + 1
        cv2.imwrite("parse_imgs/%d.png" % i, im1)
    # cv2.imshow("a", im1)
    # cv2.waitKey(0)

    except cv2.error:
        continue
print(g)


def debug():
    marked = np.zeros(img2.shape, dtype=np.uint8)

    for rectangle in y:
        rr, cc = rectangle.pixels(marked.shape)
        randcolor = randint(0, 255), randint(0, 255), randint(0, 255)
        marked[rr, cc] = randcolor

    print(img2.shape, segments.shape, marked.shape)


# io.imshow_collection([img2, segments, marked])
# io.show()

# debug()

# call("python test_network.py --model abc.model --image parse_imgs",shell = True)

x = []
li = []
odir = "parse_imgs"
for filename in glob.glob(odir + "/*.png"):
    x.append(filename[11:])

new_set = [int(s.replace('.png', '')) for s in x]

new_set.sort()
new_n_set = ["parse_imgs/" + str(s) + ".png" for s in new_set]
# print(new_n_set)

for filename in new_n_set:
    model = "abc.model"
    # load the image
    image = cv2.imread(filename)
    orig = image.copy()
    # pre-process the image for classification
    image = cv2.resize(image, (30, 30))
    image = image.astype("float") / 255.0
    image = keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    # load the trained convolutional neural network
    print("[INFO] loading network...")
    model = load_model(model)

    a = model.predict(image)[0]
    # print(a)
    pred = str(np.argmax(a))
    li.append(pred)
    print(str(np.argmax(a)) + "\t" + filename)

df = pd.DataFrame({'pred': li})

writer = ExcelWriter(str(abc) + ".xlsx")
df.to_excel(writer, 'Sheet1', index=False)
writer.save()

# python test_network.py --model santa_not_santa.model --image images/parse/santa_01.png
