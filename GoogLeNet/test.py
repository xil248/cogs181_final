# coding: utf-8
import glob
import os, math
import numpy as np

from PIL import Image
from scipy import misc
from io import BytesIO
from datetime import datetime
from flask import Flask, request, render_template
from random import random, choice

from lib import data_util
from lib.config import params_setup
from lib.googlenet import GoogLeNet




# model
args = params_setup()
gnet = GoogLeNet(args=args)
directory_names = list(set(glob.glob(os.path.join("images","tiny-imagenet-200","jpg", "*"))\
 ).difference(set(glob.glob(os.path.join("images","tiny-imagenet-200","jgp","*.*",)))))


total_count = 0.
correct_count = 0.
# len(directory_names)
for i in range(len(directory_names)):
    imgs_in_folder = glob.glob(os.path.join(directory_names[i],"images","*.JPEG"))
    for j in range(400,len(imgs_in_folder)):
        cur_img = imgs_in_folder[j]
        # img = imread(cur_img)
        # img = load_image(s)
        # img = Image.open(cur_img)
        # # img = resize_image(img, 227, 227)

        # # img = Image.open(cur_img)
        # img = img.resize((227,227), Image.ANTIALIAS)
        # img.load()
        # img = np.asarray(img, dtype="float32")
        # img /= 255.

        # img = Image.open(cur_img)
        img = misc.imread(cur_img)
        img = misc.imresize(img,(227,227))

        # print (img.shape)

        # img = img.resize((227,227), Image.ANTIALIAS)
        # img.load()
        # img = np.asarray(img, dtype="float32")



        # img = np.reshape(img,(227,227,1))

        probs = gnet.predict([img])[0]
        maxInd = 0
        maxVal = 0
        for k in range(len(probs)):
            if (maxVal < probs[k]):
                maxVal = probs[k]
                maxInd = k
            
        total_count += 1
        if (maxInd == i):
            correct_count += 1



        # print (probs)
        # cnt = int(sum([math.exp(i+4) * probs[i] for i in range(len(probs))]))

        # probs = [(i, round(100*p, 1)) for i, p in enumerate(probs)]
        # print (probs)

accuracy = correct_count / total_count 
print (total_count)
print (accuracy)

# def resize_image(in_image, new_width, new_height, out_image=None,
#                  resize_mode=Image.ANTIALIAS):
#     """ Resize an image.
#     Arguments:
#         in_image: `PIL.Image`. The image to resize.
#         new_width: `int`. The image new width.
#         new_height: `int`. The image new height.
#         out_image: `str`. If specified, save the image to the given path.
#         resize_mode: `PIL.Image.mode`. The resizing mode.
#     Returns:
#         `PIL.Image`. The resize image.
#     """
#     img = in_image.resize((new_width, new_height), resize_mode)
#     if out_image:
#         img.save(out_image)
#     return img



# example_file = glob.glob(os.path.join(directory_names[5],"*.jpg"))[9]
# print example_file
# im = imread(example_file, as_grey=True)
# plt.imshow(im, cmap=cm.gray)
# plt.show()



# def guess():
#     url = request.args.get('url', '')
#     if url:
#         X = url2sample(url)
#         probs = gnet.predict([X])[0]
#         cnt = int(sum([math.exp(i+4) * probs[i] for i in range(len(probs))]))
#         probs = [(i, round(100*p, 1)) for i, p in enumerate(probs)]
#     else:
#         cnt, probs = None, None
#     return render_template('guess.html', probs=probs, url=url, cnt=cnt)


# def url2sample(url, resize=(227, 227)):
#     try:
#         response = requests.get(url)
#         img = Image.open(BytesIO(response.content))
#         img = img.resize(resize, Image.ANTIALIAS)
#         img.load()
#         img = np.asarray(img, dtype="float32")
#         if (len(img.shape) != 3) or (img.shape[2] != 3): 
#             return None
#         img /= 255.
#         return img
#     except Exception as e:
#         print(e)
#         return None

# #---------------------------
# #   Start Server
# #---------------------------
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=8883, debug=False)

