import os

path = os.path.join(os.sep, os.getcwd(), 'val')
originDir = os.path.join(os.sep, path, 'images')
file = open(os.path.join(os.sep, path, 'val_annotations.txt'), 'r')
dir2image = {}
for line in file:
    image = line[0:line.find('\t')]
    dir = line[line.find('\t')+1:line.find('\t', line.find('\t') + 1)]
    if dir in dir2image.keys():
        dir2image[dir].append(image)
    else:
        dir2image[dir] = []
        dir2image[dir].append(image)
for key in dir2image.keys():
    curr = os.path.join(os.sep, path, key)
    try:
        os.mkdir(curr)
    except OSError as e:
        print "Oops! File exists:", curr
    for image in dir2image[key]:
        os.rename(os.path.join(os.sep, originDir, image), os.path.join(os.sep, curr, image))
os.rmdir(originDir)
os.rename(os.path.join(os.sep, os.getcwd(), 'test'), os.path.join(os.sep, os.getcwd(), 'predict'))
os.rename(os.path.join(os.sep, os.getcwd(), 'val'), os.path.join(os.sep, os.getcwd(), 'test'))