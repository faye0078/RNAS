from PIL import Image
import glob
import numpy as np
import os

def image_clip(img_path, size, save_dir):

    # 转换为数组进行分割操作，计算能完整分割的行数(row)、列数(col)
    img_name = img_path.split('.')[-2].split('/')[-1]
    img_dir = os.path.join(save_dir, img_name)
    folder = os.path.exists(img_dir)
    if not folder:
        os.makedirs(img_dir)

    imarray = np.array(Image.open(img_path))
    imshape = imarray.shape
    H = imshape[0]
    W = imshape[1]
    num_col = int(W / size[1]) - 1
    num_row = int(H / size[0]) - 1
    step_col = (W - num_col * size[1]) - size[1]
    step_row = (H - num_row * size[0]) - size[0]

    for row in range(num_row):
        for col in range(num_col):
            clipArray = imarray[row * size[0]:(row + 1) * size[0], col * size[1]:(col + 1) * size[1]]
            clipImg = Image.fromarray(clipArray)

            img_filepath = img_dir + '/' + img_name + "_" + str(
                row + 1) + "_" + str(col + 1) + "_img.tif"
            clipImg.save(img_filepath)


    # 两个for循环分割能完整分割的图像，并保存图像、坐标转换文件
    for row in range(num_row):
        clipArray = imarray[row * size[0]:(row + 1) * size[0], num_col * size[1]:(num_col + 1) * size[1]]
        clipImg = Image.fromarray(clipArray)
        img_filepath = img_dir + '/' + img_name + "_" + str(
            row + 1) + "_" + str(num_col + 1) + "_img.tif"
        clipImg.save(img_filepath)

        clipArray = imarray[row * size[0]:(row + 1) * size[0], num_col * size[1] + step_col:(num_col + 1) * size[1] + step_col]
        if (num_col + 1) * size[1] + step_col != imshape[1]:
            print('1drong!!')
        clipImg = Image.fromarray(clipArray)
        img_filepath = img_dir + '/' + img_name + "_" + str(
            row + 1) + "_" + str(num_col + 2) + "_img.tif"
        clipImg.save(img_filepath)

    for col in range(num_col):
        clipArray = imarray[num_row * size[0]:(num_row + 1) * size[0], col * size[1]:(col + 1) * size[1]]
        clipImg = Image.fromarray(clipArray)
        img_filepath = img_dir + '/' + img_name + "_" + str(
            num_row + 1) + "_" + str(col + 1) + "_img.tif"
        clipImg.save(img_filepath)

        clipArray = imarray[num_row * size[0] + step_row:(num_row + 1) * size[0] + step_row, col * size[1]:(col + 1) * size[1]]
        if (num_row + 1) * size[0] + step_row != imshape[0]:
            print('2drong!!')
        clipImg = Image.fromarray(clipArray)
        img_filepath = img_dir + '/' + img_name + "_" + str(
            num_row + 2) + "_" + str(col + 1) + "_img.tif"
        clipImg.save(img_filepath)

    clipArray = imarray[num_row * size[0]:(num_row + 1) * size[0], num_col * size[1]:(num_col + 1) * size[1]]
    clipImg = Image.fromarray(clipArray)
    img_filepath = img_dir + '/' + img_name + "_" + str(
        num_row + 1) + "_" + str(num_col + 1) + "_img.tif"
    clipImg.save(img_filepath)

    clipArray = imarray[num_row * size[0]:(num_row + 1) * size[0], num_col * size[1] + step_col:(num_col + 1) * size[1] + step_col]
    if (num_col + 1) * size[1] + step_col != imshape[1]:
        print('3drong!!')
    clipImg = Image.fromarray(clipArray)
    img_filepath = img_dir + '/' + img_name + "_" + str(
        num_row + 1) + "_" + str(num_col + 2) + "_img.tif"
    clipImg.save(img_filepath)

    clipArray = imarray[num_row * size[0] + step_row:(num_row + 1) * size[0] + step_row, num_col * size[1]:(num_col + 1) * size[1]]
    if (num_row + 1) * size[0] + step_row != imshape[0]:
        print('4drong!!')
    clipImg = Image.fromarray(clipArray)
    img_filepath = img_dir + '/' + img_name + "_" + str(
        num_row + 2) + "_" + str(num_col + 1) + "_img.tif"
    clipImg.save(img_filepath)

    clipArray = imarray[num_row * size[0] + step_row:(num_row + 1) * size[0] + step_row, num_col * size[1] + step_col:(num_col + 1) * size[1] + step_col]
    if (num_row + 1) * size[0] + step_row != imshape[0]:
        print('5drong!!')
    if (num_col + 1) * size[1] + step_col != imshape[1]:
        print('6drong!!')
    clipImg = Image.fromarray(clipArray)
    img_filepath = img_dir + '/' + img_name + "_" + str(
        num_row + 2) + "_" + str(num_col + 2) + "_img.tif"
    clipImg.save(img_filepath)

def get_gid_labels():
    return np.array([
        [255,0,0],    #buildup
        [0,255,0],   #farmland
        [0,255,255],  #forest
        [255,255,0],  #meadow
        [0,0,255] ])  #water
def label_clip(img_path, size, save_path):

    # 转换为数组进行分割操作，计算能完整分割的行数(row)、列数(col)
    img_name = img_path.split('.')[-2].split('/')[-1].replace('_24label', '')
    img_dir = os.path.join(save_path, img_name)
    folder = os.path.exists(img_dir)
    if not folder:
        os.makedirs(img_dir)

    mask = np.array(Image.open(img_path))
    imarray = np.array(mask)
    imarray = np.uint8(imarray)
    imshape = imarray.shape
    H = imshape[0]
    W = imshape[1]
    num_col = int(W / size[1]) - 1
    num_row = int(H / size[0]) - 1
    step_col = (W - num_col * size[1]) - size[1]
    step_row = (H - num_row * size[0]) - size[0]

    for row in range(num_row):
        for col in range(num_col):
            clipArray = imarray[row * size[0]:(row + 1) * size[0], col * size[1]:(col + 1) * size[1]]
            clipImg = Image.fromarray(clipArray)

            img_filepath = img_dir + '/' + img_name + "_" + str(
                row + 1) + "_" + str(col + 1) + "_label.png"
            clipImg.save(img_filepath)


    # 两个for循环分割能完整分割的图像，并保存图像、坐标转换文件
    for row in range(num_row):
        clipArray = imarray[row * size[0]:(row + 1) * size[0], num_col * size[1]:(num_col + 1) * size[1]]
        clipImg = Image.fromarray(clipArray)
        img_filepath = img_dir + '/' + img_name + "_" + str(
            row + 1) + "_" + str(num_col + 1) + "_label.png"
        clipImg.save(img_filepath)

        clipArray = imarray[row * size[0]:(row + 1) * size[0], num_col * size[1] + step_col:(num_col + 1) * size[1] + step_col]
        if (num_col + 1) * size[1] + step_col != imshape[1]:
            print('1drong!!')
        clipImg = Image.fromarray(clipArray)
        img_filepath = img_dir + '/' + img_name + "_" + str(
            row + 1) + "_" + str(num_col + 2) + "_label.png"
        clipImg.save(img_filepath)

    for col in range(num_col):
        clipArray = imarray[num_row * size[0]:(num_row + 1) * size[0], col * size[1]:(col + 1) * size[1]]
        clipImg = Image.fromarray(clipArray)
        img_filepath = img_dir + '/' + img_name + "_" + str(
            num_row + 1) + "_" + str(col + 1) + "_label.png"
        clipImg.save(img_filepath)

        clipArray = imarray[num_row * size[0] + step_row:(num_row + 1) * size[0] + step_row, col * size[1]:(col + 1) * size[1]]
        if (num_row + 1) * size[0] + step_row != imshape[0]:
            print('2drong!!')
        clipImg = Image.fromarray(clipArray)
        img_filepath = img_dir + '/' + img_name + "_" + str(
            num_row + 2) + "_" + str(col + 1) + "_label.png"
        clipImg.save(img_filepath)

    clipArray = imarray[num_row * size[0]:(num_row + 1) * size[0], num_col * size[1]:(num_col + 1) * size[1]]
    clipImg = Image.fromarray(clipArray)
    img_filepath = img_dir + '/' + img_name + "_" + str(
        num_row + 1) + "_" + str(num_col + 1) + "_label.png"
    clipImg.save(img_filepath)

    clipArray = imarray[num_row * size[0]:(num_row + 1) * size[0], num_col * size[1] + step_col:(num_col + 1) * size[1] + step_col]
    if (num_col + 1) * size[1] + step_col != imshape[1]:
        print('3drong!!')
    clipImg = Image.fromarray(clipArray)
    img_filepath = img_dir + '/' + img_name + "_" + str(
        num_row + 1) + "_" + str(num_col + 2) + "_label.png"
    clipImg.save(img_filepath)

    clipArray = imarray[num_row * size[0] + step_row:(num_row + 1) * size[0] + step_row, num_col * size[1]:(num_col + 1) * size[1]]
    if (num_row + 1) * size[0] + step_row != imshape[0]:
        print('4drong!!')
    clipImg = Image.fromarray(clipArray)
    img_filepath = img_dir + '/' + img_name + "_" + str(
        num_row + 2) + "_" + str(num_col + 1) + "_label.png"
    clipImg.save(img_filepath)

    clipArray = imarray[num_row * size[0] + step_row:(num_row + 1) * size[0] + step_row, num_col * size[1] + step_col:(num_col + 1) * size[1] + step_col]
    if (num_row + 1) * size[0] + step_row != imshape[0]:
        print('5drong!!')
    if (num_col + 1) * size[1] + step_col != imshape[1]:
        print('6drong!!')
    clipImg = Image.fromarray(clipArray)
    img_filepath = img_dir + '/' + img_name + "_" + str(
        num_row + 2) + "_" + str(num_col + 2) + "_label.png"
    clipImg.save(img_filepath)
if __name__=='__main__':
    save_dir = '/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/wy/data/FBP/756/label/train/'
    folder = os.path.exists(save_dir)
    if not folder:
        os.makedirs(save_dir)
    
    label_dir = '/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/wy/data/ori/Annotation__index/train/'
    imgs = glob.glob('{}*.png'.format(label_dir))
    for img in imgs:
        label_clip(img, [756, 756], save_dir)
    
    # save_dir = '/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/wy/data/FBP/756/image/test/'
    # folder = os.path.exists(save_dir)
    # if not folder:
    #     os.makedirs(save_dir)

    # img_dir = '/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/wy/data/ori/Image__8bit_NirRGB/test/'
    # imgs = glob.glob('{}*.tif'.format(img_dir))
    # for img in imgs:
    #     image_clip(img, [756, 756], save_dir)

