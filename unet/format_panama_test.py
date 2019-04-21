

import os


#
# images = os.listdir('./panama_canal_test')
#
#
#
#
# from PIL import Image, ImageOps
#
#
# for img in images:
#
#     original_image = Image.open(os.path.join('./panama_canal_test',img))
#     size = (512, 512)
#     fit_and_resized_image = ImageOps.fit(original_image, size, Image.ANTIALIAS)
#



def reformat_image_to_skize(image,path,size = (512,512)):
    import os
    from PIL import Image, ImageOps
    original_image = Image.open(os.path.join(path, image))
    size = size
    fit_and_resized_image = ImageOps.fit(original_image, size, Image.ANTIALIAS)

    return fit_and_resized_image