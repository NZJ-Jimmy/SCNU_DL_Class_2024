from fileinput import filename
import os

# 实现对多张图片的叠加
# 需要保证多张图片的尺寸一致

import PIL.Image as Image

def overlay_images(images, output_path):
    """
    Overlay multiple images into one image
    :param images: list of PIL.Image
    :param output_path: str
    :return: None
    """
    # make sure all images have the same size
    w, h = images[0].size
    for img in images:
        assert img.size == (w, h)

    # create a new image
    new_img = Image.new('RGBA', (w, h), (255, 255, 255, 0))

    n_imgs = len(images)
    i_imgs = 0

    for img in images:
        img = img.convert("RGBA")
        datas = img.getdata()

        new_data = []
        for item in datas:
            # change all white (also shades of whites)
            # to transparent
            if item[0] > 200 and item[1] > 200 and item[2] > 200:
                new_data.append((255, 255, 255, 0))
            else:
                new_data.append(item)

        img.putdata(new_data)
        alpha = img.split()[3]
        alpha = alpha.point(lambda p: p * ((i_imgs + 1) / n_imgs))
        img.putalpha(alpha)
        new_img = Image.alpha_composite(new_img, img)
        i_imgs += 1

    # Set the alpha channel of the final image to 100
    # alpha = new_img.split()[3]
    # alpha = alpha.point(lambda p: 255)
    # new_img.putalpha(alpha)
    new_img.save(output_path)
    print(f'Save overlay image to {output_path}')
    
# test overlay_images
if __name__ == '__main__':
    # Read all images from the 'out' directory
    image_dir = 'out'
    filename_list = []
    for i in range(15):
        filename = f'predict{i}.png'
        filename_list.append(filename)
    images = [Image.open(os.path.join(image_dir, filename)) for filename in filename_list]

    overlay_images(images, 'overlay.png')