from PIL import ImageDraw
from PIL import Image
import cv2

def show_image_bounding_box_pil(image_fp, bounding_box):
    """
    Function to visualize the bounding box of the image wiht PIL
    :param image_fp:  full image path for jpg
    :param bounding_box: bounding box coordinates (x_center, y_center, width, height)
    """
    img = Image.open(image_fp)
    draw = ImageDraw.Draw(img)
    left = bounding_box[0] - bounding_box[2]
    top = bounding_box[1] - bounding_box[3]

    right = bounding_box[0] + bounding_box[2]
    bottom = bounding_box[1] + bounding_box[3]

    draw.rectangle((left, top, right, bottom), outline="red", width=2)

    img.show()


def show_image_bounding_box_cv2(image_fp, bounding_box):
    """
    Function to visualize the bounding box of the image wiht cv2
    :param image_fp:
    :param bounding_box:
    """
    img = cv2.imread(image_fp)
    left = bounding_box[0] - bounding_box[2]
    top = bounding_box[1] - bounding_box[3]

    right = bounding_box[0] + bounding_box[2]
    bottom = bounding_box[1] + bounding_box[3]

    cv2.rectangle(img,
                  (left, top),
                  (right, bottom),
                  color=(255, 0, 0),
                  thickness=2)

    # Display the image
    cv2.imshow("Image with Rectangle", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

