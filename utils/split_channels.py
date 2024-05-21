import cv2


# split the image into its channels
def split_channels(img):
    # convert the image to hsv space and ycrcb space
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ycrcb_image = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    # split the images into its channels
    r, g, b = cv2.split(img)
    h, s, v = cv2.split(hsv_image)
    y, cr, cb = cv2.split(ycrcb_image)

    return r, g, b, h, s, v, y, cr, cb


if __name__ == "__main__":
    image = cv2.imread("data/airplane.jpg")
    r, g, b, h, s, v, y, cr, cb = split_channels(image)

    # save the channels as images
    cv2.imwrite("output/red_channel.png", r)
    cv2.imwrite("output/green_channel.png", g)
    cv2.imwrite("output/blue_channel.png", b)
    cv2.imwrite("output/hue_channel.png", h)
    cv2.imwrite("output/saturation_channel.png", s)
    cv2.imwrite("output/value_channel.png", v)
    cv2.imwrite("output/y_channel.png", y)
    cv2.imwrite("output/cr_channel.png", cr)
    cv2.imwrite("output/cb_channel.png", cb)
