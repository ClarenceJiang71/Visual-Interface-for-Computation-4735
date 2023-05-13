import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os


def load_image(image_address):
    """
    This is the method to load the image and return an image object in numpy array form

    :param image_address: a string that records the path of the source image.
    :return: a numpy array format image object, with RGB values
    """
    img = cv.imread(image_address)
    return img


def get_dimension(image):
    """
    This is a simple method the returns the height and width of an image

    :param image: a numpy array format image object
    :return: (height, width) of the numpy array image object
    """
    return image.shape[0], image.shape[1]


def data_reduction(image, original_height, origin_width):
    """
    This is the method that reduced an original image into a smaller size, the way I reduced it
    is resizing the original image into a 10% version. In other words, the resize image's height
    will be 0.1 * original height, and new width will be 0.1 * original width

    :param image: a numpy array image object
    :param original_height: the height of the image, in integer form
    :param origin_width: the width of the image, in integer form
    :return: a resized image with dimension (0.1 * original height, 0.1* original width)
    """
    new_width = int(origin_width*0.1)
    new_height = int(original_height*0.1)
    resized_image = cv.resize(image, (new_width, new_height), interpolation=cv.INTER_AREA)
    return resized_image


def check_color_distribution(resized_image):
    """
    This function is the first part of the process that explores how the RGB value changes
    when going from a black background to a white hand.

    Step: Create a 2d numpy array that each data value = the average of RGB value

    :param resized_image: a numpy array image object of a splay-center image in shape of (height, width, 3)
    :return: a 2d numpy array image in shape of (height, width), each value is the average of RGB values in
                resized_image
    """
    converted_2d_average = np.empty([resized_image.shape[0], resized_image.shape[1]])
    for i in range(resized_image.shape[0]):
        for j in range(resized_image.shape[1]):
            value = np.mean(resized_image[i][j])
            converted_2d_average[i][j] = value
    return converted_2d_average


def explore_color_jump_hand(data):
    """
    This function is the second part of the process that explores how the RGB value changes
    when going from a black background to a white hand.

    Steps:
    1. Focus on the middle row
        (since the image is a splayed hand in the center, there has to be a part of hand in the middle row)
    2. Draw a line plot that shows how the average RGB value changes as we move from left to right through the image
    3. We know it's going to go from black to white, so from observing the line plot, we could tell a threshold
        value that it needs to exceed to be identified as "hand area"
    4. Save the line plot for further use just in case

    :param data: the 2d numpy array image object returned from "check_color_distribution" function
    :return: no return, but saved a line plot demonstrating how the average RGB value changes horizontally
    """

    # Set the figure size
    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True

    data_half = data[int(data.shape[0]/2)]
    fig = plt.figure(figsize=(10, 7))
    plt.plot(range(len(data_half)), data_half)
    plt.title("column value change VS Average RGB value (row fixed at mid index)")
    plt.show()
    fig.savefig("output&result/line_plot1.png")


def convert_to_binary(image):
    """
    This is a function that takes an image and returns a new image object that is a binary intermediate form
    of it. I do not change the original image to avoid overwriting it

    Step:
        if the average RGB value > 131 (the threshold value observed from the line plot),
        the value of this (i, j) position in the binary image will be assigned as (255,255,255),
        which is a white pixel. Otherwise, the value at this (i, j) will be assigned as
        (0, 0, 0), which is a black pixel


    :param image: a numpy array image returned from the "data_reduction" function,
                    its shape should be (height, width, 3)
    :return: a numpy array image object that has only 2 possible values (255,255,255) or (0,0,0)
    """
    image_copy = image.copy()
    for i in range(image_copy.shape[0]):
        for j in range(image_copy.shape[1]):
            if np.mean(image_copy[i][j]) > 131:
                image_copy[i][j] = 255
            else:
                image_copy[i][j] = 0
    return image_copy


def get_location(image):
    """
    The idea is quite straightforward, regardless of what gesture the hand makes, the
    first white pixel throughout the numpy 2d array is quite likely be around the middle
    of the hand in the x-axis direction. If this x-value is close to the center of the image
    with a tolerable range, then it will be recognized as center

    The first half of code does not use left and right point because with just the top and bottom
    points, we could already get the location, but for determining the shape of the hand
    we need both the left and right points to draw the rectangle

    【More details explained within the code】

    :param image: a binary numpy array image object ( only 2 possible values (255,255,255) or (0,0,0) )
    :return: the location label (e.g. upper left),
                and the upper left and lower right coordinate values of a bounding rectangle

    """
    height, width = image.shape[0], image.shape[1]
    labels = ["center", "upper left", "upper right", "lower left", "lower right"]
    location = ""
    top_point = ()
    low_point = ()
    break_out_flag = False

    # Find the top point
    for i in range(height):
        for j in range(width):
            if image[i][j][0] == 255:
                top_point = (i,j)
                break_out_flag = True
                break
        if break_out_flag:
            break_out_flag = False
            break

    # Find the low point
    for i in reversed(range(height)):
        for j in reversed(range(width)):
            if image[i][j][0] == 255:
                low_point = (i,j)
                break_out_flag = True
                break
        if break_out_flag:
            break_out_flag = False
            break

    # if the x-axis value of the top point is in the range of 0.5 * width with ± 0.15, it is considered a center
    if int(0.35 * width) <= top_point[1] <= int(0.65 * width):
        location = labels[0]
    # if the x-axis value of the top point is <= 0.3 * width and its y-axis value <= 0.3 * height
    # it is considered an upper left
    elif top_point[1] <= int(width * 0.3) and top_point[0] <= int(0.3 * height):
        location = labels[1]
    # if the x-axis value of the top point is <= 0.3 * width and its y-axis value >= 0.7 * height
    # it is considered an lower left
    elif top_point[1] <= int(width * 0.3) and low_point[0] >= int(0.7 * height):
        location = labels[3]
    # if the x-axis value of the top point is >= 0.7 * width and its y-axis value <= 0.3 * height
    # it is considered an upper right
    elif top_point[1]>=int(width*0.7) and top_point[0] <= int(0.3*height):
        location = labels[2]
    # if the x-axis value of the top point is >= 0.7 * width and its y-axis value >= 0.7 * height
    # it is considered an lower right
    elif top_point[1] >= int(width*0.7) and low_point[0] >= int(0.7 * height):
        location = labels[4]
    # for gap that could be considered to more than 1 scenario, return ambiguous
    else:
        location = "ambiguous"

    # Extract the left point and right point in a similar way to help draw the bounding rectangle later
    left_point = ()
    right_point = ()

    for i in range(width):
        for j in range(height):
            if image[j][i][0] == 255:
                left_point = (j, i)
                break_out_flag = True
                break
        if break_out_flag:
            break_out_flag = False
            break

    for i in reversed(range(width)):
        for j in reversed(range(height)):
            if image[j][i][0] == 255:
                right_point = (j, i)
                break_out_flag = True
                break
        if break_out_flag:
            break

    # Find the upper left and lower right corner coordinates needed to draw rectangle
    x_start = left_point[1]
    y_start = top_point[0]
    x_end = right_point[1]
    y_end = low_point[0]

    return location, (x_start, y_start, x_end, y_end)


def find_shape(location_index, binary_image):
    """
    This function use 3 criteria together to determine if a hand gesture falls into which of the 4
    categories, which are "splayed hand", "fist", "palm", and "unknown".

    1st criterion: area of bounding rectangle
    2nd criterion: number of black pixels in the rectangle / number of total pixels in the rectangle
    3rd criterion: height/width ratio (or width/height ratio if width is the larger one)

    The logic to get a splayed hand:
    it should have large area, since its fingers stretched more than a palm or a fist.
    My experiments show that it will be remarkably greater than the other 2.
    I will also check if this hand has a decent proportion of black pixels in the bounding rectangle,
    because a splayed hand will have a lot of gaps between fingers so a splayed hand needs to
    satisfy 2 conditions: a large area + a sufficient proportion of black pixels

    The logic to get a palm:
    it should not have large area, since its fingers are tightened more, causing a smaller width than
    a splayed hand. I will also check if its height/width ratio (or width/height ratio if width is
    the larger one) is large enough to distinguish it from a fist, so a palm needs to satisfy 2 conditions:
    a not large area + a high (height/width) ratio

    The logic to get a fist:
    it should not have large area, since it's roughly part of a splayed hand. Also, it could not have a high
    height/width ratio, since its shape is more close to a square. I will also use the black pixel
    proportion to help determine fist, because a fist is people holding hand, which means there should be less
    gap between fingers. The black pixel proportion should be small, so a fist needs to satisfy 3 conditions:
    a not large area + a not high (height/width) ratio + a small proportion of black pixels

    【Detailed values are obtained with several tests and experiments】


    :param location_index: the upper left and lower right coordinates of the bounding rectangle
    :param binary_image: the binary image
    :return: shape label such as "fist" or "splay"

    """
    # Calculate area
    width = location_index[2] - location_index[0]
    height = location_index[3] - location_index[1]
    area = width * height

    # Count the number of black pixels
    count_black = 0
    for i in range(location_index[1], location_index[3] + 1):
        for j in range(location_index[0], location_index[2]+1):
            if binary_image[i][j][0] == 0:
                count_black += 1

    if area > 8100 and count_black/area > 0.45:
        return "splay"
    elif area < 8100 and (height/width > 1.8 or width/height > 1.8):
        return "palm"
    elif area < 8100 and count_black/area < 0.45:
        return "fist"
    else:
        return "unknown"


def output_image_result(filename, image):
    """
    This function is helping me save my binary intermediate image results with bounding rectangle
    and text label such as "center splay". Be careful with the change directory part within this
    code, since I saved images into another folder.

    :param filename: the
    :param image: the binary image in numpy array format with rectangle and text both included
    :return: no return
    """
    os.chdir("output&result")
    cv.imwrite(filename, image)
    os.chdir("..")


def test_everything_before_step_4():
    """
    The function is just doing everything before step 4, it basically just calls over
    almost all functions written above, and gradually build up my final system

    :return: no return, but output a lot of binary intermediate images
    """
    # image_list is a list of all images I used throughout step 1 to step 3
    image_list = [
                  "hw1_images/splay_center.jpeg", "hw1_images/splay_lower_left.jpeg", "hw1_images/fist_center2.jpeg",
                  "hw1_images/fist_center.jpeg", "hw1_images/splay_upper_right.jpeg",
                  "hw1_images/fist_center_false_negative.jpeg", "hw1_images/fist_center_false_positive.jpeg",
                  "hw1_images/splay_upper_right_false_negative.jpeg",
                  "hw1_images/splay_upper_right_false_positive.jpeg",
                  "hw1_images/unknown_c_shape.jpeg",
                  "hw1_images/unknown_bird_shape.jpeg",
                  "hw1_images/palm3.jpeg",
                  "hw1_images/palm1.jpeg",
                  "hw1_images/palm4.jpeg",
                  "hw1_images/palm5.jpeg",
                  "hw1_images/unknown_ok.jpeg",
                  "hw1_images/unknown_circle.jpeg"
                  ]

    # Iterate through each image to let the system recognize the "what" and "where" of it
    for i in image_list:
        # load the image into a numpy array format
        image = load_image(i)

        # Get dimension and reduce the dimensions of the image to make computation faster
        height, width = get_dimension(image)
        resized_image = data_reduction(image, height, width)

        """
        These 2 functions are used to explore a pixel threshold value to distinguish between 
        hand and a black background. After running this code, we will extract the threshold value,
        so there is no need to run them again, but it is definitely a key step that needs to read over.
        Details are commented within these 2 functions
        """
        # converted_2d_average = check_color_distribution(resized_image)
        # explore_color_jump_hand(converted_2d_average)

        # Convert a resized version of the original image into binary intermediate form
        binary_image = convert_to_binary(resized_image)

        # find the location and related bounding coordinates of a binary image, more details in function
        location_label, location_index = get_location(binary_image)

        # find the shape of the hand
        shape_label = find_shape(location_index, binary_image)

        # Add the predicted location and shape in text form on the binary intermediate image
        cv.putText(img=binary_image, text=location_label + " " + shape_label,
                   org=(int(binary_image.shape[1] / 2) - 100, 30),
                   fontFace=cv.FONT_HERSHEY_TRIPLEX, fontScale=0.4,
                   color=(0, 255, 0), thickness=1)

        # Add bounding rectangle
        cv.rectangle(binary_image, (location_index[0], location_index[1]), (location_index[2], location_index[3]),
                     (255, 0, 0), 4)

        # The binary intermediate file naming principle = name of original image + _binary.png
        index = i.index("/")
        name = i[index + 1:-5] + "_binary.png"
        output_image_result(name, binary_image)
        cv.imshow("Display window", binary_image)

        # waitKey helps you have a look at your image. I commented it out.
        # if you want to take a look, then un-comment it
        # cv.waitKey(0)


def open_lock(state_list, image_list):
    """
    This function is figuring out if a list of image will be correctly recognized by my system
    given a list of ground truth labels. If each image exactly matches with the corresponded
    ground truth label in order. It will print a message saying "lock open", otherwise "not open"

    The process of extracting location and shape of an image is the same as shown in the
    "test_everything_before_step_4" function.

    :param state_list: the ground truth labels in string form, e.g ["center fist", "center fist"]
    :param image_list: a list of image address in string form, e.g ["my_hand1.jpeg", "my_hand2.jpeg"]
    :return: a boolean variable indicating if lock opened
    """
    lock_open = True

    for i in range(len(state_list)):
        location = state_list[i].split(",")[0]
        shape = state_list[i].split(",")[1]
        image_address = image_list[i]
        image = load_image(image_address)
        height, width = get_dimension(image)
        resized_image = data_reduction(image, height, width)
        binary_image = convert_to_binary(resized_image)
        location_label, location_index = get_location(binary_image)
        shape_label = find_shape(location_index, binary_image)
        print(f"The expected location of image {i+1} is {location} and the expected shape is {shape}")
        print(f"The system sees the location image {i+1} as {location_label} and shape as {shape_label}")

        cv.putText(img=binary_image, text=location_label + " " + shape_label,
                   org=(int(binary_image.shape[1] / 2) - 100, 30),
                   fontFace=cv.FONT_HERSHEY_TRIPLEX, fontScale=0.4,
                   color=(0, 255, 0), thickness=1)
        cv.rectangle(binary_image, (location_index[0], location_index[1]), (location_index[2], location_index[3]),
                     (255, 0, 0), 4)
        index = image_address.index("/")
        name = image_address[index + 1:-5] + "_binary.png"
        output_image_result(name, binary_image)
        cv.imshow("Display window", binary_image)

        # Check if either location or shape is incorrect, if it is incorrect, change lock_open to False
        if location_label != location or shape_label != shape:
            # if lock_open is already False, then do nothing.
            if lock_open:
                lock_open = False

    print("-" * 50)
    if lock_open:
        print("Open lock, everything is correct")
    else:
        print("Not open lock, there is an error")
    print()
    return lock_open


def my_creative_step_easy():
    """
    This function has 3 images to test and 3 ground truth given. It will pass
    these 2 lists as parameters to the open_lock function to get results.

    :return: no return, but results of recognizing these 3 images will be printed out
    """
    print("This is my easy sequence of the step 4")
    image_list = ["hw1_images/s4_splay_center1.jpeg",
                  "hw1_images/s4_splay_center2.jpeg",
                  "hw1_images/s4_splay_center3.jpeg"]
    state_list = [
        "center,splay",
        "center,splay",
        "center,splay"
    ]
    open_lock(state_list, image_list)


def friend_creative_step_easy():
    """
    This function has 3 images to test and 3 ground truth given. It will pass
    these 2 lists as parameters to the open_lock function to get results.

    :return: no return, but results of recognizing these 3 images will be printed out
    """
    print("This is my friend's easy sequence of the step 4")
    image_list = ["hw1_images/s4_friend_easy_splay_center.jpeg",
                  "hw1_images/s4_friend_easy_fist_center.jpeg",
                  "hw1_images/s4_friend_easy_palm_center.jpeg"]
    state_list = [
        "center,splay",
        "center,fist",
        "center,palm"
    ]
    open_lock(state_list, image_list)


def my_creative_step_difficult():
    """
    This function has 3 images to test and 3 ground truth given. It will pass
    these 2 lists as parameters to the open_lock function to get results.

    :return: no return, but results of recognizing these 3 images will be printed out
    """
    print("This is my difficult sequence of the step 4")
    image_list = ["hw1_images/s4_my_difficult_splay_center.jpeg",
                  "hw1_images/s4_my_difficult_fist_center.jpeg",
                  "hw1_images/s4_my_difficult_palm_center.jpeg"]
    state_list = [
        "center,splay",
        "center,fist",
        "center,palm"
    ]
    open_lock(state_list, image_list)


def friend_creative_step_difficult():
    """
    This function has 3 images to test and 3 ground truth given. It will pass
    these 2 lists as parameters to the open_lock function to get results.

    :return: no return, but results of recognizing these 3 images will be printed out
    """
    print("This is my friend's difficult sequence of the step 4")
    image_list = ["hw1_images/s4_friend_difficult_splay_center.jpeg",
                  "hw1_images/s4_friend_difficult_fist_center.jpeg",
                  "hw1_images/s4_friend_difficult_palm_center.jpeg"]
    state_list = [
        "center,splay",
        "center,fist",
        "center,palm"
    ]
    open_lock(state_list, image_list)


def my_creative_step_interesting():
    """
    This function has 3 images to test and 3 ground truth given. It will pass
    these 2 lists as parameters to the open_lock function to get results.

    :return: no return, but results of recognizing these 3 images will be printed out
    """
    print("This is my interesting sequence of the step 4")
    image_list = ["hw1_images/s4_my_interesting_splay_center.jpeg",
                  "hw1_images/s4_my_interesting_fist_center.jpeg",
                  "hw1_images/s4_my_interesting_palm_center.jpeg"]
    state_list = [
        "center,splay",
        "center,fist",
        "center,palm"
    ]
    open_lock(state_list, image_list)


def friend_creative_step_interesting():
    """
    This function has 3 images to test and 3 ground truth given. It will pass
    these 2 lists as parameters to the open_lock function to get results.

    :return: no return, but results of recognizing these 3 images will be printed out
    """
    print("This is my friend's interesting sequence of the step 4")
    image_list = ["hw1_images/s4_friend_interesting_palm_center.jpeg",
                  "hw1_images/s4_friend_interesting_fist_center1.jpeg",
                  "hw1_images/s4_friend_interesting_fist_center2.jpeg"]
    state_list = [
        "center,palm",
        "center,fist",
        "center,fist"
    ]
    open_lock(state_list, image_list)


if __name__ == '__main__':
    # This is the function that explores the system setup from step 1 to step 3
    # Running this function will not print anything, but it will create tons of binary intermediate images
    # The images will not be shown in a pop-up window, but if you want to have a look of all the images
    # involved in this process, uncomment the code on line 364
    test_everything_before_step_4()

    # This 6 functions correspond to the 6 tests in step4
    my_creative_step_easy()
    friend_creative_step_easy()
    my_creative_step_difficult()
    friend_creative_step_difficult()
    my_creative_step_interesting()
    friend_creative_step_interesting()



