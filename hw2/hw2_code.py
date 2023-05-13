import sys

from PIL import Image
import numpy as np
import pickle


def read_image(image_path):
    """
        This is the function to read image
    :param image_path: the relative path of the 40 image
    :return: the image object, and the pixel values (length = 5340)
    """
    image = Image.open(image_path)
    return image, list(image.getdata())


def histogram_setup(red_bit, green_bit, blue_bit):
    """
        Take the leftmost bit of blue (as Prof points out, people hard to distinguish blue color),
        and 2 leftmost bits of green and red. This will result into a total of 32 colors.
    :return: A numpy array that has shape (40, 4, 4, 2), 40 for each image, and the rest for possible values of
            each r, g, b lens
    """
    hist = np.zeros((40, 2 ** red_bit, 2 ** green_bit, 2 ** blue_bit))

    for i in range(1, 41):
        image_number = str(i).zfill(2)
        file = "images/i" + image_number + ".ppm"
        pixel_data = read_image(file)[1]
        for pixel in pixel_data:
            r, g, b = pixel
            r_binary, g_binary, b_binary = format(r, '08b'), format(g, '08b'), format(b, '08b')
            r_sub, g_sub, b_sub = r_binary[:red_bit], g_binary[:green_bit], b_binary[:blue_bit]
            r_result, g_result, b_result = int(r_sub, 2), int(g_sub, 2), int(b_sub, 2)
            hist[i - 1][r_result][g_result][b_result] += 1
    return hist


def distance_matrix_setup(hist, step):
    """
    This is the function that calculates the distances between each pair of images given the histogram information
    Different steps will have different ways of calculation.

    :param hist: a list of histograms, where each element of hist is a histogram of one image
    :param step: this is a indicator to check which step of the assignment we are working on.
                    this system has different ways of calculating distance in different steps.
    :return: a numpy 2d array that represents a 40 x 40 matrices. Each value represents the distance between
                a pair of images, and the lower the value the closer 2 images.
    """
    matrix_result = np.zeros((40, 40))
    for image1 in range(40):
        for image2 in range(40):
            if image2 == image1:
                continue
            img_hist1, img_hist2 = hist[image1], hist[image2]
            if step == 1 or step == 2:
                total_diff = np.sum(np.abs(img_hist1 - img_hist2))
                l1_distance = total_diff / (2 * 60 * 89)
                matrix_result[image1][image2] = l1_distance
            elif step == 3:
                img_hist1_data = np.array(img_hist1, dtype=int)
                img_hist2_data = np.array(img_hist2, dtype=int)
                matrix_result[image1][image2] = np.count_nonzero(img_hist1_data != img_hist2_data) \
                                                / (img_hist1_data.shape[0] * img_hist1_data.shape[1])

            elif step == 4:
                matrix_result[image1][image2] = np.abs(img_hist1 - img_hist2)

    return matrix_result


def find_min_3_images(matrix_result):
    """
    This function is to find the 3 image indices with the smallest distances for each row (image). In other words,
    for each row, this function will return the indices of the 3 closet images

    :param matrix_result: the 40 * 40 distance matrix
    :return: a numpy array with size 40 * 3, where each row represents an image. In each row, there will be 3 values
                of the image index, sorted by their distance in ascending order. The closest one will be the first.
    """
    result = np.zeros((matrix_result.shape[0], 3))
    for index, row in enumerate(matrix_result):
        min_indices = np.argpartition(row, 4)[:4]
        result[index] = min_indices[np.argsort(row[min_indices])[1:]]
    return result


def calculate_score(result, step):
    """
    This function extracts the crowd value of the detected 3 image indices

    :param result: the 40 * 3 numpy array that stores the indices of the 3 closest images for each image
    :return: a 40 * 3 numpy array that stores the actual crowd values of the 3 closest images for each image
    """
    if step == 6:
        crowd = np.loadtxt("step6.txt")
    else:
        crowd = np.loadtxt("Crowd.txt")

    score_list = np.zeros((40, 3))
    for i in range(result.shape[0]):
        three_indices = result[i].astype(int)
        row = crowd[i]
        values = row[three_indices]
        score_list[i] = values
    return score_list


def shrinking_explore():
    """
        This function is part of the step 1, but it does not get executed by the main function by default,
        because this function is to explore what is the optimal bits to use for red, green, and blue, which
        would maximize the final system score.

    :return: Nothing, the score of each combination value of bits of rgb would be printed.
    """
    for i in range(1, 5):
        for j in range(1, 5):
            for k in range(1, 5):
                hist = histogram_setup(i, j, k)
                matrix_result = distance_matrix_setup(hist, 1)
                min_indices = find_min_3_images(matrix_result)
                score_list = calculate_score(min_indices, 1)
                print(f"{i}, {j}, {k}")
                print(np.sum(score_list))


def output_result_system_crowd(min_indices, score_list, system_score, happinese, step):
    """
        This function is to output a html file that shows the closest 3 images that my system detected for each of
        the 40 images. The crowd score of each image, the sum of score of each row, and the total score of the
        entire system
    :param min_indices: The indices of the 3 closest images for each row
    :param score_list:  The crowd score of the 3 closest images for each row
    :param system_score:    The final system score
    :return:    Nothing, it outputs a html file directly in the same directory
    """
    html_text = "<html><body><table>\n"
    for i in range(40):
        html_text += "<tr>\n"
        image_number = str(i + 1).zfill(2)
        query_image_name = "images/i" + image_number + ".jpg"
        html_text += "<td><img src=\"" + query_image_name + "\"><br>" + image_number + "</td>\n"
        for j in range(3):
            num = int(min_indices[i][j])
            target_image_number = str(num + 1).zfill(2)
            target_image_name = "images/i" + target_image_number + ".jpg"
            html_text += "<td><img src=\"" + target_image_name + "\"><br>" + target_image_number + \
                         f" Crowd(q,t)={int(score_list[i][j])}" + "</td>\n"
        html_text += f"<td> the rows score is {int(np.sum(score_list[i]))} </td>\n</tr>"
    html_text += f"<p> The system score is {system_score} </p>"
    if step != 6:
        html_text += f"<p> The happinese value is {happinese} </p>"
    html_text += "</html></body></table>"

    file_name = f"output_system_crowd_{step}.html"
    with open(file_name, "w") as f:
        f.write(html_text)


def output_system_my_preference(min_indices):
    """
        This function is to output the number of overlap between my preference and the system results
    :param min_indices: The image indices that the system detected
    :return: the count of amount of overlap
    """
    my_preference = np.loadtxt("yj2737.txt")
    my_preference = my_preference - 1
    count = 0
    for i in range(min_indices.shape[0]):
        for j in min_indices[i]:
            if j in my_preference[i]:
                count += 1
    return count


def common_evaluation(matrix_result, step):
    """
        This function is called common_evaluation, since this function is a pipeline of evaluation process
        that could be repeatedly utilized for several steps. It basically just calls previous functions
        1. Find the image indices of min 3 for each row
        2. Based on the image indices, find the crowd scores of min 3 for each row
        3. Based on the crowd scores, calculate the final system score
        4. Print out the system score
        5. Output the system VS crowd results in a html file
        6. Output the system VS my preference results as an overlap count
    :param matrix_result: the 40 * 40 distance matrix, the step parameter is mainly used to deal with step 6
        because in step 6, the system score and the amount of overlap is doing th
    :return: the system score
    """
    # find the min 3 for each row
    min_indices = find_min_3_images(matrix_result)
    # calculate the score for each row
    score_list = calculate_score(min_indices, step)
    system_score = np.sum(score_list)
    print(f"The system score is about {system_score}")
    count_overlap = 0
    if step != 6:
        count_overlap = output_system_my_preference(min_indices)
        print(f"The number of overlap of my system and my own preference is {count_overlap}")
    output_result_system_crowd(min_indices, score_list, system_score, count_overlap, step)
    return system_score


def step1():
    """
        Careful that the (2,3,1) I used is found through the commented function
        "shrinking_explore()". I commented it out cuz it will print a lot of outputs. Avoid mess

        Essentially the first step part of this assignment
        1. Set up the 3d histogram structures for 40 images
        2. Find the l1 distance between each pair of image
        3. Find the min 3 of each image (each row)
        4. Find the score_list of each image and the system score
    :return:
    """
    # shrinking_explore()

    hist_list = histogram_setup(2, 3, 1)
    matrix_result = distance_matrix_setup(hist_list, 1)
    common_evaluation(matrix_result, 1)
    return matrix_result


def convert_gray_image():
    """
        This is the function that converts the original 40 images to gray image by taking the
        average of RGB values
    :return: a list of gray intensity image objects
    """
    gray_image_list = []
    for index in range(1, 41):
        image_number = str(index).zfill(2)
        file = "images/i" + image_number + ".ppm"
        image, pixel_data = read_image(file)
        gray_image = image.convert("L").copy()
        gray_image_data = np.array(image, dtype=int)
        for i in range(gray_image_data.shape[0]):
            for j in range(gray_image_data.shape[1]):
                gray_image.putpixel((j, i), int(np.sum(gray_image_data[i][j]) / 3))
        gray_image_list.append(gray_image)
    return gray_image_list


def convert_laplacian():
    """
        This is the function that converts grayscale images to (Laplacian) image using the
        3*3 [1,1,1,1,-8,1,1,1] kernel.
    :return: A list of the 40 Laplacian images in PIL Image object
    """
    gray_image_list = convert_gray_image()
    laplacian_image_list = []
    for index in range(40):
        gray_image = gray_image_list[index]
        # gray_image_copy = gray_image.copy()
        gray_image_data = np.array(gray_image, dtype=int)
        pad_image = np.pad(gray_image_data, pad_width=1, mode='constant', constant_values=0)
        for i in range(1, pad_image.shape[0] - 1):
            for j in range(1, pad_image.shape[1] - 1):
                pixel = pad_image[i][j]
                sum_value = sum(pad_image[x][y] for x in range(i - 1, i + 2) for y in range(j - 1, j + 2))
                pixel = pixel * 9 - sum_value
                pixel = np.abs(pixel)
                gray_image.putpixel((j - 1, i - 1), int(pixel))
        # gray_image.show()
        # gray_image_copy.show()
        laplacian_image_list.append(gray_image)
    return laplacian_image_list


def histogram_setup_step2(bit, lap_image_list):
    """
        This function will set up the 1d histogram structure in step2. It is similar to the histogram setup function
        in step1, but because there is no RGB in this step, I wrote a different function.

    :param bit: The number of bits taken from left to right.
    :param lap_image_list:  The list of the 40 Laplacian images
    :return:    A numpy array with size 40, 2**bit, where each row represents a 1d histogram of one image
    """
    hist = np.zeros((40, 2 ** bit))
    for i in range(len(lap_image_list)):
        image = lap_image_list[i]
        for pixel in list(image.getdata()):
            pixel_binary = format(pixel, '011b')
            pixel_sub = pixel_binary[:bit]
            pixel_result = int(pixel_sub, 2)
            hist[i][pixel_result] += 1
    return hist


def explore_bin_width():
    """
        This function will explore what bit will contribute to the highest system score, and it will not be
        executed by default, since I just used this function to find the best parameter.
    :return: Nothing, each run of bit value would be reported.
    """
    laplacian_image_list = convert_laplacian()
    for i in range(1, 12):
        print(f"{i}")
        hist_list = histogram_setup_step2(i, laplacian_image_list)
        matrix_result = distance_matrix_setup(hist_list, 2)
        # find the min 3 for each row
        min_indices = find_min_3_images(matrix_result)
        # calculate the score for each row
        score_list = calculate_score(min_indices, 2)
        system_score = np.sum(score_list)
        print(f"The system score is about {system_score}")
        count_overlap = output_system_my_preference(min_indices)
        print(f"The number of overlap of my system and my own preference is {count_overlap}")


def step2():
    """
        Be careful that I used 6 because it has the 2nd best score and a relatively lower bin size. I discovered
        it from the "explore_bin_width()" that is commented out in this function.

        Step 2 general workflow:
            1. convert 40 images to the gray images and then to the intensity images
            2. Develop the histogram structure of each image
            3. Calculate the distance matrix based on the histograms we obtained from the previous step
            4. Applied the same "common_evaluation" evaluation pipeline to get a result of system score,
                and output necessary html files.
    :return: the distance matrix
    """
    # explore_bin_width()

    laplacian_image_list = convert_laplacian()
    hist_list = histogram_setup_step2(6, laplacian_image_list)
    matrix_result = distance_matrix_setup(hist_list, 2)
    common_evaluation(matrix_result, 2)
    return matrix_result


def explore_black_boundary(input):
    """
        This function is to help explore what value might be a good threshold to distinguish between black and
        white pixels.
    :param input:   The input is a Laplacian image, I will use the 18th image, which is relatively easier to tell which
                    area is black, and which area is not black
    :return:    Nothing, the system will print out the pixel value and its previous pixel value
    """
    input_data = np.array(input, dtype=int)
    middle_row = input_data[int(input_data.shape[0] / 2)]
    for i in range(1, input_data.shape[1]):
        value = middle_row[i]
        prev_value = middle_row[i - 1]
        print(f"The difference between index position {i} with its previous value is {prev_value - value}")


def convert_binary(intensity_image, threshold):
    """
    This function will convert an intensity_image to a binary_image, where 255 means white and 0 means black.

    :param intensity_image: the list of intensity image as input
    :param threshold: a threshold value, where pixel that is greater than this value will be converted to white
    :return: A list of binary images.
    """
    binary_image_list = []
    for image in intensity_image:
        image_copy = image.copy()
        for x in range(image.width):
            for y in range(image.height):
                pixel = image.getpixel((x, y))
                if pixel > threshold:
                    image_copy.putpixel((x, y), 255)
                else:
                    image_copy.putpixel((x, y), 0)
        binary_image_list.append(image_copy)
    return binary_image_list


def explore_best_black_threshold():
    """
        This function will explore how different threshold values are going to affect the final system score, which
        would not be executed by default. Only the optimized parameter situation will be executed.
    :return: Nothing, score of each parameter value will be printed
    """
    gray_image_list = convert_gray_image()
    for i in range(40, 140):
        print(f"The threshold value is {i}")
        binary_image_list = convert_binary(gray_image_list, i)
        matrix_result = distance_matrix_setup(binary_image_list, 3)
        common_evaluation(matrix_result, 3)


def step3():
    """
        Be careful that the 78 as the optimized threshold value is discovered through the
        "explore_best_black_threshold()" function that I commented out.

        General step3:
            1. Get the intensity image list
            2. Convert the intensity images to the binary images
            3. Build up the distance matrix, there is a specific way to do it for step3
            4. Find the min 3 images with closest distance
            5. Output necessary html files
    :return: The distance matrix
    """
    # explore_best_black_threshold()

    gray_image_list = convert_gray_image()
    # # Use image 18 to find the black pixel threshold
    # explore_black_boundary(laplacian_image_list[17])

    # The best threshold value discovered from exploration is 78.
    binary_image_list = convert_binary(gray_image_list, 78)

    matrix_result = distance_matrix_setup(binary_image_list, 3)
    common_evaluation(matrix_result, 3)
    return matrix_result


def calculate_symmetry_value(binary_image_list):
    """
        This is the function to calculate the symmetry value of a binary image by comparing
        the left and right column-wise using the np.count_nonzero()

    :param binary_image_list: the list of 40 binary images
    :return: the symmetry score list of these 40 images
    """
    symmetric_score_list = []
    for image in binary_image_list:
        sum = 0
        image_data = np.array(image, dtype=int)
        for i in range(44):
            right = image.width - 1 - i
            sum += np.count_nonzero(image_data[:, i] != image_data[:, right])
        symmetric_score_list.append(2 * sum / (image.width * image.height))
    return symmetric_score_list


def explore_threshold_step4():
    """
        This function is to explore how different threshold values used to build up a binary image will affect the final
        system score in this step. This function will not be executed by default, only the best parameter will be used in
        the later step4() function
    :return: nothing.
    """
    gray_image_list = convert_gray_image()
    for i in range(40, 140):
        print(f"{i}")
        binary_image_list = convert_binary(gray_image_list, i)
        symmetric_score_list = calculate_symmetry_value(binary_image_list)
        matrix_result = distance_matrix_setup(symmetric_score_list, 4)
        common_evaluation(matrix_result, 4)


def step4():
    """
        Be careful that the threshold value 99 here is obtained from the "explore_threshold_step4" function
        that is commented out below.

        The general step4:
        1. Get the Laplacian image list
        2. Get the binary image list
        3. Calculate the symmetry score of each image in the previous-step list
        4. Calculate a distance matrix based on the symmetry value of each image
    :return: return the distance matrix result
    """
    # explore_threshold_step4()

    gray_image_list = convert_gray_image()
    binary_image_list = convert_binary(gray_image_list, 99)
    symmetric_score_list = calculate_symmetry_value(binary_image_list)
    matrix_result = distance_matrix_setup(symmetric_score_list, 4)
    common_evaluation(matrix_result, 4)
    return matrix_result


def dump_variables(matrix_list):
    """
    This function is used to dump my distance matrices obtained in the previous 4 steps, so I could directly access
    them in step5 and step6

    :param matrix_list: a list of color, texture, shape, symmetry distance matrices
    :return:
    """
    names = ["color_matrix.pickle", "texture_matrix.pickle", "shape_matrix.pickle", "symmetry_matrix.pickle"]
    for i in range(len(matrix_list)):
        name = "matrix_result/" + names[i]
        with open(name, 'wb') as f:
            pickle.dump(matrix_list[i], f)


def dump_load(address):
    """
    This function is used to load the 4 distance matrices
    :param address: the address of a pickle file
    :return: the distance matrix that this pickle file stored
    """
    with open(address, 'rb') as f:
        matrix = pickle.load(f)
    return matrix


def exploration_best_weights():
    """
        This function is to help explore the best weight vector that will return the optimized system score
        by exploring all the possible weight combinations

    :return: Nothing, print the max of the system score and help find the corresponded weight vector.
    """
    color_evaluation_matrix = dump_load("matrix_result/color_matrix.pickle")
    texture_evaluation_matrix = dump_load("matrix_result/texture_matrix.pickle")
    shape_evaluation_matrix = dump_load("matrix_result/shape_matrix.pickle")
    symmetry_evaluation_matrix = dump_load("matrix_result/symmetry_matrix.pickle")
    # The combination (0.83, 0.05, 0.07, 0.05) the score would be 11633,
    # The (0.38, 0.05, 0.52, 0.05) would be 11473, color starting from 0.38 seems not changing much
    # When fixing color and shape, using the (0.38, 0.05, 0.52, 0.05), I found there is a clear evidence that
    # the lower the texture, the better the system score, I set the texture to be fixed at 0.03
    # But this is the result I got from assuming shape + color = 0.9, now I want to narrow it down 0.8
    # It makes the new score to be 11661 (0.46, 0.03, 0.34, 0.17)  11661/25200 = 46%
    # This new 20% for symmetry + texture further confirm 0.03 is the max I should assign for texture
    # This new 30% for symmetry (0.64 0.03 0.07 0.27), 11682, this process keep going down to 40% for texture+symmetry
    # When 40%, the max score decrease from 11682, and the max at 0.41 of color
    # I discover, regardless of what weight distribution, when color distribution is around 40%, it will get a max value
    # There is only 1 exception at 0.64, so my new finding is color less than i expect, texture should be as low as 0.03
    # Then I iterated through all of the possibilities and get the result of (0.42, 0.2, 0.18, 0.2). 12124

    weight_vector = [0.44, 0.09, 0.23, 0.24]
    result = []
    for texture in range(0, 101):
        t = texture / 100
        rest1 = 1 - t
        rest1_percentage = int(rest1 * 100 + 1)
        for j in range(0, rest1_percentage):
            color = j / 100
            rest = rest1 - color
            rest_percentage = int(rest * 100 + 1)
            for i in range(0, rest_percentage):
                value = i / 100
                value2 = rest - value
                print(f"Color: {color}, texture {t}, shape {value}, symmetry {value2}")
                final_result = color * color_evaluation_matrix + \
                               t * texture_evaluation_matrix + \
                               value * shape_evaluation_matrix + \
                               value2 * symmetry_evaluation_matrix
                score = common_evaluation(final_result, 5)
                # if score == 12852:
                #     sys.exit()
                result.append(score)
    print(max(result))


def step5():
    """
        This function will show the performance of the optimized weight vector and the system score results
        Be careful that the weight vector result is found by the "exploration_best_weights" function that I commented
        out

    :return: Nothing, important results will be printed out
    """
    # exploration_best_weights()

    color_evaluation_matrix = dump_load("matrix_result/color_matrix.pickle")
    texture_evaluation_matrix = dump_load("matrix_result/texture_matrix.pickle")
    shape_evaluation_matrix = dump_load("matrix_result/shape_matrix.pickle")
    symmetry_evaluation_matrix = dump_load("matrix_result/symmetry_matrix.pickle")

    weight_vector = [0.44, 0.09, 0.23, 0.24]
    final_result = weight_vector[0] * color_evaluation_matrix + \
                   weight_vector[1] * texture_evaluation_matrix + \
                   weight_vector[2] * shape_evaluation_matrix + \
                   weight_vector[3] * symmetry_evaluation_matrix
    common_evaluation(final_result, 5)


def construct_sparse_matrix():
    """
        This is the function to construct the sparse matrix for step6 by manipulating a 2d numpy array
    :return: nothing, the sparse matrix will be outputted as a txt file called "step6.txt"
    """
    source = np.zeros((40, 40), dtype=int)
    my_result = np.loadtxt("yj2737.txt")
    for i in range(my_result.shape[0]):
        for j in range(1, my_result.shape[1]):
            index = int(my_result[i][j]) - 1
            source[i][index] = 4 - j
    np.savetxt("step6.txt", source)


def explore_my_preference():
    """
        This is a function to explore the weight vector that will optimize the final system score that is built
        upon on my own preference file. It has a similar process as exploration_best_weights()
    :return: nothing, important result is printed out
    """
    construct_sparse_matrix()
    color_evaluation_matrix = dump_load("matrix_result/color_matrix.pickle")
    texture_evaluation_matrix = dump_load("matrix_result/texture_matrix.pickle")
    shape_evaluation_matrix = dump_load("matrix_result/shape_matrix.pickle")
    symmetry_evaluation_matrix = dump_load("matrix_result/symmetry_matrix.pickle")
    result = []
    for texture in range(0, 101):
        t = texture / 100
        rest1 = 1 - t
        rest1_percentage = int(rest1 * 100 + 1)
        for j in range(0, rest1_percentage):
            color = j / 100
            rest = rest1 - color
            rest_percentage = int(rest * 100 + 1)
            for i in range(0, rest_percentage):
                value = i / 100
                value2 = rest - value
                print(f"Color: {color}, texture {t}, shape {value}, symmetry {value2}")
                final_result = color * color_evaluation_matrix + \
                               t * texture_evaluation_matrix + \
                               value * shape_evaluation_matrix + \
                               value2 * symmetry_evaluation_matrix
                count = common_evaluation(final_result, 6)
                if count == 123:
                    return  # best value is 118 with combination of (0.48, 0.33 0.18 0.01) # 118
                result.append(count)
    print(max(result))


def step6():
    """
        This is the function to present my result in step6. Be careful that the linear weight vector
        of [0.31, 0.32, 0.33, 0.04] is calculated through the function explore_my_preference()
    :return: nothing, the important information will be printed.
    """

    # explore_my_preference()

    color_evaluation_matrix = dump_load("matrix_result/color_matrix.pickle")
    texture_evaluation_matrix = dump_load("matrix_result/texture_matrix.pickle")
    shape_evaluation_matrix = dump_load("matrix_result/shape_matrix.pickle")
    symmetry_evaluation_matrix = dump_load("matrix_result/symmetry_matrix.pickle")

    weight_vector = [0.31, 0.32, 0.33, 0.04]

    final_result = weight_vector[0] * color_evaluation_matrix + \
                   weight_vector[1] * texture_evaluation_matrix + \
                   weight_vector[2] * shape_evaluation_matrix + \
                   weight_vector[3] * symmetry_evaluation_matrix
    common_evaluation(final_result, 6)


if __name__ == '__main__':
    # color_evaluation_matrix = step1()
    # texture_evaluation_matrix = step2()
    # shape_evaluation_matrix = step3()
    # symmetry_evaluation_matrix = step4()
    # dump_variables([color_evaluation_matrix, texture_evaluation_matrix, shape_evaluation_matrix, symmetry_evaluation_matrix])
    #
    # step5()
    step6()
