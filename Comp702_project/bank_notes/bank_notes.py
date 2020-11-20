import cv2
import sys
import os
import glob
import numpy as np
import mahotas as mt
from sklearn.svm import LinearSVC
from math import copysign, log10
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt



# function to compute histogram equalization
def histogram_equalization(Image):
    Image_Height = Image.shape[0]
    Image_Width = Image.shape[1]

    Histogram = np.zeros([256], np.int32)

    for x in range(0, Image_Height):
        for y in range(0, Image_Width):
            Histogram[Image[x, y]] += 1

    return Histogram


# function to extract haralick textures from an image
def extract_features(image1):
    # calculate haralick texture features for 4 types of adjacency

    textures = mt.features.haralick(image1)

    # take the mean of it and return it
    ht_mean = textures.mean(axis=0)
    return ht_mean


def main():
    print("-----------------Calculation of Humoments and Haralick features------------")
    sample_vector = []
    feature_vector1 = [[]]
    showLogTransformedHuMoments = True
    p = ""
    a1 = "dataset1/train/10Rand"
    a2 = "dataset1/train/20Rand"
    a3 = "dataset1/train/50Rand"
    a4 = "dataset1/train/100Rand"
    a5 = "dataset1/train/200Rand"


    for i in range(5):

        if i == 0:
            p = a1
        elif i == 1:
            p = a2
        elif i == 2:
            p = a3
        elif i == 3:
            p = a4
        else:
            p = a5
        path = glob.glob(p + "/*.jpg")

        for img in path:

            filename = img
            im = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            _, im = cv2.threshold(im, 128, 255, cv2.THRESH_BINARY)
            moment = cv2.moments(im)
            huMoments = cv2.HuMoments(moment)
            print("{}: ".format(filename))

            sample_features = extract_features(im)
            for k in range(len(sample_features)):
                sample_vector.append(sample_features[k])

            for i in range(0, 7):
                if showLogTransformedHuMoments:
                    # Log transform Hu Moments to make
                    # squash the range
                    print("{:.5f}".format(-1 * copysign(1.0, huMoments[i]) * log10(abs(huMoments[i]))))
                    sample_vector.append(-1 * copysign(1.0, huMoments[i]) * log10(abs(huMoments[i])))
                else:
                    # Hu Moments without log transform
                    print("{:.5f}".format(huMoments[i]))
                    sample_vector.append(-1 * copysign(1.0, huMoments[i]) * log10(abs(huMoments[i])))
            print()

            feature_vector1.append(sample_vector)
            sample_vector = []

    print("----------------------DONE Calculating Hu moments and haralick features------------------")

    #print(feature_vector1[2])  # vector to contain all the features from hu moments and harakick

    print("----------Calculation of Hu Moments and feature extraction for input image--------------")
    '''
    let us now calculate the input image hu-moments and haralick features
    '''
    #showLogTransformedHuMoments2 = True
    path = glob.glob("input/test" + "/*.jpg")
    input_vector = []
    feature_vector2 = []
    for img in path:

        filename = img
        im = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        _, im = cv2.threshold(im, 128, 255, cv2.THRESH_BINARY)
        moment = cv2.moments(im)
        huMoments = cv2.HuMoments(moment)
        print("{}: ".format(filename))

        input_features = extract_features(im)
        for k in range(len(input_features)):
            input_vector.append(input_features[k])

        for i in range(0, 7):
            if showLogTransformedHuMoments:
                # Log transform Hu Moments to make
                # squash the range
                print("{:.5f}".format(-1 * copysign(1.0, huMoments[i]) * log10(abs(huMoments[i]))))
                input_vector.append(-1 * copysign(1.0, huMoments[i]) * log10(abs(huMoments[i])))
            else:
                # Hu Moments without log transform
                print("{:.5f}".format(huMoments[i]))
                input_vector.append(-1 * copysign(1.0, huMoments[i]) * log10(abs(huMoments[i])))
        print()
    print("--------------------------------End of calculation---------------------------------------------------")

    for i in input_vector:
        feature_vector2.append(i)

    print("-----------------------------------Feature vector-------------------------------------------------------")
    print(feature_vector2)

    print("-----------------------------------Feature vector-------------------------------------------------------")

    found = True
    count = 0
    v1 = []
    v2 = []
    i = 0

    while i < len(feature_vector1):

        for j in feature_vector1[i]:
            v1.append(j)
        for k in feature_vector2:
            v2.append(k)
        print("------------------------v1-- 20 Features from the dataset image----------------")
        print(v1)

        print(v2)
        print("------------------------v2 -- 20 Features from the Input image ------------------")
        if len(v1) > 0:

            for c in range(20):
                if v1[c] == v2[c]:
                    count = count + 1

        if count == 21:
            found = True
            print("Image found !!!!!!")
            break
        count = 1
        i = i + 1
        v1 = []
        v2 = []

    '''
    In the following operation we try to put a label on the output image
    '''
    print("--------------Linear SVM Classifier-----------------------------")

    # load the training dataset
    train_path = "dataset1/train"
    train_names = os.listdir(train_path)

    # empty list to hold feature vectors and train labels
    train_features = []
    train_labels = []

    # loop over the training dataset
    print "[STATUS] Started extracting haralick textures.."
    for train_name in train_names:
        cur_path = train_path + "/" + train_name
        cur_label = train_name
        i = 1

        for file1 in glob.glob(cur_path + "/*.jpg"):
            print "Processing Image - {} in {}".format(i, cur_label)
            # read the training image
            image = cv2.imread(file1)

            # convert the image to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # extract haralick texture from the image
            features = extract_features(gray)

            # append the feature vector and label
            train_features.append(features)
            train_labels.append(cur_label)

            # show loop update
            i += 1
    # have a look at the size of our feature vector and labels
    print "Training features: {}".format(np.array(train_features).reshape(1, -1))
    print "Training labels: {}".format(np.array(train_labels).reshape(1, -1))

    # create the classifier
    print "[STATUS] Creating the classifier.."

    clf_svm = LinearSVC(max_iter=5000, random_state=9)


    # fit the training data and labels
    print "[STATUS] Fitting data/label to model.."
    clf_svm.fit(train_features, train_labels)

    # loop over the test images
    test_path = "input/test"
    for file1 in glob.glob(test_path + "/*.jpg"):
        # read the input image
        image = cv2.imread(file1)

        # convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # extract haralick texture from the image
        features = extract_features(gray)


        if found:
            # evaluate the model and predict label
            prediction = clf_svm.predict(features.reshape(1, -1))[0]

            # show the label
            cv2.putText(image, prediction, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

            print "Prediction - {}".format(prediction)
            # display the output image
            cv2.imshow("Test_Image", image)
            cv2.waitKey(0)

        else:

            # show the label
            results = "image not found"
            cv2.putText(image, results, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
            print "Prediction - {}".format(results)
            # display the output image
            cv2.imshow("Test_Image", image)
            cv2.waitKey(0)




if __name__ == "__main__":
    main()
