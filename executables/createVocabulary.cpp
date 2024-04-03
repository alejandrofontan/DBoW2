/**
 * File: CreateVocabulary.cpp
 * Date: March 2024
 * Original Author: Dorian Galvez-Lopez
 * Modified by Alejandro Fontan Villacampa for AnyFeature-VSLAM
 * Description: create binary vocabulary application of DBoW2
 * License: see the LICENSE.txt file
 */

enum DescriptorType {
    DESC_KAZE64 = 4,
    DESC_SURF64 = 3,
    DESC_BRISK = 2,
    DESC_AKAZE61 = 1,
    DESC_ORB = 0
};

#include <iostream>
#include <vector>

// DBoW2
#include "DBoW2.h"

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

#include <fstream>

using namespace DBoW2;
using namespace std;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
std::vector<std::vector<std::string>> read_txt(const std::string &filePath, const size_t &numCols, char delimiter ,int headerRows);
void displayProgressBar(int width, double progressPercentage);

void loadBinaryFeatures(vector<vector<cv::Mat>> &features, const std::vector<std::string>& imagePaths);
void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out);
void testBinaryVocCreation(const vector<vector<cv::Mat>> &features);

template <typename T>
void loadNonBinaryFeatures(vector<vector<T>> &features, const std::vector<std::string>& imagePaths);
template <typename T>
void changeStructure(const cv::Mat &plain, vector<T> &out);
template <typename T>
void testNonBinaryVocCreation(const vector<vector<T>> &features);

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
int numberOfImages{};
string savePath{""};
string vocName{""};
int descriptorId{};
string descriptorName{""};
int k = 9; // branching factor
int L = 3; // depth levels
const WeightingType weight = TF_IDF;
const ScoringType scoring = L1_NORM;
const string script_label = "[createVocabulary.cpp] ";

// ----------------------------------------------------------------------------
int main(int argc,char **argv){
    if(argc != 8){
        cerr << endl << "Usage: ./createVocabulary descriptorName isBinary descriptorId savePath imageFolder " << endl;
        return 1;
    }

    descriptorName = argv[1];
    bool isBinary = bool(stoi(argv[2]));
    descriptorId = stoi(argv[3]);
    savePath = argv[4];
    string rgbTxt = argv[5];
    k =  stoi(argv[6]); // branching factor
    L = stoi(argv[7]); // depth levels

    vocName = descriptorName + "_DBoW2";

    // Load images
    std::vector<std::string> imagePaths;
    std::vector<std::vector<std::string>> imagesTxt = read_txt(rgbTxt,1,' ',0);
    for(size_t imageId{0}; imageId < imagesTxt.size(); imageId = imageId + 6)
        imagePaths.push_back(imagesTxt[imageId][0]);

    numberOfImages = imagePaths.size();
    cout << script_label + "Number Of Images = " << numberOfImages << endl;

    if(isBinary){
        vector<vector<cv::Mat>> features;
        loadBinaryFeatures(features,imagePaths);
        testBinaryVocCreation(features);
    }
    else{
        vector<vector<vector<float>>> features;
        loadNonBinaryFeatures(features,imagePaths);
        testNonBinaryVocCreation(features);
    }

    return 0;
}

// ----------------------------------------------------------------------------
void loadBinaryFeatures(vector<vector<cv::Mat>> &features, const std::vector<std::string>& imagePaths)
{
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    features.clear();
    features.reserve(numberOfImages);

    const int progressBarWidth = 50; // Width of the progress bar in characters
    const int totalIterations = numberOfImages; // Total iterations for the process

    cout << script_label + "Extracting " + descriptorName + " features..." << endl;
    switch(descriptorId) { // loadBinaryFeatures
        case DESC_BRISK:{          
            cv::Ptr<cv::BRISK> brisk = cv::BRISK::create();
            for(int iRGB = 0; iRGB < numberOfImages; ++iRGB){
                double progress = (double)iRGB / totalIterations;
                displayProgressBar(progressBarWidth, progress);
                vector<cv::KeyPoint> keypoints;
                cv::Mat descriptors;
                cv::Mat image = cv::imread(imagePaths[iRGB], 0);
                cv::Mat mask{};
                brisk->detectAndCompute(image, mask, keypoints, descriptors);
                features.push_back(vector<cv::Mat>());
                changeStructure(descriptors, features.back());
            }
            break;          
        }      
        case DESC_AKAZE61: {
            // Provide your detector
            cv::Ptr<cv::AKAZE> akaze61 = cv::AKAZE::create();
            akaze61->setNOctaves(4);
            akaze61->setNOctaveLayers(4);
            //akaze61->setMaxPoints(1000);
            for(int i = 0; i < numberOfImages; ++i){
                double progress = (double)i / totalIterations;
                displayProgressBar(progressBarWidth, progress);
            	vector<cv::KeyPoint> keypoints;
            	cv::Mat descriptors;
            	cv::Mat image = cv::imread(imagePaths[i], 0);
            	cv::Mat mask{};
                akaze61->detectAndCompute(image, mask, keypoints, descriptors);
                features.push_back(vector<cv::Mat>());
                changeStructure(descriptors, features.back());
            }
            break;
        }
        case DESC_ORB: {
            cv::Ptr<cv::ORB> orb = cv::ORB::create();
            orb->setNLevels(8);
            orb->setScaleFactor(1.2f);
            orb->setMaxFeatures(1000);
            for(int i = 0; i < numberOfImages; ++i){
                double progress = (double)i / totalIterations;
                displayProgressBar(progressBarWidth, progress);
                vector<cv::KeyPoint> keypoints;
                cv::Mat descriptors;
                cv::Mat image = cv::imread(imagePaths[i], 0);
                cv::Mat mask{};
                orb->detectAndCompute(image, mask, keypoints, descriptors);
                features.push_back(vector<cv::Mat>());
                changeStructure(descriptors, features.back());
            }
            break;
        }
    }
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

    double tduration = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
    cout << "        finished (" + std::to_string(tduration) + " s). " << endl;
}

template <typename T>
void loadNonBinaryFeatures(vector<vector<T>> &features, const std::vector<std::string>& imagePaths)
{
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    features.clear();
    features.reserve(numberOfImages);

    const int progressBarWidth = 50; // Width of the progress bar in characters
    const int totalIterations = numberOfImages; // Total iterations for the process

    cout << script_label + "Extracting " + descriptorName + " features..." << endl;
    switch(descriptorId) { // loadNonBinaryFeatures
        case DESC_KAZE64:{
            cv::Ptr<cv::KAZE> kaze64 = cv::KAZE::create();
            for(int iRGB = 0; iRGB < numberOfImages; ++iRGB){
                double progress = (double)iRGB / totalIterations;
                displayProgressBar(progressBarWidth, progress);
                vector<cv::KeyPoint> keypoints;
                cv::Mat descriptors;
                cv::Mat image = cv::imread(imagePaths[iRGB], 0);
                cv::Mat mask{};
                kaze64->detectAndCompute(image, mask, keypoints, descriptors);
                features.push_back(vector<vector<float>>());
                changeStructure(descriptors, features.back());
            }
            break;
        }
        case DESC_SURF64:{
            cv::Ptr<cv::xfeatures2d::SURF> surf64 = cv::xfeatures2d::SURF::create();
            for(int iRGB = 0; iRGB < numberOfImages; ++iRGB){
                double progress = (double)iRGB / totalIterations;
                displayProgressBar(progressBarWidth, progress);
                vector<cv::KeyPoint> keypoints;
                cv::Mat descriptors;
                cv::Mat image = cv::imread(imagePaths[iRGB], 0);
                cv::Mat mask{};
                surf64->detectAndCompute(image, mask, keypoints, descriptors);
                features.push_back(vector<vector<float>>());
                changeStructure(descriptors, features.back());
            }
            break;   
        }
    }
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

    double tduration = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
    cout << "        finished (" + std::to_string(tduration) + " s). " << endl;
}

// ----------------------------------------------------------------------------
void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out)
{
  out.resize(plain.rows);

  for(int i = 0; i < plain.rows; ++i)
  {
    out[i] = plain.row(i);
  }
}

template <typename T>
void changeStructure(const cv::Mat &plain, vector<T> &out)
{
    out.resize(plain.rows);

    for(int i = 0; i < plain.rows; ++i)
    {
        out[i] = plain.row(i);
    }
}

// ----------------------------------------------------------------------------
void testBinaryVocCreation(const vector<vector<cv::Mat>> &features){
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

    cout << script_label + "Creating a " << k << "^" << L << " vocabulary..." << endl;

    switch (descriptorId) { // create binary vocabulary      
        case DESC_BRISK:{
            BriskVocabulary voc(k, L, weight, scoring);
            voc.create(features);
            cout << script_label + "Vocabulary information: " << endl
                 << voc << endl << endl;
            cout << endl << "Saving vocabulary..." << endl;
            voc.saveToTextFile(savePath + "/Brisk_DBoW2_voc.txt");
            break;
        }
        case DESC_AKAZE61:
        {
            Akaze61Vocabulary voc(k, L, weight, scoring);
            voc.create(features);
            cout << script_label + "Vocabulary information: " << endl
                 << voc << endl << endl;
            cout << endl << "Saving vocabulary..." << endl;
            //voc.save(savePath + "/" + descriptorName + "_DBoW2_voc.yml.gz");
            voc.saveToTextFile(savePath + "/" + descriptorName + "_DBoW2_voc.txt");
            break;
        }
        case DESC_ORB:
        {
            OrbVocabulary voc(k, L, weight, scoring);
            voc.create(features);
            cout << script_label + "Vocabulary information: " << endl
                 << voc << endl << endl;
            cout << endl << "Saving vocabulary..." << endl;
            //voc.save(savePath + "/" + descriptorName + "_DBoW2_voc.yml.gz");
            voc.saveToTextFile(savePath + "/" + descriptorName + "_DBoW2_voc.txt");
            break;
        }
    }

    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    double tduration = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
    cout << "        finished (" + std::to_string(tduration) + " s). " << endl;
}

template <typename T>
void testNonBinaryVocCreation(const vector<vector<T>> &features){
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

    cout << script_label + "Creating a " << k << "^" << L << " "<< descriptorName << " vocabulary..." << endl;

    switch (descriptorId) { // create  non binary vocabulary
        case DESC_KAZE64:{
            Kaze64Vocabulary voc(k, L, weight, scoring);
            voc.create(features);
            cout << script_label + "Vocabulary information: " << endl
                 << voc << endl << endl;
            cout << endl << "Saving vocabulary..." << endl;
            voc.saveToTextFile(savePath + "/Kaze64_DBoW2_voc.txt");
            break;
        }
        case DESC_SURF64:{
            Surf64Vocabulary voc(k, L, weight, scoring);
            voc.create(features);
            cout << script_label + "Vocabulary information: " << endl
                 << voc << endl << endl;
            cout << endl << "Saving vocabulary..." << endl;
            voc.saveToTextFile(savePath + "/Surf64_DBoW2_voc.txt");
            break;
        }
    }

    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    double tduration = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
    cout << "        finished (" + std::to_string(tduration) + " s). " << endl;
}

// ----------------------------------------------------------------------------

std::vector<std::vector<std::string>> read_txt(const std::string &filePath, const size_t &numCols, char delimiter ,int headerRows) {
    std::vector<std::vector<std::string>> txtFile{};
    std::ifstream file(filePath);
    std::string line, word;

    if (file.good()) {
        file.clear();
        file.seekg(0, std::ios::beg);

        // Skip the rows of the header
        for (int jRow{0}; jRow < headerRows; jRow++) {
            getline(file, line);
        }

        // Stpre rows and works in txtFile vectors
        while (getline(file, line)) {
            std::stringstream line_stream(line);
            std::vector <std::string> row{};
            for (int jRow{0}; jRow < numCols; jRow++) {
                getline(line_stream, word, delimiter);
                row.push_back(word);
            }
            txtFile.push_back(row);
        }
    } else {
        // throw std::invalid_argument("File not found : " + filePath);
        cout << "File not found : " + filePath<< endl;
    }
    return txtFile;
}

void displayProgressBar(int width, double progressPercentage) {
    int pos = width * progressPercentage;
    std::cout << "[";
    for (int i = 0; i < width; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(progressPercentage * 100.0) << " %\r";
    std::cout.flush();
}
