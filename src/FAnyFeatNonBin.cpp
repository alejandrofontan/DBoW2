 /**
 * File: FAnyFeatNonBin.cpp
 * Date: March 2024
 * Original Author: Dorian Galvez-Lopez
 * Modified by Alejandro Fontan Villacampa for AnyFeature-VSLAM
 * Description: functions for AnyFeatureNonBin descriptors
 * License: see the LICENSE.txt file
 *
 */
 
#include <vector>
#include <string>
#include <sstream>

#include "FClass.h"
#include "FAnyFeatNonBin.h"

using namespace std;

namespace DBoW2 {

    // --------------------------------------------------------------------------
    void FAnyFeatNonBin::meanValue(const std::vector<FAnyFeatNonBin::pDescriptor> &descriptors, FAnyFeatNonBin::TDescriptor &mean)
    {
        mean.resize(0);
        mean.resize(FAnyFeatNonBin::L, 0);

        float s = descriptors.size();

        vector<FAnyFeatNonBin::pDescriptor>::const_iterator it;
        for(it = descriptors.begin(); it != descriptors.end(); ++it)
        {
            const FAnyFeatNonBin::TDescriptor &desc = **it;
            for(int i = 0; i < FAnyFeatNonBin::L; i ++)
                mean[i] += desc[i] / s;
        }
    }

    // --------------------------------------------------------------------------
    double FAnyFeatNonBin::distance(const FAnyFeatNonBin::TDescriptor &a, const FAnyFeatNonBin::TDescriptor &b)
    {
        double sqd{0.0};
        for(int i{0}; i < FAnyFeatNonBin::L; i++)
            sqd += (a[i] - b[i])*(a[i] - b[i]);
        return sqd;
    }

    // --------------------------------------------------------------------------
    std::string FAnyFeatNonBin::toString(const FAnyFeatNonBin::TDescriptor &a)
    {
        stringstream ss;
        for(int i = 0; i < FAnyFeatNonBin::L; ++i)
        {
            ss << a[i] << " ";
        }
        return ss.str();
    }

    // --------------------------------------------------------------------------

    void FAnyFeatNonBin::fromString(FAnyFeatNonBin::TDescriptor &a, const std::string &s)
    {
        a.resize(FAnyFeatNonBin::L);
        stringstream ss(s);
        for(int i = 0; i < FAnyFeatNonBin::L; ++i)
            ss >> a[i];
    }

    // --------------------------------------------------------------------------
    void FAnyFeatNonBin::toMat32F(const std::vector<TDescriptor> &descriptors,cv::Mat &mat)
    {
        if(descriptors.empty())
        {
            mat.release();
            return;
        }

        const int N = descriptors.size();
        const int L = FAnyFeatNonBin::L;

        mat.create(N, L, CV_32F);

        for(int i = 0; i < N; ++i)
        {
            const TDescriptor& desc = descriptors[i];
            float *p = mat.ptr<float>(i);
            for(int j = 0; j < L; ++j, ++p)
            {
                *p = desc[j];
            }
        }
    }

    // --------------------------------------------------------------------------

} // namespace DBoW2

