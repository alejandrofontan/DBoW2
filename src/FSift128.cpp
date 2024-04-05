 /**
 * File: FSift128.cpp
 * Date: March 2024
 * Original Author: Dorian Galvez-Lopez
 * Modified by Alejandro Fontan Villacampa for AnyFeature-VSLAM
 * Description: functions for Sift128 descriptors
 * License: see the LICENSE.txt file
 *
 */
 
#include <vector>
#include <string>
#include <sstream>

#include "FClass.h"
#include "FSift128.h"

using namespace std;

namespace DBoW2 {

// --------------------------------------------------------------------------

void FSift128::meanValue(const std::vector<FSift128::pDescriptor> &descriptors, 
  FSift128::TDescriptor &mean)
{
  mean.resize(0);
  mean.resize(FSift128::L, 0);
  
  float s = descriptors.size();
  
  vector<FSift128::pDescriptor>::const_iterator it;
  for(it = descriptors.begin(); it != descriptors.end(); ++it)
  {
    const FSift128::TDescriptor &desc = **it;
    for(int i = 0; i < FSift128::L; i += 4)
    {
      mean[i  ] += desc[i  ] / s;
      mean[i+1] += desc[i+1] / s;
      mean[i+2] += desc[i+2] / s;
      mean[i+3] += desc[i+3] / s;
    }
  }
}

// --------------------------------------------------------------------------
  
double FSift128::distance(const FSift128::TDescriptor &a, const FSift128::TDescriptor &b)
{
  double sqd = 0.;
  for(int i = 0; i < FSift128::L; i += 4)
  {
    sqd += (a[i  ] - b[i  ])*(a[i  ] - b[i  ]);
    sqd += (a[i+1] - b[i+1])*(a[i+1] - b[i+1]);
    sqd += (a[i+2] - b[i+2])*(a[i+2] - b[i+2]);
    sqd += (a[i+3] - b[i+3])*(a[i+3] - b[i+3]);
  }
  return sqd;
}

// --------------------------------------------------------------------------

std::string FSift128::toString(const FSift128::TDescriptor &a)
{
  stringstream ss;
  for(int i = 0; i < FSift128::L; ++i)
  {
    ss << a[i] << " ";
  }
  return ss.str();
}

// --------------------------------------------------------------------------
  
void FSift128::fromString(FSift128::TDescriptor &a, const std::string &s)
{
  a.resize(FSift128::L);
  
  stringstream ss(s);
  for(int i = 0; i < FSift128::L; ++i)
  {
    ss >> a[i];
  }
}

// --------------------------------------------------------------------------

void FSift128::toMat32F(const std::vector<TDescriptor> &descriptors, 
    cv::Mat &mat)
{
  if(descriptors.empty())
  {
    mat.release();
    return;
  }
  
  const int N = descriptors.size();
  const int L = FSift128::L;
  
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

