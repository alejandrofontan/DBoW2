 /**
 * File: FR2d2.cpp
 * Date: March 2024
 * Original Author: Dorian Galvez-Lopez
 * Modified by Alejandro Fontan Villacampa for AnyFeature-VSLAM
 * Description: functions for R2d2 descriptors
 * License: see the LICENSE.txt file
 *
 */
 
#include <vector>
#include <string>
#include <sstream>

#include "FClass.h"
#include "FR2d2.h"

using namespace std;

namespace DBoW2 {

// --------------------------------------------------------------------------

void FR2d2::meanValue(const std::vector<FR2d2::pDescriptor> &descriptors, 
  FR2d2::TDescriptor &mean)
{
  mean.resize(0);
  mean.resize(FR2d2::L, 0);
  
  float s = descriptors.size();
  
  vector<FR2d2::pDescriptor>::const_iterator it;
  for(it = descriptors.begin(); it != descriptors.end(); ++it)
  {
    const FR2d2::TDescriptor &desc = **it;
    for(int i = 0; i < FR2d2::L; i += 4)
    {
      mean[i  ] += desc[i  ] / s;
      mean[i+1] += desc[i+1] / s;
      mean[i+2] += desc[i+2] / s;
      mean[i+3] += desc[i+3] / s;
    }
  }
}

// --------------------------------------------------------------------------
  
double FR2d2::distance(const FR2d2::TDescriptor &a, const FR2d2::TDescriptor &b)
{
  double sqd = 0.;
  for(int i = 0; i < FR2d2::L; i += 4)
  {
    sqd += (a[i  ] - b[i  ])*(a[i  ] - b[i  ]);
    sqd += (a[i+1] - b[i+1])*(a[i+1] - b[i+1]);
    sqd += (a[i+2] - b[i+2])*(a[i+2] - b[i+2]);
    sqd += (a[i+3] - b[i+3])*(a[i+3] - b[i+3]);
  }
  return sqd;
}

// --------------------------------------------------------------------------

std::string FR2d2::toString(const FR2d2::TDescriptor &a)
{
  stringstream ss;
  for(int i = 0; i < FR2d2::L; ++i)
  {
    ss << a[i] << " ";
  }
  return ss.str();
}

// --------------------------------------------------------------------------
  
void FR2d2::fromString(FR2d2::TDescriptor &a, const std::string &s)
{
  a.resize(FR2d2::L);
  
  stringstream ss(s);
  for(int i = 0; i < FR2d2::L; ++i)
  {
    ss >> a[i];
  }
}

// --------------------------------------------------------------------------

void FR2d2::toMat32F(const std::vector<TDescriptor> &descriptors, 
    cv::Mat &mat)
{
  if(descriptors.empty())
  {
    mat.release();
    return;
  }
  
  const int N = descriptors.size();
  const int L = FR2d2::L;
  
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

