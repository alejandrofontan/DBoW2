 /**
 * File: FKaze64.cpp
 * Date: March 2024
 * Original Author: Dorian Galvez-Lopez
 * Modified by Alejandro Fontan Villacampa for AnyFeature-VSLAM
 * Description: functions for Kaze64 descriptors
 * License: see the LICENSE.txt file
 *
 */
 
#include <vector>
#include <string>
#include <sstream>

#include "FClass.h"
#include "FKaze64.h"

using namespace std;

namespace DBoW2 {

// --------------------------------------------------------------------------

void FKaze64::meanValue(const std::vector<FKaze64::pDescriptor> &descriptors, 
  FKaze64::TDescriptor &mean)
{
  mean.resize(0);
  mean.resize(FKaze64::L, 0);
  
  float s = descriptors.size();
  
  vector<FKaze64::pDescriptor>::const_iterator it;
  for(it = descriptors.begin(); it != descriptors.end(); ++it)
  {
    const FKaze64::TDescriptor &desc = **it;
    for(int i = 0; i < FKaze64::L; i += 4)
    {
      mean[i  ] += desc[i  ] / s;
      mean[i+1] += desc[i+1] / s;
      mean[i+2] += desc[i+2] / s;
      mean[i+3] += desc[i+3] / s;
    }
  }
}

// --------------------------------------------------------------------------
  
double FKaze64::distance(const FKaze64::TDescriptor &a, const FKaze64::TDescriptor &b)
{
  double sqd = 0.;
  for(int i = 0; i < FKaze64::L; i += 4)
  {
    sqd += (a[i  ] - b[i  ])*(a[i  ] - b[i  ]);
    sqd += (a[i+1] - b[i+1])*(a[i+1] - b[i+1]);
    sqd += (a[i+2] - b[i+2])*(a[i+2] - b[i+2]);
    sqd += (a[i+3] - b[i+3])*(a[i+3] - b[i+3]);
  }
  return sqd;
}

// --------------------------------------------------------------------------

std::string FKaze64::toString(const FKaze64::TDescriptor &a)
{
  stringstream ss;
  for(int i = 0; i < FKaze64::L; ++i)
  {
    ss << a[i] << " ";
  }
  return ss.str();
}

// --------------------------------------------------------------------------
  
void FKaze64::fromString(FKaze64::TDescriptor &a, const std::string &s)
{
  a.resize(FKaze64::L);
  
  stringstream ss(s);
  for(int i = 0; i < FKaze64::L; ++i)
  {
    ss >> a[i];
  }
}

// --------------------------------------------------------------------------

void FKaze64::toMat32F(const std::vector<TDescriptor> &descriptors, 
    cv::Mat &mat)
{
  if(descriptors.empty())
  {
    mat.release();
    return;
  }
  
  const int N = descriptors.size();
  const int L = FKaze64::L;
  
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

