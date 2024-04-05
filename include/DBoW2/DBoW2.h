/*
 * File: DBoW2.h
 * Date: November 2011
 * Author: Dorian Galvez-Lopez
 * Description: Generic include file for the DBoW2 classes and
 *   the specialized vocabularies and databases
 * License: see the LICENSE.txt file
 *
 */

/*! \mainpage DBoW2 Library
 *
 * DBoW2 library for C++:
 * Bag-of-word image database for image retrieval.
 *
 * Written by Dorian Galvez-Lopez,
 * University of Zaragoza
 * 
 * Check my website to obtain updates: http://doriangalvez.com
 *
 * \section requirements Requirements
 * This library requires the DUtils, DUtilsCV, DVision and OpenCV libraries,
 * as well as the boost::dynamic_bitset class.
 *
 * \section citation Citation
 * If you use this software in academic works, please cite:
 <pre>
   @@ARTICLE{GalvezTRO12,
    author={Galvez-Lopez, Dorian and Tardos, J. D.}, 
    journal={IEEE Transactions on Robotics},
    title={Bags of Binary Words for Fast Place Recognition in Image Sequences},
    year={2012},
    month={October},
    volume={28},
    number={5},
    pages={1188--1197},
    doi={10.1109/TRO.2012.2197158},
    ISSN={1552-3098}
  }
 </pre>
 *
 */

#ifndef __D_T_DBOW2__
#define __D_T_DBOW2__

/// Includes all the data structures to manage vocabularies and image databases
namespace DBoW2
{
}

#include "TemplatedVocabulary.h"
#include "TemplatedDatabase.h"
#include "BowVector.h"
#include "FeatureVector.h"
#include "QueryResults.h"
#include "FBrief.h"

#include "FOrb.h"
#include "FR2d2.h"
#include "FSift128.h"
#include "FKaze64.h"
#include "FSurf64.h"
#include "FBrisk.h"
#include "FAkaze61.h"

// Vocabulary Templates

/// R2d2 Vocabulary
typedef DBoW2::TemplatedVocabulary<DBoW2::FR2d2::TDescriptor, DBoW2::FR2d2> R2d2Vocabulary;

/// FR2d2 Database
typedef DBoW2::TemplatedDatabase<DBoW2::FR2d2::TDescriptor, DBoW2::FR2d2> R2d2Database;

/// Sift128 Vocabulary
typedef DBoW2::TemplatedVocabulary<DBoW2::FSift128::TDescriptor, DBoW2::FSift128> Sift128Vocabulary;

/// FSift128 Database
typedef DBoW2::TemplatedDatabase<DBoW2::FSift128::TDescriptor, DBoW2::FSift128> Sift128Database;

/// Kaze64 Vocabulary
typedef DBoW2::TemplatedVocabulary<DBoW2::FKaze64::TDescriptor, DBoW2::FKaze64> Kaze64Vocabulary;

/// FKaze64 Database
typedef DBoW2::TemplatedDatabase<DBoW2::FKaze64::TDescriptor, DBoW2::FKaze64> Kaze64Database;

/// Surf64 Vocabulary
typedef DBoW2::TemplatedVocabulary<DBoW2::FSurf64::TDescriptor, DBoW2::FSurf64> Surf64Vocabulary;

/// FSurf64 Database
typedef DBoW2::TemplatedDatabase<DBoW2::FSurf64::TDescriptor, DBoW2::FSurf64> Surf64Database;

/// Brisk Vocabulary
typedef DBoW2::TemplatedVocabulary<DBoW2::FBrisk::TDescriptor, DBoW2::FBrisk> BriskVocabulary;

/// FBrisk Database
typedef DBoW2::TemplatedDatabase<DBoW2::FBrisk::TDescriptor, DBoW2::FBrisk> BriskDatabase;

/// Orb Vocabulary
typedef DBoW2::TemplatedVocabulary<DBoW2::FOrb::TDescriptor, DBoW2::FOrb> OrbVocabulary;

/// FOrb Database
typedef DBoW2::TemplatedDatabase<DBoW2::FOrb::TDescriptor, DBoW2::FOrb> OrbDatabase;

/// Akaze61 Vocabulary
typedef DBoW2::TemplatedVocabulary<DBoW2::FAkaze61::TDescriptor, DBoW2::FAkaze61> Akaze61Vocabulary;

/// FAkaze61 Database
typedef DBoW2::TemplatedDatabase<DBoW2::FAkaze61::TDescriptor, DBoW2::FAkaze61> Akaze61Database;
  
/// BRIEF Vocabulary
typedef DBoW2::TemplatedVocabulary<DBoW2::FBrief::TDescriptor, DBoW2::FBrief> BriefVocabulary;

/// BRIEF Database
typedef DBoW2::TemplatedDatabase<DBoW2::FBrief::TDescriptor, DBoW2::FBrief> BriefDatabase;

#endif

