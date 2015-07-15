/*
 * functionUtils.h
 *
 *  Created on: Jun 4, 2015
 *      Author: vvminh
 */

#ifndef UTILS_FUNCTIONUTILS_H_
#define UTILS_FUNCTIONUTILS_H_

#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <algorithm> // std::for_each, std::shuffle
#include <random>
#include <iterator>
#include <cassert>
#include <ctime>
#include <cstdlib>
#include <string>

#include "Eigen3.h"
using namespace Eigen;

// for split string using regex, see:
// http://garajeando.blogspot.fr/2014/03/using-c11-to-split-strings-without.html

inline std::string truncateExtension(std::string str) {
	// http://stackoverflow.com/questions/14265581/parse-split-a-string-in-c-using-string-delimiter-standard-c
	std::string delimiter = ".";
	return str.substr(0, str.find(delimiter));
}

template<typename Derived>
inline bool is_finite(const Eigen::MatrixBase<Derived>& x)
{
   return ( (x - x).array() == (x - x).array()).all();
}


template<typename Derived>
inline bool is_nan(const Eigen::MatrixBase<Derived>& x)
{
   return ((x.array() == x.array())).all();
}

inline void dump(const std::string tag, const std::vector<int>& v) {
	std::cout << "TAG: " << tag << std::endl;
	std::for_each(v.begin(), v.end(), [](const int i) {
		std::cout << i << " ";
	});
	std::cout << std::endl;
}

inline void initRandomSeed() {
	std::srand(unsigned(std::time(0)));
}

/**
 * shuffle a vector of int using std default random engine
 * @param v vector of int
 */
inline void shuffleVector(std::vector<int>& v) {
//	auto engine = std::default_random_engine { };
//	std::shuffle(std::begin(v), std::end(v), engine);
	std::random_shuffle(std::begin(v), std::end(v));

}

/**
 * @return string format of input int, eg (int) 10 => (string) 010
 * example: http://stackoverflow.com/questions/2815746/formatting-an-integer-in-c
 */
inline std::string intToStringXXX(const int i) {
	std::ostringstream oss;
	oss << std::setfill('0') << std::setw(3) << i;
	return (oss.str());
}

/**
 * Create an array of nCount elements randomly generated from minRange to maxRange
 */
inline int* randomIndex(const int minRange = 0,
	const int maxRange = 999, const int nCount = 10) {
	//std::default_random_engine generator;
	//std::uniform_int_distribution<int> distribution(minRange, maxRange);

	std::srand(std::time(0));
	int* indices = new int[nCount];
	for (int j = 0; j < nCount; ++j) {
		indices[j] = minRange + std::rand() % maxRange;
	}
	return indices;
}

inline int randomInRange(const int minRange = 0, const int maxRange = 100) {
	return minRange + std::rand() % maxRange;
}

/**
 * VMeasure: http://www1.cs.columbia.edu/~amaxwell/pubs/v_measure-emnlp07.pdf
 * @param vAssigment the assigment of each data point to its cluster
 * @param nClasses the number of growth truth class
 * @param nClusters the number of cluster generated
 * @beta weight of homogeneity and completeness, default = 0.5
 */
inline float VMeasure(const std::vector<int>& vAssigment,
	const int nClasses, const int nClusters, const float beta = 0.5) {
	float N = (int) vAssigment.size() * 1.0;
	MatrixXf confusionMat(nClasses, nClusters);
	confusionMat.setZero();
	for (int i = 0; i < N; ++i) {
		++confusionMat((i / 100), vAssigment[i]);
	}

	MatrixXf logElemt = confusionMat.array().log();
	VectorXf sumForEachClass = confusionMat.rowwise().sum();
	VectorXf logOfSumForEachClass = sumForEachClass.array().log();
	VectorXf sumForEachCluster = confusionMat.colwise().sum();
	VectorXf logOfSumForEachCluster = sumForEachCluster.array().log();

	VectorXf HCK(nClasses);
	for (int c = 0; c < nClasses; ++c) {
		VectorXf oneClass = confusionMat.row(c) / N;
		VectorXf logOfOneClass = logElemt.row(c);

		HCK[c] = 0.0f;
		for (int k = 0; k < nClusters; ++k) {
			if (oneClass[k] != 0) {
				HCK[c] += oneClass[k] * (logOfOneClass[k] - logOfSumForEachCluster(k));
			}
		}
	}
	VectorXf t1 = sumForEachClass / N;
	VectorXf t2 = logOfSumForEachClass.array() - std::log(N);
	float HC = -t1.cwiseProduct(t2).sum();
	float homogeneity = 1.0 - (-HCK.sum()) / HC;

	VectorXf HKC(nClusters);
	for (int k = 0; k < nClusters; ++k) {
		VectorXf oneCluster = confusionMat.col(k) / N;
		VectorXf logOfOneCluster = logElemt.col(k);

		HKC[k] = 0.0f;
		for (int c = 0; c < nClasses; ++c) {
			if (oneCluster[c] != 0) {
				HKC[k] += oneCluster[c] * (logOfOneCluster[c] - logOfSumForEachClass[c]);
			}
		}
	}
	t1 = sumForEachCluster / N;
	t2 = logOfSumForEachCluster.array() - std::log(N);
	float HK = -t1.cwiseProduct(t2).sum();
	float completeness = 1.0 - (-HKC.sum()) / HK;

	return ((1.0 + beta) * homogeneity * completeness /
	        (beta * homogeneity + completeness));
}

/**
 * calculate VMeasure from vector of assigment and vector of grouth truth class
 * build confusion matrix: row is classId, column is assigned clusterId
 * mat[classId][clusterId] = number of image of classId but assigned to clusterId
 */
inline float VMeasure(const std::vector<int>& vAssigment, const std::vector<int>& vGrouthTruth,
	const int nClasses, const int nClusters, const float beta = 0.5) {
    assert((vAssigment.size() == vGrouthTruth.size()) && "Invalid vector size");

	float N = (int) vAssigment.size() * 1.0;
	MatrixXf confusionMat(nClasses, nClusters);
	confusionMat.setZero();
	for (int i = 0; i < N; ++i) {
        int assignedCluster = vAssigment.at(i);
        int grouthTruthClass = vGrouthTruth.at(i);
        confusionMat(grouthTruthClass, assignedCluster) ++;
	}

	MatrixXf logElemt = confusionMat.array().log();
	VectorXf sumForEachClass = confusionMat.rowwise().sum();
	VectorXf logOfSumForEachClass = sumForEachClass.array().log();
	VectorXf sumForEachCluster = confusionMat.colwise().sum();
	VectorXf logOfSumForEachCluster = sumForEachCluster.array().log();

	VectorXf HCK(nClasses);
	for (int c = 0; c < nClasses; ++c) {
		VectorXf oneClass = confusionMat.row(c) / N;
		VectorXf logOfOneClass = logElemt.row(c);

		HCK[c] = 0.0f;
		for (int k = 0; k < nClusters; ++k) {
			if (oneClass[k] != 0) {
				HCK[c] += oneClass[k] * (logOfOneClass[k] - logOfSumForEachCluster(k));
			}
		}
	}
	VectorXf t1 = sumForEachClass / N;
	VectorXf t2 = logOfSumForEachClass.array() - std::log(N);
	float HC = -t1.cwiseProduct(t2).sum();
	float homogeneity = 1.0 - (-HCK.sum()) / HC;

	VectorXf HKC(nClusters);
	for (int k = 0; k < nClusters; ++k) {
		VectorXf oneCluster = confusionMat.col(k) / N;
		VectorXf logOfOneCluster = logElemt.col(k);

		HKC[k] = 0.0f;
		for (int c = 0; c < nClasses; ++c) {
			if (oneCluster[c] != 0) {
				HKC[k] += oneCluster[c] * (logOfOneCluster[c] - logOfSumForEachClass[c]);
			}
		}
	}
	t1 = sumForEachCluster / N;
	t2 = logOfSumForEachCluster.array() - std::log(N);
	float HK = -t1.cwiseProduct(t2).sum();
	float completeness = 1.0 - (-HKC.sum()) / HK;

	return ((1.0 + beta) * homogeneity * completeness /
	        (beta * homogeneity + completeness));
}

// ham vmeasure cua chi Phuong
inline double vmeasure(const std::vector<int>& vAssignment) {
	const int predictedClass = 10;
	const int trueClass = 10;
	const int nbrImage = 1000;

	int matrix[predictedClass][trueClass] = { 0 };
	for (int i = 0; i < nbrImage; ++i) {
		int c = (int) (i / 100);
		int k = vAssignment[i];
		matrix[c][k]++;
	}

	/************************* compute homogeneity **************************************/
	double h = 1;
	// compute H(C|K)
	double h_ck = 0;
	for (int i = 0; i < predictedClass; i++)
	        {
		double Ni = 0;
		for (int j = 0; j < trueClass; j++)
		        {
			Ni += matrix[i][j];
		}
		if (Ni > 0)
		        {
			for (int j = 0; j < trueClass; j++)
			        {
				if (matrix[i][j] > 0)
				        {
					h_ck -= ((double) matrix[i][j] / (double) nbrImage) * log10((double) matrix[i][j] / (double) Ni);
				}
			}
		}
	}

	if (h_ck != 0)
	        {
		// compute H(C)
		double h_c = 0;
		for (int j = 0; j < trueClass; j++)
		        {
			double Nj = 0;
			for (int i = 0; i < predictedClass; i++)
			        {
				Nj += matrix[i][j];
			}
			if (Nj > 0)
			        {
				h_c -= (Nj / (double) nbrImage) * log10(Nj / (double) nbrImage);
			}
		}
		if (h_c > 0)
		        {
			h = 1 - h_ck / h_c;
		}
	}

	/************************* compute completeness *************************************/
	double c = 1;
	// compute H(K|C)
	double h_kc = 0;
	for (int j = 0; j < trueClass; j++)
	        {
		double Nj = 0;
		for (int i = 0; i < predictedClass; i++)
		        {
			Nj += matrix[i][j];
		}
		if (Nj > 0)
		        {
			for (int i = 0; i < predictedClass; i++)
			        {
				if (matrix[i][j] > 0)
				        {
					h_kc -= (matrix[i][j] / (double) nbrImage) * log10(matrix[i][j] / Nj);
				}
			}
		}
	}

	if (h_kc != 0)
	        {
		// compute H(K)
		double h_k = 0;
		for (int i = 0; i < predictedClass; i++)
		        {
			double Ni = 0;
			for (int j = 0; j < trueClass; j++)
			        {
				Ni += matrix[i][j];
			}
			if (Ni > 0)
			        {
				h_k -= (Ni / (double) nbrImage) * log10(Ni / (double) nbrImage);
			}
		}
		if (h_k > 0)
		        {
			c = 1 - h_kc / h_k;
		}
	}

	/************************* compute V-mesure *****************************************/
	double beta = 0.5;
	double result = (1 + beta) * h * c / (beta * h + c);

//	beta = 0.25;
//	result = (1 + beta) * h * c / (beta * h + c);
//	std::cout << "beta " << beta << ", vmeasure = " << result << std::endl;
//
//	beta = 0.75;
//	result = (1 + beta) * h * c / (beta * h + c);
//	std::cout << << "beta " << beta << ", vmeasure = " << result << std::endl;

	return result;

}

#endif /* UTILS_FUNCTIONUTILS_H_ */
