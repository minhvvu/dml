/*
 * dataUtils.h
 *
 *  Created on: Jun 4, 2015
 *      Author: vvminh
 */

#ifndef UTILS_DATAUTILS_H_
#define UTILS_DATAUTILS_H_

#include <sstream>
#include <iostream>
#include <fstream>
#include <string>
#include <stdexcept>
#include "Eigen3.h"
#include "functionUtils.h"

using namespace Eigen;

/**
 * read ground truth file into a vector: grouthTruth[imageId] = classId
 * the number of ground truth classes is assigned to in-out param nClasses
 */
std::vector<int> readGroundTruth(const std::string fileName, int& nClasses) {
    std::ifstream infile(fileName.c_str());
    if (!infile.is_open()) {
        throw std::runtime_error("Unable to open grouth truth file: " + fileName);
    }

    // read number of grouth truth class
    infile >> nClasses;

    // build grouth truth label: grouthTruth[imageId] = classId
    std::vector<int> grouthTruth;
    int idx = 0, classId = 0;
    while (infile >> idx >> classId) {
        grouthTruth.push_back(classId);
    }
    infile.close();
    return grouthTruth;
}

inline MatrixXf readMatrix(const std::string fileName) {
    MatrixXf mat;
    std::ifstream infile(fileName.c_str());
    if (!infile.is_open()) {
        std::cerr << "Can not open mat file: " << fileName << std::endl;
        return mat;
    }
    std::string keyName;
	int nExamples = 0;
	int nDimensions = 0;
	float val = 0.0f;
	infile >> keyName >> nDimensions >> keyName >> nExamples;
	mat = MatrixXf(nDimensions, nExamples);

	for (int dim = 0; dim < nDimensions; ++dim) {
		for (int exampleIdx = 0; exampleIdx < nExamples; ++exampleIdx) {
			infile >> val;
			mat(dim, exampleIdx) = val;
		}
	}
    return mat;
}

/*
 * Distribution of value in Wang with rgSIFT, codebook size = 200
 * minValue: 0, maxValue: 11.4308
 * 169126	20860	6199	2282	884	365	160	72	40	8	3	1
 */
inline float* readWangDB(int nData = 1000, int nDims = 200) {
	std::string basePath = "WangDatabase/rgSIFTBagOfWord_200/";
	std::string fileExt = ".jpg.txt";

	float* rawData = new float[nData * nDims];
	std::ifstream infile;
	int nTemp; //not uesd
	float oneValue = 0.0f;

	for (int i = 0; i < nData; ++i) {

		std::string fileName = basePath + intToStringXXX(i) + fileExt;
		infile.open(fileName.c_str());
		if (!infile.is_open()) {
			std::cerr << "Can not open file " << fileName << std::endl;
			continue;
		}
		infile >> nTemp;

		for (int j = 0; j < nDims; ++j) {
			rawData[j + i * nDims] = 0;
			infile >> oneValue;
			rawData[j + i * nDims] = oneValue;
		}

		infile.close();
	}
	return rawData;
}

/**
 * Read all descriptor of WangDB from txt file, put them into an array,
 * convert this array to Matrix and save this Matrix into file
 */
inline void writeWangDBtoMatFile(int nData = 1000, int nDims = 200) {
	float* rawData = readWangDB(nData, nDims);
	MatrixXf data = Map < MatrixXf > (rawData, nDims, nData);

	char fileName[128];
	sprintf(fileName, "WangDatabase/wang-%d-%d.mat", nData, nDims);
	std::ofstream ofs(fileName, std::ios::binary);
	ofs.write((char*) data.data(), data.size() * sizeof(float));
	ofs.close();
	delete rawData;
}

/**
 * Read Wang Descriptor from binary file to a matrix
 */
inline MatrixXf readWangDBFromMatFile(int nData = 1000, int nDims = 200) {
	char fileName[128];
	sprintf(fileName, "WangDatabase/wang-%d-%d.mat", nData, nDims);
	std::ifstream infile(fileName, std::ios::binary);
	if (!infile.is_open()) {
		std::cerr << "Can not open data file: " << fileName;
		exit(-1);
	}
	MatrixXf data(nDims, nData);
	infile.read((char *) data.data(), nDims * nData * sizeof(float));
	infile.close();
	return data;
}

inline MatrixXf getPCA(const Ref<const MatrixXf>& X, int k = 2) {
	// do compare EVD SVD

	//std::cout << "\n\nEVD:\n\n";
	MatrixXf cov = X*X.transpose() * (1.0 / (X.cols() - 1));
	EigenSolver<MatrixXf> ev(cov);
	// float sumEigenValues = ev.eigenvalues().real().sum();
	// float acc0 = 0.0f;
	// for (int i = 0; i < ev.eigenvalues().real().size(); ++i) {
	// 	acc0 += ev.eigenvalues().real()[i];
	// 	std::cout << "Principale component " << i
	// 		<< "\twith value: " << ev.eigenvalues().real()[i]
	// 		<< "\taccumulate: " << acc0
	// 		<< "\texplains " << (acc0 /sumEigenValues)*100 << " % data\n";
	// }

	//std::cout << "\n\nSVD:\n\n";
	JacobiSVD<MatrixXf> svd(X, ComputeThinU);
	// float sumSingularVars = svd.singularValues().array().square().sum() / (X.cols() - 1);
	// float acc = 0.0f;
	// for (int i = 0; i < svd.singularValues().size(); ++i) {
	// 	float sv = (svd.singularValues()[i]) * (svd.singularValues()[i]) / (X.cols() - 1);
	// 	acc += sv;
	// 	std::cout << "Principale component " << i
	// 		<< "\twith value: " << sv
	// 		<< "\taccumulate: " << acc
	// 		<< "\texplains " << (acc /sumSingularVars)*100 << " % data\n";
	// }

	return ev.eigenvectors().real().leftCols(k).transpose() * X;
	// return svd.matrixU().leftCols(k).transpose() * X;
}

inline void writePCAResult(const Ref<const MatrixXf>& Xp, const std::vector<int>& vAssign) {
	std::string fileName = "/home/vvminh/git/dml/plotting/data/points.csv";
	std::ofstream outfile(fileName.c_str());
	if (!outfile.is_open()) {
		std::cerr << "Can not open output file: " << fileName;
		exit(-1);
	}
	outfile << "pos1,pos2,clusterId\n";
	for (int i = 0; i < Xp.cols(); ++i) {
		outfile << Xp(0, i) << "," << Xp(1, i) << "," << vAssign[i] << '\n';
	}
	outfile.close();
}

inline void visualize(const Ref<const MatrixXf>& X, const std::vector<int>& vAssign) {
	MatrixXf Xp = getPCA(X, 2);
	writePCAResult(Xp, vAssign);
}

#endif /* UTILS_DATAUTILS_H_ */
