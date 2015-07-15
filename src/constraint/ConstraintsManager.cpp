/*
 * ConstraintsManager.cpp
 *
 *  Created on: Jun 12, 2015
 *      Author: vvminh
 */

#include "ConstraintsManager.h"

#include <fstream>
#include <sstream>
#include <string>
#include <iterator>
#include <iostream>
#include <utility>
#include <cassert>
#include "../utils/functionUtils.h"
#include "InitManager.h"

namespace dml {

ConstraintsManager::ConstraintsManager(std::string inputFileName)
	:fileName(inputFileName) {}

ConstraintsManager::~ConstraintsManager() {}

void ConstraintsManager::readConnectedComponents() {
	std::string fileSCC = fileName + ".scc";
	std::ifstream infile(fileSCC.c_str());
	assert(infile.is_open() && "Can not open SCC file");

	std::string line;
	while(std::getline(infile, line)) {
		std::istringstream iss(line);
		std::vector<int> tokens{
			std::istream_iterator<int>{iss},
            std::istream_iterator<int>{}};
        scc.push_back(tokens);
	}
	infile.close();
}

void ConstraintsManager::readConstraintsFromFile() {
	std::string fileLinks = fileName + ".links";
	std::ifstream infile(fileLinks.c_str());
	assert(infile.is_open() && "Can not open Constraints file");

	int nodeA = 0, nodeB = 0, nodeType = 0;
	infile >> nConstraintsOriginal >> nConstraintsDeduced;

	for (int i = 0; i < nConstraintsDeduced; ++i) {
		infile >> nodeA >> nodeB >> nodeType;

		if (MUST_LINK == nodeType) {
			ML[nodeA].push_back(nodeB);
			ML[nodeB].push_back(nodeA);
		} else if (CANNOT_LINK == nodeType) {
			CL[nodeA].push_back(nodeB);
			CL[nodeB].push_back(nodeA);
		}
	}
	infile.close();
	refineConstraints();
}

void ConstraintsManager::refineConstraints() {
	numML = refineAndCount(ML);
	numCL = refineAndCount(CL);
}

int ConstraintsManager::refineAndCount(ConstraintMap& constraintsMap) {
	int nCount = 0;
	for (auto& it : constraintsMap) {
		int key = it.first;
		it.second.sort();
		it.second.unique();
		it.second.remove(key);
		nCount += it.second.size();
	}
	return nCount;
}

void ConstraintsManager::dumpConstraints() {
	std::cout << "MUST LINKS: " << std::endl;
	dumpConstraints (ML);

	std::cout << "CANNOT LINKS: " << std::endl;
	dumpConstraints (CL);
}

void ConstraintsManager::dumpConstraints(const ConstraintMap& constraintsMap) {
	for (const auto& it : constraintsMap) {
		std::cout << it.first << " => ( ";
		for (const auto& it2 : it.second) {
			std::cout << it2 << ", ";
		}
		std::cout << ")\n";
	}
}

using namespace Eigen;

MatrixXf ConstraintsManager::genInitCentersFromML(const Ref<const MatrixXf>& X, int nClusters) {
	int nDims = X.rows();
	int nComps = scc.size();

	MatrixXf initCenters(nDims, nClusters);
	InitManager initMgnr(nDims, nClusters);

	if (0 == nComps) {
		initMgnr.fillWithTotalRandomInit(X, initCenters);
	} else { 
		MatrixXf compCentroids = getComponentCenters(X);
		if (nComps <= nClusters) {
			for (int compId = 0; compId < nComps; ++compId) {
				initCenters.col(compId) = compCentroids.col(compId);
			}
			if (nComps < nClusters) {
				VectorXf globalMean = X.rowwise().mean();
				for (int compId = nComps; compId < nClusters; ++compId) {
					initCenters.col(compId) = globalMean + VectorXf::Random(nDims);
				}
			}
		} else {
			std::vector<float> weights = getComponentWeights();
			initMgnr.fillWithFarthestFirstTraversal(compCentroids, weights, initCenters);
		}
	}
	return initCenters;
}

MatrixXf ConstraintsManager::getComponentCenters(const Ref<const MatrixXf>& X) {
	int nDims = X.rows();
	int nComps = (int)scc.size();
	MatrixXf compCentroids = MatrixXf::Zero(nDims, nComps);

	for (int compId = 0; compId < nComps; ++compId) {
		int oneCompSize = (int)scc.at(compId).size();
		if (0 == oneCompSize) continue;

		MatrixXf oneCompData(nDims, oneCompSize);
		for (int idx = 0; idx < oneCompSize; ++idx) {
			int dataIndex = scc.at(compId).at(idx);
			oneCompData.col(idx) = X.col(dataIndex);
		}
		compCentroids.col(compId) = oneCompData.rowwise().mean();
	}
	return compCentroids;
}

std::vector<float> ConstraintsManager::getComponentWeights() {
	float sumWeight = 0.0f;
	std::vector<float> weights;
	for (const auto& v : scc) {
		weights.push_back(1.0 * v.size());
		sumWeight += 1.0 * v.size();
	}
	for (auto& weight : weights) {
		weight /= sumWeight;
	}
	return weights;
}

} /* namespace dml */
