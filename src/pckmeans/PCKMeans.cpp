/*
 * PCKMeans.cpp
 *
 *  Created on: Jun 22, 2015
 *      Author: vvminh
 */

#include "PCKMeans.h"
#include "../utils/functionUtils.h"

#include <iostream>
#include <limits>

namespace dml {

using namespace Eigen;

PCKMeans::PCKMeans(const ConstMatrixRef& dataset, const int numClts,
	const std::string constraintFileName, const CovType type)
	:EMKMeans(dataset, numClts, type) {	
	constr = ConstraintPtr(new ConstraintsManager(constraintFileName));
	constr->readConstraintsFromFile();
	constr->readConnectedComponents();
	// constr->dumpConstraints();
	// std::cout << "Using : " << constr->numML << " mustlinks deduced (coeff " << constr->mlConst << ")\n"
    // << "Using : " << constr->numCL << " cannotlinks deduced (coeff " << constr->clConst << ")\n";
}

PCKMeans::~PCKMeans() {}

/*virtual*/ void PCKMeans::createInitCenters() {
	MatrixXf initCenters = constr->genInitCentersFromML(data, nClusters);
	for (int cltId = 0; cltId < nClusters; ++cltId) {
		vMixture.at(cltId)->setInitCenter(initCenters.col(cltId));
	}
}

/*virtual*/ void PCKMeans::doVeryFirstClustering() {
	for (int cltId = 0; cltId < nClusters; ++cltId) {
		vMixture.at(cltId)->cacheDistPoint2Point(data);
		vMixture.at(cltId)->cacheDistPoint2Mean(data);
	}
}

/*virtual*/ void PCKMeans::findBestCluster(const std::vector<int>& randomIndex) {
	for (int i = 0; i < nData; ++i) {
		int idx = randomIndex[i];

		float minCost = std::numeric_limits<float>::max();
		int minIndex = -1;

		for (int cltId = 0; cltId < nClusters; ++cltId) {
			float cost = getVariance(idx, cltId)
			        + constr->mlConst * getMustLinksPenalty(idx, cltId)
			        + constr->clConst * getCannotLinksPenalty(idx, cltId);

			if (cost < minCost) {
				minCost = cost;
				minIndex = cltId;
			}
		}
		vAssign[idx] = minIndex;
	}
}

/*virtual*/ float PCKMeans::distanceByCluster(int idx1, int idx2, int cltId) {
	return vMixture.at(cltId)->distance(idx1, idx2);
}

/*virtual*/ float PCKMeans::distanceToMeanOfCluster(int idx, int cltId) {
	return vMixture.at(cltId)->distanceToMean(idx);
}

/*virtual*/ float PCKMeans::maxDistanceByCluster(int cltId) {
	return vMixture.at(cltId)->getMaxDistance();
}

/*virtual*/ float PCKMeans::logDetByCluster(int cltId) {
	return vMixture.at(cltId)->getLogDet();
}

float PCKMeans::getVariance(const int idx, const int cltId) {
	return (
		distanceToMeanOfCluster(idx, cltId) -
		logDetByCluster(cltId)
	);
}

float PCKMeans::getMustLinksPenalty(const int idx1, const int cltId1) {
	float penalty = 0.0f;
	if (constr->ML.count(idx1) > 0) {
		for (const auto& idx2 : constr->ML.at(idx1)) {
			int cltId2 = vAssign[idx2];
			if (cltId1 != cltId2) {
				penalty += 0.5 * (
					distanceByCluster(idx1, idx2, cltId1) +
					distanceByCluster(idx1, idx2, cltId2)
				);
			}
		}
	}
	return penalty;
}

float PCKMeans::getCannotLinksPenalty(const int idx1, const int cltId1) {
	float penalty = 0.0f;
	if (constr->CL.count(idx1) > 0) {
		float maxDistanceOfThisCluster = maxDistanceByCluster(cltId1);
		for (const auto& idx2 : constr->CL.at(idx1)) {
			int cltId2 = vAssign[idx2];
			if (cltId1 == cltId2) {
				penalty += (
					maxDistanceOfThisCluster -
					distanceByCluster(idx1, idx2, cltId1)
				);
			}
		}
	}
	return penalty;
}

/*virtual*/ void PCKMeans::updateMixtures() {
	for (int cltId = 0; cltId < nClusters; ++cltId) {
		vMixture.at(cltId)->updateMean();
		vMixture.at(cltId)->cacheDistPoint2Point(data);
		vMixture.at(cltId)->cacheDistPoint2Mean(data);
	}
}

/*virtual*/ float PCKMeans::calculateObjFunc() {
	float totalVar = totalVariance();
	float totalMLPen = constr->mlConst * totalMLPenalty();
	float totalCLPen = constr->clConst * totalCLPenalty();
	// std::cout << "PCKMeans::calculateObjFunc:\nVariances = " << totalVar
	// 	<< "\tML Penalty = " << totalMLPen
	// 	<< "\tCL Penalty = " << totalCLPen << '\n';
	return totalVar + totalMLPen + totalCLPen;
}

float PCKMeans::totalVariance() {
	float var = 0.0f;
	for (int idx = 0; idx < nData; ++idx) {
		int cltId = vAssign.at(idx);
		var += (
			distanceToMeanOfCluster(idx, cltId) -
			logDetByCluster(cltId)
		);
	}
	return var;
}

float PCKMeans::totalMLPenalty() {
	float penalty = 0.0f;
	countMLViolation = 0;

	for (const auto& it : constr->ML) {
		int idx1 = it.first;
		int cltId1 = vAssign[idx1];

		for (const int& idx2 : it.second) {
			int cltId2 = vAssign[idx2];
			if (cltId1 != cltId2) {
				countMLViolation++;
				penalty += 0.5 * (
					distanceByCluster(idx1, idx2, cltId1) +
					distanceByCluster(idx1, idx2, cltId2)
				);
			}
		}
	}
	//std::cout << "\t@itr " << currIter
		//<< " ML Violation = " << countMLViolation << '\n';
	return penalty * 0.5;
}

float PCKMeans::totalCLPenalty() {
	float penalty = 0.0f;
	countCLViolation = 0;

	for (const auto& it : constr->CL) {
		int idx1 = it.first;
		int cltId1 = vAssign[idx1];
		float maxDistanceByOneCluster = maxDistanceByCluster(cltId1);

		for (const int& idx2 : it.second) {
			int cltId2 = vAssign[idx2];
			if (cltId1 == cltId2) {
				countCLViolation++;
				penalty += (
					maxDistanceByOneCluster -
					distanceByCluster(idx1, idx2, cltId1)
				);
			}
		}
	}

	//std::cout << "\t@itr " << currIter
		//<< " CL Violation = " << countCLViolation << '\n';

	return penalty;
}

/*virtual*/ EMResult PCKMeans::getResult() {
    EMResult result = EMKMeans::getResult();

    result.nConstraintOriginal = constr->nConstraintsOriginal;
	result.nConstraintDeduced = constr->nConstraintsDeduced;
    result.nMLStart = constr->numML;
    result.nCLStart = constr->numCL;
    result.mlConst = constr->mlConst;
    result.clConst = constr->clConst;
    result.nMLViolation = countMLViolation;
    result.nCLViolation = countCLViolation;

    return result;
}

} /* namespace dml */
