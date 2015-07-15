// Name        : ml.cpp
// Author      : vvminh
// Version     : 1.0
// Copyright   : The MIT License (MIT)

// 
// The MIT License (MIT)
// 
// Copyright (c) [2015] [VU Viet Minh]
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
// 
// Description : implementation of some distance metric learning algorithm
// Update : 15-07-2015
//============================================================================


#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <cmath>
#include <chrono>

#include "utils/propertyutil.h"
#include "utils/dataUtils.h"
#include "utils/functionUtils.h"
#include "utils/testUtils.h"

#include "emkmeans/EMResult.h"
#include "emkmeans/EMKMeans.h"
#include "pckmeans/PCKMeans.h"
#include "mpckmeans/MPCKMeans.h"
#include "globalMetric/GlobalMetricKMeans.h"

#include "utils/Eigen3.h"
using namespace Eigen;

/**
 * read the file that containts name of all constraints files
 */
std::vector<std::string> getListOfConstraintFile(
        std::string listFileName, std::string dataDir);

/**
 * run experiment with one algorithm and one constraints file
 */
dml::EMResult executeAlgo(std::string algoName, std::string constraintFileName,
        const MatrixXf& inputData, int nClusters, int maxIter, float minObjChange,
        std::vector<int>& vAssign);

/**
 * calculate averge result of all repeats of one experimentation
 */
dml::EMResult calculateAvgResult(const std::vector<dml::EMResult>& results);

/**
 * write list of result objects to json file
 */
void writeListResultsToJson(const std::vector<dml::EMResult>& listResults,
	const std::string fileName);
void appendResult(const dml::EMResult result, const std::string fileName);

/**
 * main programm: 
 * -> read config file (propertiesFile) to get params
 * -> just one algorithm is chosen
 * -> run this algo with multiple (n) constraints-files 
 * -> each constraints-files contains different number of constraints
 * -> each constraints-files is one experiment
 * -> run each experiment (k) times to get avg result
 * -> write (n) results of (n) experiments into json file
 */
int main(int argc, char* argv[]) {
	// remember to init random seed for using globally
    initRandomSeed();

    // read params from properties file
	PropertyUtil::PropertyMapT params;
	PropertyUtil prop;
    std::string propertyFile = "../config/default.propertites";
    if (2 == argc) {
        propertyFile = std::string(argv[1]);
    }

    std::cout << "Using properties file: " << propertyFile << std::endl;
    prop.read(propertyFile.c_str(), params);

    std::cout << "Detected params: \n";
	prop.print(std::cout, params);

    // read input matrix
	std::string inputFile = params["dataDir"] + params["inputDataFile"];
	MatrixXf X = readMatrix(inputFile);
	std::cout << "Inputdata nExamples = " << X.cols()
		<< ", dimensions = " << X.rows() << std::endl;

    // mean-normalize input data (subtract mean from each column of X)
    MatrixXf X_aligned = X.colwise() - X.rowwise().mean();

    // read ground truth
    int nClasses = 0;
    std::string groundTruthFile = params["dataDir"] + params["groundTruthFile"];
    std::vector<int> vGroundTruthLabel = readGroundTruth(groundTruthFile, nClasses);

    // read algorithm params
    std::string algoName = params["algo"];
    int maxIter = std::stoi(params["maxIteration"]);
    float minObjChange = std::stof(params["minObjectiveFunctionChange"]);
    int nClusters = std::stoi(params["numberClusters"]);

    // get list constraints file
    std::string listConstraintFileName = params["listOfConstraintFile"];
    std::vector<std::string> vFiles = getListOfConstraintFile(
            listConstraintFileName.c_str(), params["dataDir"]);
    std::cout << "Run experiment with " << vFiles.size() << " different constraints files\n";

    // prepare experimentation
    int nRepeatTimes = stoi(params["repeatTimes"]);
    // std::vector<dml::EMResult> resultOfAllExperiments;
    float progressCount = 0;

    // create algorithm, and execute
    // for each experiment (each constraint file),
    // 		execute the algo k times separately and get the avg result
    using namespace std::chrono;
    for (const auto& constraintFileName : vFiles) {
        std::vector <dml::EMResult> oneExperiment;
        for (int nRun = 0; nRun < nRepeatTimes; ++nRun) {
            std::vector<int> vAssign;
            
            high_resolution_clock::time_point t1 = high_resolution_clock::now();
            dml::EMResult result = executeAlgo(algoName, constraintFileName,
			    X_aligned, nClusters, maxIter, minObjChange, vAssign);
            high_resolution_clock::time_point t2 = high_resolution_clock::now();
            if (-1 == result.reachLocalMinimal) {continue;}

            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();
            result.duration = (float)duration;
            result.vMeasure = VMeasure(vAssign, vGroundTruthLabel, nClasses, nClusters);
		    
            oneExperiment.push_back(result);
        }
        dml::EMResult avgResult;
        if (oneExperiment.size() > 0) {
            avgResult = calculateAvgResult(oneExperiment);   
        } else {
            avgResult.reachLocalMinimal = -1;//case error
        }
        // resultOfAllExperiments.push_back(avgResult);

        std::string resultFullName = params["resultDir"] + params["resultFile"];
        appendResult(avgResult, resultFullName);
        progressCount ++;

        std::cout << "Progress: " << (progressCount / vFiles.size() * 100.0) << std::endl;
    }

    // save result to json file to analyse
	// std::string resultFullName = params["resultDir"] + params["resultFile"];
    // std::cout << "Write avg result to " << resultFullName << std::endl;
	// writeListResultsToJson(resultOfAllExperiments, resultFullName);
    
    std::cout << "\nCode done! Release resource\n\n";
	return 0;
}

std::vector<std::string> getListOfConstraintFile(
		std::string listFileName, std::string dataDir) {
    std::ifstream infile((dataDir + listFileName).c_str());
    if (!infile.is_open()) {
        throw std::runtime_error("Can not open list constraints file: " + listFileName);
    }
    std::vector<std::string> vFiles;
    std::string sName;
    while (infile >> sName) {
        vFiles.push_back(dataDir + sName);
    }
    infile.close();
    return vFiles;
}

dml::EMResult executeAlgo(std::string algoName, std::string constraintFileName,
        const MatrixXf& X, int nClusters, int maxIter, float minObjChange,
        std::vector<int>& vAssign ) {
    std::cout << "\nExecute " << algoName << " with " << constraintFileName << "\n";

    dml::EMKMeans* emkmeans;
    if (0 == algoName.compare("PCKMEANS_NOMETRIC")) {
        emkmeans = new dml::GlobalMetricKMeans(
			X, nClusters, constraintFileName, dml::DIST_EUCLIDEAN);
    } else if (0 == algoName.compare("MPCKMEANS_GLOBAL_DIAGONAL")) {
		emkmeans = new dml::GlobalMetricKMeans(
			X, nClusters, constraintFileName, dml::DIST_MAHALANOBIS_DIAG);
    } else if (0 == algoName.compare("MPCKMEANS_GLOBAL_FULL")) {
        emkmeans = new dml::GlobalMetricKMeans(
			X, nClusters, constraintFileName, dml::DIST_MAHALANOBIS_FULL);
    } else if (0 == algoName.compare("MPCKMEANS_LOCAL_DIAGONAL")) {
        emkmeans = new dml::MPCKMeans(
			X, nClusters, constraintFileName, dml::COV_DIAG);
    } else if (0 == algoName.compare("MPCKMEANS_LOCAL_FULL")) {
        emkmeans = new dml::MPCKMeans(
			X, nClusters, constraintFileName, dml::COV_FULL);
    } else {
        throw std::runtime_error("Can not detect algorithm " + algoName);
    }

    dml::EMResult result;
    try {
        vAssign = emkmeans->doClustering(maxIter, minObjChange);
        result = emkmeans->getResult();
    } catch (...) {
        std::cout << "DIE HARD\n";
        result.reachLocalMinimal = -1;//case error
    }

    delete emkmeans;
	return result;
}

dml::EMResult calculateAvgResult(const std::vector<dml::EMResult>& allResults) {
    dml::EMResult avgResult;
    int nResultOk = 0;
    for (const auto& oneResult : allResults) {
        if (oneResult.reachLocalMinimal > 0) {
            nResultOk ++;
            avgResult.add(oneResult);
        }
    }
    if (nResultOk > 0) {
        avgResult.divise(nResultOk);
    }
    // the pourcentage of reaching local minimal
    avgResult.reachLocalMinimal = (1.0 * nResultOk) / (float)allResults.size();
    return avgResult;
}

void writeListResultsToJson(const std::vector<dml::EMResult>& listResults,
	const std::string fileName) {
    std::ofstream outfile(fileName.c_str());
    if (!outfile.is_open()) {
        std::cerr << "Can not open outfile to write json: " << fileName << "\n";
        std::cerr << "Try to open default outfile: default_outfile.json\n";
        outfile.open("default_outfile.json");
    }

    outfile << "[";
    for (const auto& oneResult : listResults) {
        outfile << std::endl << oneResult.toJson() << ",";
    }
    outfile.seekp(-1,std::ios::end);
    outfile << "\n]";
    outfile.close();
}

void appendResult(const dml::EMResult result, const std::string fileName) {
    std::ofstream outfile;
    outfile.open(fileName.c_str(), std::fstream::app);
    if (!outfile.is_open()) {
        std::cerr << "Can not open outfile to write json: " << fileName << "\n";
        std::cerr << "Try to open default outfile: default_outfile.json\n";
        outfile.open("default_outfile.txt", std::fstream::app);
    }
    outfile << std::endl << result.toJson() << ",";
    outfile.close();
}