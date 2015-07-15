/*
 * testUtils.h
 *
 *  Created on: Jun 27, 2015
 *      Author: vvminh
 */

#ifndef UTILS_TESTUTILS_H_
#define UTILS_TESTUTILS_H_

#include <sstream>
#include <iostream>
#include <fstream>
#include <string>

#include "Eigen3.h"
#include "functionUtils.h"

using namespace Eigen;

void testAlignMat() {
	Matrix3f m;
	m << 1, 	1.2, 	2,
		 1.7, 	1, 		1.2,
		 3, 	4, 		5;
	std::cout << "m:\n" << m << '\n';

	Vector3f mean = m.rowwise().mean();
	std::cout << "mean: \n" << mean << '\n';

	Matrix3f m_aligned = m.colwise() - mean;
	std::cout << "m aligned: \n" << m_aligned << '\n';

	Vector3f m_cov = m_aligned.array().square().matrix().rowwise().sum();
	std::cout << "m cov: \n" << m_cov << '\n';

	Matrix3f temp = m_aligned.array().square() / m.array();
	std::cout << "cwiseQuotient\n" << temp << '\n';
}

void testStandardize() {
	MatrixXf m(2, 5);
	m << 600, 470, 170, 430, 300,
		300, 430, 170, 470, 600;
	std::cout << "m\n" << m << '\n';

	VectorXf mean = m.rowwise().mean();
	std::cout << "mean\n" << mean << '\n';

	MatrixXf m_aligned = m.colwise() - mean;

	// way1: auto (note sqrt(5) = 2.23606)
	MatrixXf m_std1 = m_aligned.rowwise().normalized() * 2.23606;
	std::cout << "std1\n" << m_std1 << '\n';

	// way2: manual
	VectorXf sd_square = m_aligned.array().square().matrix().rowwise().sum();
	sd_square /= 5;
	VectorXf sd = sd_square.array().sqrt().matrix();
	std::cout << "sd:\n" << sd << '\n';

	MatrixXf m_std2 = m_aligned.array() / sd.replicate<1, 5>().array();
	std::cout << "m_std2\n" << m_std2 << '\n';
}

void testMultiplyMat() {
	MatrixXf m1 = MatrixXf::Random(3, 3);
	std::cout << "m1 before: \n" << m1 << '\n';

	MatrixXf m2 = MatrixXf::Random(3, 3);
	std::cout << "m2 before: \n" << m2 << '\n';

	m2 = m1 * m2;
	std::cout << "m2 after: \n" << m2 << '\n';

}

void compareSVD_EVD(const Ref<const MatrixXf>& X_aligned) {
	int nData = X_aligned.cols();
	MatrixXf covMat = (1.0 / nData) * X_aligned * X_aligned.transpose();

	std::cout << "Do SVD:\n";
	JacobiSVD<MatrixXf> svd(covMat, ComputeThinU|ComputeThinV);

	std::cout << "DO EVD:\n";
	EigenSolver<MatrixXf> es(covMat);

	std::cout << "Compare EigenVectors: " 
		<< svd.matrixU().isApprox(es.eigenvectors().real()) << '\n';
	std::cout << "Matrix U: \n" << svd.matrixU().block(0, 0, 5, 5) << '\n';
	std::cout << "Matrix V: \n" << svd.matrixV().block(0, 0, 5, 5) << '\n';
	std::cout << "Matrix E: \n" << es.eigenvectors().real().block(0, 0, 5, 5) << '\n';


	std::cout << "Compare EigenValues: \n";
	std::cout << "Singular Values: \n"
		<< svd.singularValues().head(20).transpose() << '\n';
	std::cout << "Eigen Values: \n" 
		<< es.eigenvalues().real().head(20).transpose() << '\n';
}

void testLogDet() {
	Matrix4f X = Matrix4f::Random();
	Matrix4f X_aligned = X.colwise() - X.rowwise().mean();

	Matrix4f covX;
	covX = (1.0 / 4) * X_aligned * X_aligned.transpose();
	
	std::cout << "Caculate logDet(invCov) directely:\n";
	Matrix4f covInv;
	covInv = X.inverse();
	std::cout << "covInv\n" << covInv << '\n';

	float det = covX.determinant();
	float logDet = std::log(det);
	std::cout << "det = " << det << "\t logdet = " << logDet << "\n";
	std::cout << "1/det = " << 1.0 / det << "log(1/det)" << std::log(1.0/det) << "\n\n";

	float detOfInv = covInv.determinant();
	float logDetOfInv = std::log(detOfInv);
	std::cout << "det of Inv = " << detOfInv << "\t logdetInv = " << logDetOfInv << "\n\n";

	std::cout << "Using SVD:\n";
	JacobiSVD<Matrix4f> svd(covX, ComputeFullU | ComputeFullV);
	Vector4f S = svd.singularValues();
	float logDet_SVD = S.array().log().sum();
	std::cout << "LogDet SVD = " << logDet_SVD << "\n\n";

	std::cout << "Using EVD:\n";
	EigenSolver<Matrix4f> es(covX);
	Vector4f D = es.eigenvalues().real();
	float logDet_EVD = D.array().log().sum();
	std::cout << "LogDet EVD = " << logDet_EVD << "\n\n";

	std::cout << "Using Chol Decomp:\n";
	LLT<Matrix4f> llt(covX);
	Matrix4f L = llt.matrixL();
	float logDet_LLT = 2.0f * L.diagonal().array().log().sum();
	std::cout << "LogDet LLT = " << logDet_LLT << "\n\n";

	std::cout << "Using QR Decomp:\n";
	HouseholderQR<MatrixXf> qr(covX);
	float logAbsDet= qr.logAbsDeterminant();
	std::cout << "Log of Abs of Det = " << logAbsDet << "\n\n";

}

void testMahaDist() {
	std::cout << "Test mahalanobis distance\n";

	VectorXf mean(2);
	mean << 0, 0;
	MatrixXf covX(2, 2);
	covX <<     
		0.9229,    0.8273,
    	0.8273,    0.9253;
	std::cout << "Cov mat: " << covX << '\n';

	VectorXf v1 = VectorXf(2, 1);
	v1 << 0.84, -0.12;

	std::cout << "Inv :\n";
	MatrixXf covInv = covX.inverse();
	std::cout << "Inv Cov Mat : \n" << covInv << '\n';
	float det = covInv.determinant();
	std::cout << "det = " << det << "\tlogDet = " << std::log(det) << "\n";

	VectorXf diff = v1 - mean;
	float d = diff.transpose() * covInv * diff;
	std::cout << "Dist = " << std::sqrt(d) << "\n\n";

	std::cout << "Using EVD:\n";
	EigenSolver<MatrixXf> es(covX);
	VectorXf D = es.eigenvalues().real();
	float logDet = D.array().log().sum();
	std::cout << "\tlogDet = " << (- logDet) << "\n";

	MatrixXf E = es.eigenvectors().real();
	VectorXf diffTrans = E.transpose() * (v1 - mean);
	float d_evd = (diffTrans.array().square() / D.array()).sum();
	std::cout << "dist EVD = " << std::sqrt(d_evd) << "\n\n";

	std::cout << "Using SVD:\n";
	JacobiSVD<MatrixXf> svd(covX, ComputeFullU);
	VectorXf S = svd.singularValues();
	float logDet2 = S.array().log().sum();
	std::cout << "\tlogDet2 = " << (- logDet2) << "\n";

	MatrixXf U = svd.matrixU();
	VectorXf diffTrans2 = U.transpose() * (v1 - mean);
	float d_svd = (diffTrans2.array().square() / S.array()).sum();
	std::cout << "dist SVD = " << std::sqrt(d_svd) << "\n\n";

}

#endif /* UTILS_TESTUTILS_H_ */
