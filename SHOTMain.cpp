/*
	Copyright (C) 2010 Samuele Salti, Federico Tombari, all rights reserved.

	This file is part of SHOT. SHOT has been developed by the 
	Computer Vision Laboratory of the University of Bologna
	(http://www.vision.deis.unibo.it)
	
	SHOT is an implementation of the work described in
	F. Tombari, S. Salti and L. Di Stefano 
	"Unique Signatures of Histograms for Local Surface Description"
	The 11th IEEE European Conference on Computer Vision (ECCV) 2010

	Contacts:
	Samuele Salti mailto:samuele.salti@unibo.it
	Federico Tombari mailto:federico.tombari@unibo.it


    SHOT is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    SHOT is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with SHOT.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "shot.h"
#include "utils.h"

#include <vtkXMLPolyDataWriter.h>
#include <vtkPLYWriter.h>
#include <vtkSmartPointer.h>


#include <iostream>
#include <vector>

#include "opencv2/opencv.hpp"
#include <fstream>

struct CmdLineParams
{
	float matchTh;
	double radiusMR;
	double sigmaNoiseMR;
	int nFeat;
	int minNeighbors;
	float rotationAngle;
	float rotationAxis[3];

	int nThreads;

	bool describeColor;
	bool describeShape;
	
	std::string datapath;
	std::string outputFile;

	int shotShapeBins;
	int shotColorBins;

	CmdLineParams() //default
	{
		shotShapeBins = 30;
		shotColorBins = 30;
		matchTh = 0.75f;//0.25
		radiusMR = 50;
		sigmaNoiseMR = 0.1;
		nFeat = 2000;
		minNeighbors = 200;
		describeShape = true;
		describeColor = false;
		rotationAngle = 60.0f;
		rotationAxis[0] = 0.75f;
		rotationAxis[1] = 0.1f;
		rotationAxis[2] = 1-0.75f*0.75f-0.1f*0.1f;

	
		datapath = "../Scene4.ply";
		outputFile = "../bin/shot.txt";

		nThreads = 0;
	}

	bool parseCommandLine(int argc, char** argv)
	{
		int tempArgC = 1;
		while ((tempArgC < argc)){
			
			if (argv[tempArgC][0] != '-')
				break;

			if (argv[tempArgC][1] != 'e' && argv[tempArgC][1] != 'f' && (tempArgC+1 >= argc) )
				break;

			switch(argv[tempArgC][1])
			{
			case 'b':
				shotShapeBins = atoi(argv[tempArgC+1]);
				break;
			case 'c':
				shotColorBins = atoi(argv[tempArgC+1]);
				break;
			case 'e':
				describeShape = false;
				tempArgC--;
				break;
			case 'f':
				describeColor = false;
				tempArgC--;
				break;
			case 'i':
				datapath = argv[tempArgC+1];
				break;
			case 'k':
				nFeat = atoi(argv[tempArgC+1]);
				break;
			case 'n':
				minNeighbors = atoi(argv[tempArgC+1]);
				break;
			case 'o':
				outputFile = argv[tempArgC+1];
				break;
			case 'p':
				nThreads = atoi(argv[tempArgC+1]);
				break;
			case 'r':
				radiusMR = atof(argv[tempArgC+1]);
				break;
			case 's':
				sigmaNoiseMR = atof(argv[tempArgC+1]);
				break;
			case 't':
				matchTh = atof(argv[tempArgC+1]);
				break;
			case 'w':
				rotationAngle = atof(argv[tempArgC+1]);
				break;
			case 'x':
				rotationAxis[0] = atof(argv[tempArgC+1]);
				break;
			case 'y':
				rotationAxis[1] = atof(argv[tempArgC+1]);
				break;
			case 'z':
				rotationAxis[2] = atof(argv[tempArgC+1]);
				break;
			}
			tempArgC+=2;
		}

		return (tempArgC == argc);
	};
};



int main(int argc, char** argv)
{
	CmdLineParams params;
	if (!params.parseCommandLine(argc, argv))
	{
		std::cout << "Error on command line!\n";
		std::cout << "Usage: SHOT  -[optionSelector] optionValue  -[optionSelector] optionValue ...\n";
		std::cout << "Options:\n";
		std::cout << "\t a \t\t number of bins in each color Histogram of the SHOT descriptor\n";
		std::cout << "\t b \t\t number of bins in each shape Histogram of the SHOT descriptor\n";
		std::cout << "\t e \t\t insert this flag to disable shape description\n";
		std::cout << "\t f \t\t insert this flag to disable color description\n";
		std::cout << "\t i \t\t input mesh (PLY, OFF and OBJ format supported)\n";
		std::cout << "\t k \t\t number of random feature points to describe\n";
		std::cout << "\t n \t\t minimum points in the neighborhood to describe a point\n";
		std::cout << "\t o \t\t output file (one descriptor per row)\n";
		std::cout << "\t p \t\t number of threads (0 for hardware concurrency)\n";
		std::cout << "\t r \t\t neighborhood raidus (in unit of mesh resolution)\n";
		std::cout << "\t s \t\t noise sigma (in unit of mesh resolution)\n";
		std::cout << "\t t \t\t matching threshold\n";
		std::cout << "\t w \t\t rotation angle between original and noisy mesh (in degrees)\n";
		std::cout << "\t x,y,z \t\t rotation axis components between original and noisy mesh\n";
	}
	//datapath += "\\3D models\\Aim@Shape Watertight\\data\\1.off";
	//datapath += "\\3D models\\Stanford\\dragon\\dragon_vrip_res2.off";


	vtkPolyData* mesh = LoadPolyData(params.datapath);
	cleanPolyData(mesh);

	vtkPolyData* mesh2 = LoadPolyData("../dragon.ply"); //LoadPolyData("../estatua.ply");
	cleanPolyData(mesh2);


	/*
	vtkSmartPointer<vtkPLYWriter> writer=vtkSmartPointer<vtkPLYWriter>::New(); 
    writer->SetInputData(mesh2); 
    //writer->SetFileTypeToASCII(); 
    //writer->SetColorModeToDefault(); 
    //writer->SetArrayName("RGB"); //Pasa el color
    writer->SetFileName("esta.ply"); 
    writer->Write();*/

    //int nFeat = mesh->GetNumberOfPoints()/2;
    int nFeat = 10000;//mesh->GetNumberOfPoints();
    int nFeat2 = 10000;//mesh2->GetNumberOfPoints();
    cout<<"Cantidad Puntos: "<< nFeat <<endl;
    cout<<"Cantidad Puntos2: "<< nFeat2 <<endl;

    if (mesh->GetNumberOfPoints())
	{
		double meshRes = computeMeshResolution(mesh);

		// Select feature points;
		Random3DDetector detector(nFeat, true, meshRes * params.radiusMR, params.minNeighbors);

		Feature3D* feat;
		
		int nActualFeat = detector.extract(mesh, feat);

		SHOTParams shotParams;
		shotParams.radius = meshRes * params.radiusMR;
		shotParams.localRFradius = meshRes * params.radiusMR;
		shotParams.minNeighbors = params.minNeighbors;
		shotParams.shapeBins = params.shotShapeBins;
		shotParams.colorBins = params.shotColorBins;
		shotParams.describeColor = params.describeColor;
		shotParams.describeShape = params.describeShape;
		shotParams.nThreads = params.nThreads;

		SHOTDescriptor descriptor(shotParams);

		double** desc;
		descriptor.describe(mesh, feat, desc, nActualFeat);

		std::ofstream outfile(params.outputFile.c_str());
		if (!outfile.is_open())
			std::cout << "\nWARNING\n It is not possible to write on the requested output file\n";

		cv::Mat features(nFeat, descriptor.getDescriptorLength(), CV_32FC1);
		for (int i = 0; i < nActualFeat; i++)
		{
			if (outfile.is_open())
				outfile << feat[i].index << " " << feat[i].x << " " << feat[i].y << " " << feat[i].z;

			for (int j = 0; j < descriptor.getDescriptorLength(); j++)
			{
				features.at<float>(i,j) = desc[i][j];
				if (outfile.is_open())
					outfile << " " << desc[i][j];
			}
			if (outfile.is_open())
					outfile << "\n ";
		}

		double noiseSigma = params.sigmaNoiseMR * meshRes;
		//rotate(mesh, params.rotationAngle, params.rotationAxis);






		double meshRes2 = computeMeshResolution(mesh2);
		// Select feature points;
		Random3DDetector detector2(nFeat2, true, meshRes2 * params.radiusMR, params.minNeighbors);
		Feature3D* feat2;
		int nActualFeat2 = detector2.extract(mesh2, feat2);



		
		


		//addGaussianNoise(mesh, noiseSigma);
		computeNormals(mesh2);

		SHOTDescriptor descriptor2(shotParams);
		double** noisyDesc;
		descriptor2.describe(mesh2, feat2, noisyDesc, nActualFeat2);


		cv::flann::Index kdtree(features, cv::flann::KDTreeIndexParams());
		std::vector<float> dists;
		dists.resize(2);
		std::vector<int> knn;
		knn.resize(2);
		int correctMatch = 0, totalMatch = 0;

		//Puntos
		vtkPoints *pointsN = vtkPoints::New();
		double x[3];
		int pos = 0;

		for(int i = 0; i < nActualFeat2; i++)
		{
			std::vector<float> query;
			for (int j = 0; j < descriptor2.getDescriptorLength(); j++)
				query.push_back(noisyDesc[i][j]);

			kdtree.knnSearch(query, knn, dists, 2, cv::flann::SearchParams());

			assert(dists[0] <= dists[1]);

			if (dists[0] <= params.matchTh * params.matchTh * dists[1])
			{
				//printf("Match %d: %d\n", i, knn[0]);

				//mesh->GetPoint(knn[0], x);
				mesh->GetPoint(detector.idsRandon[knn[0]], x);

				pointsN->InsertPoint(pos,  x);
				pos++;

				if (i == knn[0])
					correctMatch++;

				totalMatch++;
			}
		}

		vtkPolyData *polyN = vtkPolyData::New();
		polyN->SetPoints(pointsN);

		vtkSmartPointer<vtkPLYWriter> writer=vtkSmartPointer<vtkPLYWriter>::New(); 
	    writer->SetInputData(polyN); 
	    //writer->SetFileTypeToASCII(); 
	    //writer->SetColorModeToDefault(); 
	    //writer->SetArrayName("RGB"); //Pasa el color
	    writer->SetFileName("meshOut.ply"); 
	    writer->Write(); 

		printf("\n\nDescribed keypoints: %d. \nCorrect Matches: %d out of %d. \nMatches under threshold: %d. \nRecall: %f, 1-Precision: %f\n", 
			nActualFeat, correctMatch, totalMatch, nActualFeat - totalMatch, correctMatch*1.0/nActualFeat, (totalMatch - correctMatch)*1.0/totalMatch);
		mesh->Delete();
	}

	getchar();
}
