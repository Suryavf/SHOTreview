#include "shot.h"
#include "utils.h"
#include <iostream>
#include <vector>

#include <vtkSphereSource.h>
#include <vtkPolyData.h>
#include <vtkPLYReader.h>
#include <vtkSmartPointer.h>
#include <vtkPolyDataPointSampler.h>
#include <vtkPolyDataMapper.h>
#include <vtkTransform.h>
#include <vtkTransformPolyDataFilter.h>
#include <vtkActor.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkProperty.h>

#include "opencv2/opencv.hpp"
#include <fstream>

std::string original = "../data/gata001_768.ply";
std::string scene    = original; 

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
		shotShapeBins = 10;
		shotColorBins = 30;
		matchTh = 0.70f;
		radiusMR = 20;
		sigmaNoiseMR = 0.3;
		nFeat = 5000;
		minNeighbors = 5;
		describeShape = true;
		describeColor = false;
		rotationAngle = 60.0f;
		rotationAxis[0] = 0.75f;
		rotationAxis[1] = 0.1f;
		rotationAxis[2] = 1-0.75f*0.75f-0.1f*0.1f;

#ifdef __linux__		
		datapath = "/home/victor/shot/data/dataset3/3D models/CVLab/2009-10-27/Scene1.ply";//"../data/Mario.ply";
		outputFile = "../bin/shot.txt";
#else
		datapath = "/home/victor/shot/data/dataset3/3D models/CVLab/2009-10-27/Scene1.ply";//"../../data/Mario.ply";
		outputFile = "../../bin/shot.txt";
#endif

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
	
	//
	// SHOT settings
	// -------------------------------------------------
	SHOTParams shotParams;
	shotParams.minNeighbors  = params.minNeighbors;
	shotParams.shapeBins     = params.shotShapeBins;
	shotParams.colorBins     = params.shotColorBins;
	shotParams.describeColor = params.describeColor;
	shotParams.describeShape = params.describeShape;
	shotParams.nThreads      = params.nThreads;

	double meshRes;

	//
	// Original
	// -------------------------------------------------
	vtkPolyData* meshOriginal;
	meshOriginal = LoadPolyData(original);
	cleanPolyData(meshOriginal);

	meshRes = computeMeshResolution(meshOriginal);
	shotParams.radius        = meshRes * params.radiusMR;
	shotParams.localRFradius = meshRes * params.radiusMR;

	// Random 3D detector Setting
	Random3DDetector  detectorOriginal(params.nFeat, true, meshRes * params.radiusMR, params.minNeighbors);
	SHOTDescriptor  descriptorOriginal(shotParams);
	
	Feature3D* featOriginal;
	int nOriginalFeat = detectorOriginal.extract(meshOriginal, featOriginal);
	
	double** originalDesc;
	descriptorOriginal.describe(meshOriginal, featOriginal, originalDesc, nOriginalFeat);
	
	cv::Mat featuresOriginal(params.nFeat, descriptorOriginal.getDescriptorLength(), CV_32FC1);
	for (int i = 0; i < nOriginalFeat; i++){
		for (int j = 0; j < descriptorOriginal.getDescriptorLength(); j++){
			featuresOriginal.at<float>(i,j) = originalDesc[i][j];
		}
	}
	
	//
	// Scene
	// -------------------------------------------------
	
	vtkPolyData* meshScene;
	meshScene = LoadPolyData(scene);
	cleanPolyData(meshScene);


	double noiseSigma = params.sigmaNoiseMR * meshRes;
	rotate(meshScene, params.rotationAngle, params.rotationAxis);
	addGaussianNoise(meshScene, noiseSigma);
	computeNormals(meshScene);


	meshRes = computeMeshResolution(meshScene);
	shotParams.radius        = meshRes * params.radiusMR;
	shotParams.localRFradius = meshRes * params.radiusMR;

	// Random 3D detector Setting
	Random3DDetector  detectorScene(params.nFeat, true, meshRes * params.radiusMR, params.minNeighbors);
	SHOTDescriptor  descriptorScene(shotParams);
	
	Feature3D* featScene;
	int nSceneFeat = detectorScene.extract(meshScene, featScene);
	
	double** sceneDesc;
	descriptorScene.describe(meshScene, featScene, sceneDesc, nSceneFeat);


	cv::flann::Index kdtree(featuresOriginal, cv::flann::KDTreeIndexParams());
	std::vector<float> dists;
	dists.resize(2);
	std::vector<int> knn;
	knn.resize(2);

	Feature3D aux;
	std::vector <Feature3D> keyPointOriginal;
	std::vector <Feature3D> keyPointScene   ;
	for(int i = 0; i < nOriginalFeat; i++){
		std::vector<float> query;
		for (int j = 0; j < descriptorScene.getDescriptorLength(); j++)
			query.push_back(sceneDesc[i][j]);

		/*
		knnSearch (const Mat& query, Mat& indices, Mat& dists, int knn, const SearchParams& params)
		---------
		query   – The query point
		indices – Vector that will contain the indices of the K-nearest neighbors found. It must have at least knn size.
		dists   – Vector that will contain the distances to the K-nearest neighbors found. It must have at least knn size.
		knn     – Number of nearest neighbors to search for.
		*/
		kdtree.knnSearch(query, knn, dists, 2, cv::flann::SearchParams());

		assert(dists[0] <= dists[1]);

		if (dists[0] <= params.matchTh * params.matchTh * dists[1]){
			keyPointOriginal.push_back(featOriginal[knn[0]]);
			keyPointScene   .push_back(featScene   [i]);

			printf("Match %d: %d\n", i, knn[0]);
		}
	}
	

	// 
	// VTK Original
	// -------------------------------------------------
	vtkSmartPointer<vtkPLYReader> reader1 = vtkSmartPointer<vtkPLYReader>::New();
	reader1->SetFileName ( original.c_str() );

	vtkSmartPointer<vtkPolyDataMapper> mapper1 = vtkSmartPointer<vtkPolyDataMapper>::New();
	mapper1->SetInputConnection(reader1->GetOutputPort());

	vtkSmartPointer<vtkActor> actorOriginal = vtkSmartPointer<vtkActor>::New();
	actorOriginal->SetMapper(mapper1);


	// 
	// VTK Scene
	// -------------------------------------------------
	//vtkSmartPointer<vtkPLYReader> reader2 = vtkSmartPointer<vtkPLYReader>::New();
	//reader2->SetFileName ( scene.c_str() );
	
	vtkSmartPointer<vtkPolyDataPointSampler> reader2 = vtkSmartPointer<vtkPolyDataPointSampler>::New();
	reader2->SetInputData(meshScene);

	vtkSmartPointer<vtkTransform> translation = vtkSmartPointer<vtkTransform>::New();
	translation->Translate(0.50, 0.00, 0.0);//(0.50, 0.00, 0.0);

	vtkSmartPointer<vtkTransformPolyDataFilter> transformFilter = vtkSmartPointer<vtkTransformPolyDataFilter>::New();
	transformFilter->SetInputConnection(reader2->GetOutputPort());
	transformFilter->SetTransform(translation);
	transformFilter->Update();

	vtkSmartPointer<vtkPolyDataMapper> mapper2 = vtkSmartPointer<vtkPolyDataMapper>::New();
	mapper2->SetInputConnection(transformFilter->GetOutputPort());

	vtkSmartPointer<vtkActor> actorScene = vtkSmartPointer<vtkActor>::New();
	actorScene->SetMapper(mapper2);


	// 
	// VTK Render/Visualization
	// -------------------------------------------------
	vtkSmartPointer<vtkRenderer    > renderer     = vtkSmartPointer<vtkRenderer    >::New();
	vtkSmartPointer<vtkRenderWindow> renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
	renderWindow->AddRenderer(renderer);
	vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
	renderWindowInteractor->SetRenderWindow(renderWindow);

	renderer->SetBackground(0.1804,0.5451,0.3412); // Sea green
	renderer->AddActor(actorOriginal);
	renderer->AddActor(actorScene   );


	//
	// VTK Add Points
	// -------------------------------------------------
	vtkSmartPointer<vtkSphereSource  > sphereSource;
	vtkSmartPointer<vtkPolyDataMapper>  mapperPoint;
	vtkSmartPointer<vtkActor         >   actorPoint;

	vtkSmartPointer<vtkTransformPolyDataFilter> transScenePoints;
	
	float r,g,b;
	for (size_t i = 0; i< keyPointOriginal.size();++i){
		r = ((float) rand() / (RAND_MAX));
		g = ((float) rand() / (RAND_MAX));
		b = ((float) rand() / (RAND_MAX));
		
		// Original
		sphereSource = vtkSmartPointer<vtkSphereSource>::New();
		sphereSource->SetCenter(keyPointOriginal[i].x, keyPointOriginal[i].y, keyPointOriginal[i].z);
  		sphereSource->SetRadius(0.01);
		
		mapperPoint = vtkSmartPointer<vtkPolyDataMapper>::New();
		mapperPoint->SetInputConnection(sphereSource->GetOutputPort());

		actorPoint = vtkSmartPointer<vtkActor>::New();
		actorPoint->GetProperty()->SetColor(r, g, b);
		actorPoint->SetMapper(mapperPoint);

		renderer->AddActor(actorPoint);


		// Scene
		sphereSource = vtkSmartPointer<vtkSphereSource>::New();
		sphereSource->SetCenter(keyPointScene[i].x, keyPointScene[i].y, keyPointScene[i].z);
  		sphereSource->SetRadius(0.01);

		transScenePoints = vtkSmartPointer<vtkTransformPolyDataFilter>::New();
		transScenePoints->SetInputConnection(sphereSource->GetOutputPort());
		transScenePoints->SetTransform(translation);
		transScenePoints->Update();

		mapperPoint = vtkSmartPointer<vtkPolyDataMapper>::New();
		mapperPoint->SetInputConnection(transScenePoints->GetOutputPort());

		actorPoint = vtkSmartPointer<vtkActor>::New();
		actorPoint->GetProperty()->SetColor(r, g, b);
		actorPoint->SetMapper(mapperPoint);

		renderer->AddActor(actorPoint);
	}
	
	renderWindow->Render();
	renderWindowInteractor->Start();

	meshOriginal->Delete();
	meshScene   ->Delete();
	
	getchar();
}
