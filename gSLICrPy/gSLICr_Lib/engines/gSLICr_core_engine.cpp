// Copyright 2014-2015 Isis Innovation Limited and the authors of gSLICr

#pragma once
#include "gSLICr_core_engine.h"
#include <fstream>
#include <iostream>

using namespace gSLICr;
using namespace std;

gSLICr::engines::core_engine::core_engine(const objects::settings& in_settings)
{
	slic_seg_engine = new seg_engine_GPU(in_settings);
}

gSLICr::engines::core_engine::~core_engine()
{
		delete slic_seg_engine;
}

void gSLICr::engines::core_engine::Process_Frame(UChar4Image* in_img)
{
	slic_seg_engine->Perform_Segmentation(in_img);
}

const IntImage * gSLICr::engines::core_engine::Get_Seg_Res()
{
	return slic_seg_engine->Get_Seg_Mask();
}

void gSLICr::engines::core_engine::Draw_Segmentation_Result(UChar4Image* out_img)
{
	slic_seg_engine->Draw_Segmentation_Result(out_img);
}

void gSLICr::engines::core_engine::Write_Seg_Res_To_TXT(const char* fileName)
{
	const IntImage* idx_img = slic_seg_engine->Get_Seg_Mask();
	int width = idx_img->noDims.x;
	int height = idx_img->noDims.y;
	const int* data_ptr = idx_img->GetData(MEMORYDEVICE_CPU);

    ofstream f;
    f.open(fileName);
//	ofstream f(fileName, std::ios_base::out | std::ios_base::binary | std::ios_base::trunc);
//	f << "P5\n" << width << " " << height << "\n65535\n";
    cout<< "len of txt file is "<< height*width<<endl;
	for (int i = 0; i < height*width; ++i)
    {
        //for (int j =0; j < width; ++j)
	    //{
		 ushort label = (ushort)data_ptr[i];
         f << label <<"\n" ;
		//ushort lable_buffer = (lable << 8 | lable >> 8);
		//f.write((const char*)&lable_buffer, sizeof(ushort));
	//}
        //f << "\n";
    }

    cout<<"saving superpixel info to " <<fileName <<endl;
	f.close();
}
