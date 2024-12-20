#include "TH1D.h"
#include "TFile.h"
#include "TCanvas.h"
#include "TLegend.h"
#include <vector>

#include <iostream>

using namespace std;

int drawAndSaveHists(TString parentDir, int fields){

	TFile *file = new TFile("file.root", "read");
    
	if(!file){
		cout << "Could not load file, leaving..." << endl;
		return 1;
	}

    vector<TString> descriptions;
    descriptions.push_back("source size_y = 0 #mu m");
    descriptions.push_back("source size_y = 10 #mu m");
    descriptions.push_back("source size_y = 50 #mu m");
    descriptions.push_back("source size_y = 100 #mu m");


    vector<TH1D*> histY;
    vector<TH1D*> histPx;
    vector<TH1D*> histPy;
    vector<TH1D*> histX;

    histPx.push_back( (TH1D*)file->Get("histPx0") );
    histPx.push_back( (TH1D*)file->Get("histPx1") );
    histPx.push_back( (TH1D*)file->Get("histPx2") );
    histPx.push_back( (TH1D*)file->Get("histPx3") );

    histPy.push_back( (TH1D*)file->Get("histPy0") );
    histPy.push_back( (TH1D*)file->Get("histPy1") );
    histPy.push_back( (TH1D*)file->Get("histPy2") );
    histPy.push_back( (TH1D*)file->Get("histPy3") );

    histY.push_back( (TH1D*)file->Get("histY0") );
    histY.push_back( (TH1D*)file->Get("histY1") );
    histY.push_back( (TH1D*)file->Get("histY2") );
    histY.push_back( (TH1D*)file->Get("histY3") );

    histX.push_back( (TH1D*)file->Get("histX0") );
    histX.push_back( (TH1D*)file->Get("histX1") );
    histX.push_back( (TH1D*)file->Get("histX2") );
    histX.push_back( (TH1D*)file->Get("histX3") );


    // decide how to name the output files
    vector<TString> fileName;
    if( parentDir.Contains("1") ){
        parentDir.Remove( parentDir.Index("1"), 1 );
        fileName.push_back("topHatFieldsPlus.pdf");
        fileName.push_back("astraFringeFieldsPlus.pdf");
        fileName.push_back("fieldProfilesPlus.pdf");
    }else if( parentDir.Contains("2") ){
        parentDir.Remove( parentDir.Index("2"), 1 );
        fileName.push_back("topHatFieldsMinus.pdf");
        fileName.push_back("astraFringeFieldsMinus.pdf");
        fileName.push_back("fieldProfilesMinus.pdf");
    }else{
        fileName.push_back("topHatFields.pdf");
        fileName.push_back("astraFringeFields.pdf");
        fileName.push_back("fieldProfiles.pdf");
    }


    TCanvas *canvas = new TCanvas("canvas", "canvas", 1000, 800);
    gStyle->SetOptStat(0);
    canvas->Divide(2,2);
    canvas->cd(1);


    for (int i = 0; i < histPx.size(); ++i){
        histPx[i]->SetLineColor(i+2);
        histPx[i]->SetLineWidth(2);
        histPx[i]->SetFillColorAlpha(i+2, 0.3);
        histPx[i]->SetMarkerStyle(20);
        histPx[i]->SetMarkerColor(i+2);
        if(i == 0){
            histPx[i]->Draw("HIST E");
        }else{
            histPx[i]->Draw("same HIST E");
        }
    }

    canvas->cd(2);

    for (int i = 0; i < histPy.size(); ++i){
        
        histPy[i]->SetLineColor(i + 2 );
        histPy[i]->SetLineWidth(2);
        histPy[i]->SetFillColorAlpha(i + 2, 0.3);
        histPy[i]->SetMarkerStyle(20+i);
        histPy[i]->SetMarkerColor(i + 2);
        if(i == 0){
            histPy[i]->Draw("HIST E");
        }else{
            histPy[i]->Draw("same HIST E");
        }
    }
    canvas->cd(3);

    TLegend *leg = new TLegend(0.55, 0.6, 0.85,0.8);
    leg->SetTextSize(0.03);
    leg->SetFillColorAlpha(kWhite, 0.);


    for (int i = 0; i < histX.size(); ++i){

        histX[i]->SetLineColor(i+2);
        histX[i]->SetLineWidth(2);
        histX[i]->SetMarkerColor(i+2);
        histX[i]->SetFillColorAlpha(i+2, 0.1);
        histX[i]->SetMarkerStyle(20 + i);
        histX[i]->SetMarkerSize(1);
        if(i == 0){
            histX[i]->Draw("HIST E");
        }else{
            histX[i]->Draw("same HIST E");
        }
        leg->AddEntry(histY[i], histX[i]->GetTitle()/* descriptions[i]*/, "lep");
    }
    leg->Draw("same");

    canvas->cd(4);

    for (int i = 0; i < histY.size(); ++i){

        histY[i]->SetLineColor(i+2);
        histY[i]->SetLineWidth(2);
        histY[i]->SetMarkerColor(i+2);
        histY[i]->SetFillColorAlpha(i+2, 0.1);
        histY[i]->SetMarkerStyle(20 + i);
        histY[i]->SetMarkerSize(1);
        if(i == 0){
            histY[i]->Draw("HIST E");
        }else{
            histY[i]->Draw("same HIST E");
        }
    }


    canvas->Update();
    canvas->SaveAs( parentDir + TString("/") + fileName[fields]);
    canvas->Close();

    file->Close();
    delete file;

    cout << "All plots created using ROOT." << endl;
    return 0;
}