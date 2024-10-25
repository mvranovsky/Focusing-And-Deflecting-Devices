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


    vector<TH1D*> hist;
    TH1D *histPx = (TH1D*)file->Get("histPx");
    TH1D *histX = (TH1D*)file->Get("histX");
    TList *list = file->GetListOfKeys();

    TIter next(list);
    TKey *key;
    TObject *obj;
    while( (key = (TKey*)next()) ){
        obj = key->ReadObj();
        if(strcmp(obj->GetName(), "histPx") == 0 || strcmp(obj->GetName(), "histX") == 0){
            continue;
        }
        TH1D* h = (TH1D*)obj;
        hist.push_back(h);
    }

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


    TCanvas *canvas = new TCanvas("canvas", "canvas", 600, 800);
    gStyle->SetOptStat(0);
    canvas->Divide(1,2);
    canvas->cd(1);

    histPx->SetLineColor(kBlack);
    histPx->SetLineWidth(2);
    histPx->SetFillColorAlpha(kBlack, 0.3);
    histPx->SetMarkerStyle(20);
    histPx->SetMarkerColor(kBlack);
    histPx->SetTitle( histPx->GetTitle() );
    histPx->Draw("hist E");

    canvas->cd(2);

    histX->SetLineColor(kBlack);
    histX->SetLineWidth(2);
    histX->SetFillColorAlpha(kBlack, 0.3);
    histX->SetMarkerStyle(20);
    histX->SetMarkerColor(kBlack);
    histX->SetTitle( histX->GetTitle() );
    histX->Draw("hist E");


    canvas->Update();
    canvas->SaveAs( TString("quad1Study/") + parentDir + TString("/Control") + fileName[fields]);
    canvas->Close();


    TCanvas *c = new TCanvas("c", "c", 800, 600);
    gStyle->SetOptStat(0);
    c->cd();


    TLegend *leg = new TLegend(0.55, 0.6, 0.85,0.8);
    leg->SetTextSize(0.03);
    leg->SetFillColorAlpha(kWhite, 0.);


    for (int i = 0; i < hist.size(); ++i){
        hist[i]->SetLineColor(i+2);
        hist[i]->SetLineWidth(2);
        hist[i]->SetMarkerColor(i+2);
        hist[i]->SetFillColorAlpha(i+2, 0.1);
        hist[i]->SetMarkerStyle(20 + i);
        hist[i]->SetMarkerSize(2);
        if(i == 0){
            hist[i]->Draw("HIST E");
        }else{
            hist[i]->Draw("same HIST E");
        }
        leg->AddEntry(hist[i], hist[i]->GetTitle(), "lep");
    }
    leg->Draw("same");

    c->Update();
    c->SaveAs( TString("quad1Study/") + parentDir + TString("/") + fileName[fields]);
    c->Close();

    file->Close();
    delete file;

    cout << "All plots created using ROOT." << endl;
    return 0;
}