
#include <iostream>
#include "TFile.h"
#include "TTree.h"
#include "TH1D.h"

using namespace std;


TFile *file;
TTree *tree;

Int_t offsets[4] = {10, 100,500, 1000};
Int_t colors[4] = {kBlue, kRed, kMagenta, kGreen};

void plotEmittance();
void plotXYRMS();

void CreatePlots(const TString inputFile){

	file = new TFile(inputFile, "r");

	if(!file){
		cout << "Did not get file." << endl;
		return;
	}

	tree = (TTree*)file->Get("outputTree");

	if(!tree){
		cout << "Did not get tree." << endl;
		return;
	}

	plotEmittance();
	//plotXYRMS();

}

void plotEmittance(){
	gStyle->SetOptStat(0);
	TCanvas *c = new TCanvas("c", "c", 800, 600);
	//c->Divide(2,1);

	//c->cd(1);
	vector<TH1D*> hist;

	TLegend *leg = new TLegend(0.6,0.6,0.85,0.85);
	leg->SetTextSize(0.04);
	leg->SetFillColor(kWhite);
	leg->SetTextColor(kBlack);

	for (int i = 0; i < 4; ++i){
		cout << "condition: " << TString::Format("offsetInMicrons == %d", offsets[i]) << endl;
		tree->Draw(TString::Format("outEmitNormX>>hist%d(50, 1.173, 1.18)", i), TString::Format("offsetInMicrons == %d", offsets[i]));
		
		TH1D* hist_temp = (TH1D*)gDirectory->Get(TString::Format("hist%d",i) );
		if(!hist_temp || hist_temp->GetEntries() == 0){
			cout << "Did not get histogram, leaving." << endl;
			break;
		}
		cout << "Number of entries: " << hist_temp->GetEntries() << endl;
		hist.push_back( hist_temp );
		hist[i]->SetMarkerStyle(20+i);
		hist[i]->SetMarkerColor(colors[i]);
		hist[i]->SetMarkerSize(2);
		hist[i]->SetLineColor(colors[i]);
		hist[i]->GetXaxis()->SetTitle("#epsilon_{x,norm} [#pi mm mrad]");
		hist[i]->GetYaxis()->SetTitle("counts");
		hist[i]->SetTitle("Normalized emittance X");

		leg->AddEntry(hist[i],TString::Format("offset: %d #mu m", offsets[i]), "lep");

	}

	hist[0]->Draw("HIST");
	hist[0]->Draw("same P");
	hist[1]->Draw("same HIST");
	hist[1]->Draw("same P");
	hist[2]->Draw("same HIST");
	hist[2]->Draw("same P");
	hist[3]->Draw("same HIST");
	hist[3]->Draw("same P");


	leg->Draw("same");
	/*

	c->cd(2);

	vector<TH1D*> hist2;

	TLegend *leg2 = new TLegend(0.7,0.7,0.9,0.9);
	leg2->SetTextSize(0.04);
	leg2->SetFillColor(kWhite);
	leg2->SetTextColor(kBlack);

	for (int i = 0; i < sizeof(offsets); ++i){
		tree->Draw("outEmitNormY>>hist(50, 1.173, 1.18)", TString::Format("offsetInMicrons == %d", offsets[i]));
		hist2.push_back( (TH1D*)file->Get("hist") );
		hist2[i]->SetMarkerStyle(i);
		hist2[i]->SetMarkerColor(i+2);
		hist2[i]->SetMarkerSize(2);
		hist2[i]->SetLineColor(i+2);
		hist2[i]->GetXaxis()->SetTitle("#epsilon_{y,norm} [#pi mm mrad]");
		hist2[i]->GetYaxis()->SetTitle("counts");
		hist2[i]->SetTitle("Normalized emittance Y");

		hist2[i]->Draw("same EP");

		leg2->AddEntry(hist2[i],TString::Format("offset: %d #mu m", offsets[i]), "lep");

	}
	leg2->Draw("same");
	*/
	c->SaveAs("emittancePlot.pdf");
	c->Close();

}


void plotXYRMS(){

	TCanvas *c = new TCanvas("c", "c", 600, 800);
	c->Divide(1,2);

	c->cd(1);
	vector<TH1D*> hist;

	TLegend *leg = new TLegend(0.7,0.7,0.9,0.9);
	leg->SetTextSize(0.04);
	leg->SetFillColor(kWhite);
	leg->SetTextColor(kBlack);

	for (int i = 0; i < sizeof(offsets); ++i){
		tree->Draw("outRMSX>>hist(50, 0.005, 0.02)", TString::Format("offsetInMicrons == %d", offsets[i]));
		hist.push_back( (TH1D*)file->Get("hist") );
		hist[i]->SetMarkerStyle(i);
		hist[i]->SetMarkerColor(i+1);
		hist[i]->SetMarkerSize(2);
		hist[i]->SetLineColor(i+1);
		hist[i]->GetXaxis()->SetTitle("RMS X [mm]");
		hist[i]->GetYaxis()->SetTitle("counts");
		hist[i]->SetTitle("Root Mean Square X");

		hist[i]->Draw("same EP");

		leg->AddEntry(hist[i],TString::Format("offset: %d #mu m", offsets[i]), "lep");

	}
	leg->Draw("same");
	//----------------------------------------------------------------------------------------
	c->cd(2);

	vector<TH1D*> hist2;

	TLegend *leg2 = new TLegend(0.7,0.7,0.9,0.9);
	leg2->SetTextSize(0.04);
	leg2->SetFillColor(kWhite);
	leg2->SetTextColor(kBlack);

	for (int i = 0; i < sizeof(offsets); ++i){
		tree->Draw("outRMSY>>hist(50, 0.005, 0.02)", TString::Format("offsetInMicrons == %d", offsets[i]));
		hist2.push_back( (TH1D*)file->Get("hist") );
		hist2[i]->SetMarkerStyle(i);
		hist2[i]->SetMarkerColor(i+2);
		hist2[i]->SetMarkerSize(2);
		hist2[i]->SetLineColor(i+2);
		hist2[i]->GetXaxis()->SetTitle("RMS Y [mm]");
		hist2[i]->GetYaxis()->SetTitle("counts");
		hist2[i]->SetTitle("Root mean square Y");

		hist2[i]->Draw("same EP");

		leg2->AddEntry(hist2[i],TString::Format("offset: %d #mu m", offsets[i]), "lep");

	}
	leg2->Draw("same");

	c->SaveAs("RMSPlot.pdf");
	c->Close();

}
