
#include <iostream>
#include "TFile.h"
#include "TTree.h"
#include "TH1D.h"

using namespace std;


TFile *file;
TTree *tree;

// define which offsets to plot, more offsets than this is not recommended because they overlap too much
Int_t offsets[4] = {10, 100,500, 1000};
Int_t colors[4] = {kBlue, kRed, kMagenta, kGreen};

void plotEmittance(int s);
void plotRMS(int s);

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

	plotEmittance(1);
	plotEmittance(2);

	plotRMS(1);
	plotRMS(2);

}

void plotEmittance(int s){
	gStyle->SetOptStat(0);
	TCanvas *c = new TCanvas("c", "c", 800, 600);

	c->SetLogy();
	vector<TH1D*> hist;

	// create legend
	TLegend *leg = new TLegend(0.6,0.6,0.85,0.85);
	leg->SetTextSize(0.04);
	leg->SetFillColor(kWhite);
	leg->SetTextColor(kBlack);

	for (int i = 0; i < 4; ++i){
		cout << "condition: " << TString::Format("offsetInMicrons == %d", offsets[i]) << endl;
		// load the hists from the tree for different offsets
		if(s == 1){
			tree->Draw(TString::Format("outEmitNormX>>hist%d(30, 1.173, 1.2)", i), TString::Format("offsetInMicrons == %d", offsets[i]));
		}else{
			tree->Draw(TString::Format("outEmitNormY>>hist%d(30, 1.173, 1.2)", i), TString::Format("offsetInMicrons == %d", offsets[i]));
		}
		
		TH1D* hist_temp = (TH1D*)gDirectory->Get(TString::Format("hist%d",i) );
		if(!hist_temp || hist_temp->GetEntries() == 0){
			cout << "Did not get histogram, leaving." << endl;
			break;
		}
		cout << "Number of entries: " << hist_temp->GetEntries() << endl;
		// style the histograms 
		hist.push_back( hist_temp );
		hist[i]->SetMarkerStyle(20+i);
		hist[i]->SetMarkerColor(colors[i]);
		hist[i]->SetMarkerSize(2);
		hist[i]->SetLineColor(colors[i]);
		hist[i]->SetFillColorAlpha(colors[i], 0.1);
		hist[i]->GetYaxis()->SetRangeUser(1, 1000);
		if(s==1){
			hist[i]->GetXaxis()->SetTitle("#epsilon_{x,norm} [#pi mm mrad]");
			hist[i]->SetTitle("Normalized emittance X");
		}else{
			hist[i]->GetXaxis()->SetTitle("#epsilon_{y,norm} [#pi mm mrad]");
			hist[i]->SetTitle("Normalized emittance Y");
		}
		hist[i]->GetYaxis()->SetTitle("counts");

		leg->AddEntry(hist[i],TString::Format("offset: %d #mu m", offsets[i]), "lep");

	}
	// draw histograms
	hist[0]->Draw("HIST");
	hist[0]->Draw("same P");
	hist[1]->Draw("same HIST");
	hist[1]->Draw("same P");
	hist[2]->Draw("same HIST");
	hist[2]->Draw("same P");
	hist[3]->Draw("same HIST");
	hist[3]->Draw("same P");


	leg->Draw("same");

	// save 
	if(s == 1){
		c->SaveAs("emittancePlotX.pdf");
	}else{
		c->SaveAs("emittancePlotY.pdf");
	}
	c->Close();

}


void plotRMS(int s){
	gStyle->SetOptStat(0);
	TCanvas *c = new TCanvas("c", "c", 800, 600);
	//c->SetLogy();
	vector<TH1D*> hist;

	TLegend *leg = new TLegend(0.6,0.6,0.85,0.85);
	leg->SetTextSize(0.04);
	leg->SetFillColor(kWhite);
	leg->SetTextColor(kBlack);

	for (int i = 0; i < 4; ++i){
		cout << "condition: " << TString::Format("offsetInMicrons == %d", offsets[i]) << endl;
		if(s == 1){
			tree->Draw(TString::Format("outRMSX>>hist%d(30, 0.006, 0.015)", i), TString::Format("offsetInMicrons == %d", offsets[i]));
		}else{
			tree->Draw(TString::Format("outRMSY>>hist%d(30, 0.00, 0.015)", i), TString::Format("offsetInMicrons == %d", offsets[i]));
		}
		
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
		hist[i]->SetFillColorAlpha(colors[i], 0.1);
		hist[i]->GetYaxis()->SetRangeUser(1, 1000);
		if(s==1){
			hist[i]->GetXaxis()->SetTitle("RMS_{x} [mm]");
			hist[i]->SetTitle("Normalized emittance X");
		}else{
			hist[i]->GetXaxis()->SetTitle("RMS_{y} [mm]");
			hist[i]->SetTitle("Normalized emittance Y");
		}
		hist[i]->GetYaxis()->SetTitle("counts");

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

	if(s == 1){
		c->SaveAs("RMSPlotX.pdf");
	}else{
		c->SaveAs("RMSPlotY.pdf");
	}
	c->Close();

}
