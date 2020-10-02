#pragma once
#include <Windows.h>
#include <atlstr.h>
#include "src/darknet.h"

#include <msclr/marshal.h>
#include <string>

#include "opencv2/opencv.hpp"

using namespace System::Runtime::InteropServices;

namespace Face_Detection {
	using namespace System;
	using namespace System::ComponentModel;
	using namespace System::Collections;
	using namespace System::Windows::Forms;
	using namespace System::Data;
	using namespace System::Drawing;
	using namespace System::Collections::Generic;

	/// <summary>
	/// Summary for MyForm
	/// </summary>
	public ref class MyForm : public System::Windows::Forms::Form
	{
	public:
		MyForm(void)
		{
			InitializeComponent();
			//
			//TODO: Add the constructor code here
			//
		}

	protected:
		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		~MyForm()
		{
			if (components)
			{
				delete components;
			}
		}
	private: System::Windows::Forms::MenuStrip^  menuStrip1;
	protected:
	private: System::Windows::Forms::ToolStripMenuItem^  openToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^  detectionToolStripMenuItem;
	private: System::Windows::Forms::OpenFileDialog^  openFileDialog1;
	private: System::Windows::Forms::PictureBox^  pictureBox1;










	private: System::Windows::Forms::Label^  label7;


	private:
		/// <summary>
		/// Required designer variable.
		/// </summary>
		System::ComponentModel::Container ^components;


#pragma region Windows Form Designer generated code
		/// <summary>
		/// Required method for Designer support - do not modify
		/// the contents of this method with the code editor.
		/// </summary>
		void InitializeComponent(void)
		{
			this->menuStrip1 = (gcnew System::Windows::Forms::MenuStrip());
			this->openToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->detectionToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->openFileDialog1 = (gcnew System::Windows::Forms::OpenFileDialog());
			this->pictureBox1 = (gcnew System::Windows::Forms::PictureBox());
			this->label7 = (gcnew System::Windows::Forms::Label());
			this->menuStrip1->SuspendLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox1))->BeginInit();
			this->SuspendLayout();
			// 
			// menuStrip1
			// 
			this->menuStrip1->Items->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(1) { this->openToolStripMenuItem });
			this->menuStrip1->Location = System::Drawing::Point(0, 0);
			this->menuStrip1->Name = L"menuStrip1";
			this->menuStrip1->Size = System::Drawing::Size(792, 24);
			this->menuStrip1->TabIndex = 0;
			this->menuStrip1->Text = L"menuStrip1";
			// 
			// openToolStripMenuItem
			// 
			this->openToolStripMenuItem->DropDownItems->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(1) { this->detectionToolStripMenuItem });
			this->openToolStripMenuItem->Name = L"openToolStripMenuItem";
			this->openToolStripMenuItem->Size = System::Drawing::Size(59, 20);
			this->openToolStripMenuItem->Text = L"Process";
			// 
			// detectionToolStripMenuItem
			// 
			this->detectionToolStripMenuItem->Name = L"detectionToolStripMenuItem";
			this->detectionToolStripMenuItem->Size = System::Drawing::Size(125, 22);
			this->detectionToolStripMenuItem->Text = L"Detection";
			this->detectionToolStripMenuItem->Click += gcnew System::EventHandler(this, &MyForm::detectionToolStripMenuItem_Click);
			// 
			// openFileDialog1
			// 
			this->openFileDialog1->FileName = L"openFileDialog1";
			// 
			// pictureBox1
			// 
			this->pictureBox1->Location = System::Drawing::Point(1, 46);
			this->pictureBox1->Name = L"pictureBox1";
			this->pictureBox1->Size = System::Drawing::Size(512, 512);
			this->pictureBox1->SizeMode = System::Windows::Forms::PictureBoxSizeMode::AutoSize;
			this->pictureBox1->TabIndex = 1;
			this->pictureBox1->TabStop = false;
			// 
			// label7
			// 
			this->label7->AutoSize = true;
			this->label7->Location = System::Drawing::Point(12, 30);
			this->label7->Name = L"label7";
			this->label7->Size = System::Drawing::Size(60, 13);
			this->label7->TabIndex = 7;
			this->label7->Text = L"Test Image";
			// 
			// MyForm
			// 
			this->AutoScaleDimensions = System::Drawing::SizeF(6, 13);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->ClientSize = System::Drawing::Size(792, 654);
			this->Controls->Add(this->label7);
			this->Controls->Add(this->pictureBox1);
			this->Controls->Add(this->menuStrip1);
			this->MainMenuStrip = this->menuStrip1;
			this->Name = L"MyForm";
			this->Text = L"Face Detector";
			this->menuStrip1->ResumeLayout(false);
			this->menuStrip1->PerformLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox1))->EndInit();
			this->ResumeLayout(false);
			this->PerformLayout();

		}
#pragma endregion
		//member data
		box_label *Detected_Faces;

		String^ image_filename;
		char **names;
		int no_faces;

		void draw_detected_faces() {
			for (int i = 0; i < no_faces; i++) {
				//drawing each bound box to picture box
				Pen ^pen = gcnew Pen(Color::Red);
				Graphics ^g = pictureBox1->CreateGraphics();

				g->DrawRectangle(pen, Detected_Faces[i].left, Detected_Faces[i].top, Detected_Faces[i].w, Detected_Faces[i].h);
				g->DrawString(gcnew String(names[Detected_Faces[i].id]), gcnew System::Drawing::Font("Arial", 14), Brushes::Green, Detected_Faces[i].left, Detected_Faces[i].top);
			}
			
		}

		void draw_bound_box(image im, detection *dets, int num, int classes) {
			if (Detected_Faces != NULL) {
				delete Detected_Faces;
			}
			Detected_Faces = new box_label[num];

			no_faces = 0;

			for (int i = 0; i < num; ++i) {
				int _class = -1;
				for (int j = 0; j < classes; ++j) {
					if (dets[i].prob[j] > 0.5) {
						if (_class < 0) {
							_class = j;
						}
					}
				}
				if (_class >= 0) {
					box b = dets[i].bbox;

					Detected_Faces[no_faces].id = _class;
					Detected_Faces[no_faces].x = b.x;
					Detected_Faces[no_faces].y = b.y;
					Detected_Faces[no_faces].h = b.h*im.h;
					Detected_Faces[no_faces].w = b.w*im.w;
					Detected_Faces[no_faces].left = (b.x - b.w / 2)*im.w;
					Detected_Faces[no_faces].right = (b.x + b.w / 2)*im.w;
					Detected_Faces[no_faces].top = (b.y - b.h / 2)*im.h;
					Detected_Faces[no_faces].bottom = (b.y + b.h / 2)*im.h;

					if (Detected_Faces[no_faces].left < 0) Detected_Faces[no_faces].left = 0;
					if (Detected_Faces[no_faces].right > im.w-1) Detected_Faces[no_faces].right = im.w - 1;
					if (Detected_Faces[no_faces].top < 0) Detected_Faces[no_faces].top = 0;
					if (Detected_Faces[no_faces].bottom > im.h - 1) Detected_Faces[no_faces].bottom = im.h - 1;

					no_faces++;
				}
			}

			draw_detected_faces();
		}

		void DrawCVImage(System::Windows::Forms::Control^ control, cv::Mat& colorImage) {
			System::Drawing::Graphics^ graphics = control->CreateGraphics();
			System::IntPtr ptr(colorImage.ptr());
			System::Drawing::Bitmap^ b = gcnew System::Drawing::Bitmap(colorImage.cols, colorImage.rows, colorImage.step, System::Drawing::Imaging::PixelFormat::Format24bppRgb, ptr);
			System::Drawing::RectangleF rect(0, 0, control->Width, control->Height);
			graphics->DrawImage(b, rect);
			delete graphics;
		}

		image ipl_to_image(IplImage* src)
		{
			int h = src->height;
			int w = src->width;
			int c = src->nChannels;
			image im = make_image(w, h, c);
			unsigned char *data = (unsigned char *)src->imageData;
			int step = src->widthStep;
			int i, j, k;

			for (i = 0; i < h; ++i) {
				for (k = 0; k < c; ++k) {
					for (j = 0; j < w; ++j) {
						im.data[k*w*h + i * w + j] = data[i*step + j * c + k] / 255.;
					}
				}
			}
			return im;
		}

		void rgbgr_image(image im)
		{
			int i;
			for (i = 0; i < im.w*im.h; ++i) {
				float swap = im.data[i];
				im.data[i] = im.data[i + im.w*im.h * 2];
				im.data[i + im.w*im.h * 2] = swap;
			}
		}

		image mat_to_image(cv::Mat m)
		{
			IplImage ipl = m;
			image im = ipl_to_image(&ipl);
			rgbgr_image(im);
			return im;
		}

		void test_detector(char *cfgfile, char *weightfile, char *label_file, float thresh)
		{
			//loading Image
			char buff[256];
			char* input = buff;
			cv::VideoCapture capture(0);
			cv::Mat frame;

			//getting first frame and resizing GUI
			capture.read(frame);
			image im = mat_to_image(frame);
			pictureBox1->Width = im.w;
			pictureBox1->Height = im.h;
			pictureBox1->Refresh();

			//loading labels
			names = get_labels(label_file);

			//loading network
			network *net = load_network(cfgfile, weightfile, 0);
			set_batch_network(net, 1);
			srand(2222222);
			float hier_thresh = 0.5;
			float nms = .45;
			layer l = net->layers[net->n - 1];
			
			no_faces = 0;
			while (1) {
				capture.read(frame);
				DrawCVImage(pictureBox1, frame);
				draw_detected_faces();

				im = mat_to_image(frame);

				//detection
				double time = what_time_is_it_now();
				image sized = resize_image(im, net->w, net->h);

				float *X = sized.data;
				network_predict(net, X);

				int nboxes = 0;
				detection *dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes);

				if (nms) do_nms_sort(dets, nboxes, l.classes, nms);
				draw_bound_box(im, dets, nboxes, l.classes);
				//end of detection
			}
		}

		void run_detector(int n, char **s) {
			int check_mistakes = 0;
			char *cfg = s[0]; //ag yapisinin tutuldugu dosya
			char *weights = (n > 2) ? s[1] : 0; //agirlik degerleri
			if (weights)
				if (strlen(weights) > 0)
					if (weights[strlen(weights) - 1] == 0x0d) weights[strlen(weights) - 1] = 0;
			char *label_file = s[2];

			//textBox1->Text += "filename : " +gcnew String(filename) + "\r\n";
			float thresh = .5;
			test_detector(cfg, weights, label_file, thresh);
		}//run_detector

		void darknetMain(int argc, const char **argv) {
			//const char'dan char'a donusturmak icin strdup kullanildi
			char **c = new char*[argc];
			for (int i = 0; i < argc; i++) {
				c[i] = _strdup(argv[i]);
				//textBox1->Text += gcnew String(c[i]) +"\r\n"; 
			}
			run_detector(argc, c);
		}//darknetMain

	private: System::Void detectionToolStripMenuItem_Click(System::Object^  sender, System::EventArgs^  e) {
		int rows = 3;
		const char **c = new const char*[3];
		System::IO::DirectoryInfo^ info = gcnew System::IO::DirectoryInfo(System::IO::Directory::GetCurrentDirectory());
		msclr::interop::marshal_context ^ context = gcnew msclr::interop::marshal_context();

		MessageBoxA(0, "Load Network Architecture ", "CFG Dosyasi", MB_OK);
		if (openFileDialog1->ShowDialog() == System::Windows::Forms::DialogResult::OK) {
			c[0] = context->marshal_as<const char*>(openFileDialog1->FileName); //icra edilecek ag mimarisinin alinmasi
		}
		MessageBoxA(0, "Load the weight for pre-trained model", "WEIGHTS Dosyasi", MB_OK);
		if (openFileDialog1->ShowDialog() == System::Windows::Forms::DialogResult::OK) {
			c[1] = context->marshal_as<const char*>(openFileDialog1->FileName); //test sureci icin egitilmis agin agirlik degerlerinin tutuldugu dosya
		}
		MessageBoxA(0, "Load the label file", "Label Dosyasi", MB_OK);
		if (openFileDialog1->ShowDialog() == System::Windows::Forms::DialogResult::OK) {
			c[2] = context->marshal_as<const char*>(openFileDialog1->FileName); //test sureci icin egitilmis agin agirlik degerlerinin tutuldugu dosya
			darknetMain(rows, c);
		}
	}

	};
}


