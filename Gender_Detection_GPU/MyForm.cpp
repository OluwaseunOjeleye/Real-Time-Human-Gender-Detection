#include "MyForm.h"
using namespace System;
using namespace System::Windows::Forms;


[STAThread]
void Main(array<String^>^ args)
{
	Application::EnableVisualStyles();
	Application::SetCompatibleTextRenderingDefault(false);

	Face_Detection::MyForm form;
	Application::Run(%form);
}

