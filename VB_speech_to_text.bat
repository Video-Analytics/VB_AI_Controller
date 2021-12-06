@echo off
call C:\ProgramData\Anaconda3\Scripts\activate.bat
call conda activate vbank
cd "Desktop/VB_AI_Controller"
echo Running VideoBank Artificial Intelligence Controller
python.exe ML_GUI.py