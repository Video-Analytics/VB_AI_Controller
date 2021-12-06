# VB_AI_Controller
 GUI for python and machine learning tools

Installation: 

0) pre-requisites: Anaconda (for now), ffmpeg, internet (for a little)

1) copy all files, open command prompt and navigate to VB_AI_Controller main directory [cd to/the/VB_AI_Controller]

2) create new conda env called 'vbank'(if not already existant) [conda create -n vbank python=3.8], then activate it [conda activate vbank]

*3) install git and pywin32 [conda install git pywin32] -----> now moved into the condalist

4) install conda (+git and pywin32) pytorch [conda install --file condalist*.txt -c pytorch]  *-gpu for machines with a gpu available, *-cpu for those without

5) install all other requirements [pip install -r piplist.txt]

6) {optional} for contextualspellcheck [python -m spacy download en_core_web_sm]

7) {optional} for look and feel - where conda envs are stored in C: (either in .conda or Anaconda3 --> envs --> vbank --> Lib --> site-packages --> gooey / kaldi_io) go into vbank's gooey module any copy folders from gooey/kaldi_io folder in VB_AI_Controller to replace default ones

8) with no internet connection, put the .cache folder under "C:Users/{current_user}/" for NeMo and Silero models

9) RUN && adjust accordingly with errors...