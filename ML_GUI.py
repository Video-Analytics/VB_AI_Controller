#remove all warnings - comment out if unknown bugs
from __future__ import generators

import warnings
warnings.filterwarnings("ignore")

import sys, os
from gooey import Gooey, GooeyParser
from speech.speech_to_text import run_stt
from sound.sound_utils import run_autosound
from yolov4.detect import run_object_detection
from concat.concat import run_concat
from facenet.face_detect import run_face_detection
from paravision import face_detect, enroll

def get_mapped_drives():
    import win32net
    resume = 0
    mapped_drives = []
    while 1:
        (_drives, total, resume) = win32net.NetUseEnum(None, 0, resume)
        for drive in _drives:
            if drive['local']:
                mapped_drives.append(drive['local'])
        if not resume: break
    return mapped_drives


# customize watchdog through the following parameters [there are defaults]
@Gooey(optional_cols=2, program_name="VideoBank Analytics Control Panel", default_size=(825,900), navigation='TABBED', clear_before_run=True, image_dir='./external/gooey/gooey_images[notpartof]',
progress_regex=r"^progress: (?P<current>\d+)/(?P<total>\d+)$", progress_expr="current / total * 100", disable_progress_bar_animation=False, hide_progress_msg=True,

timing_options = {
        'show_time_remaining':True,
        'hide_time_remaining_on_complete':False,
                  },

menu=[{'name': 'File', 'items': [{
    'type': 'AboutDialog',
    'menuTitle': 'About',
    'name': 'VideoBank Analytics Control Panel',
    'developer':'Developed by VideoBank',
    'year': '2020',
    'version': '5.0'},
{
    'type': 'Link',
    'menuTitle': 'Visit Our Site',
    'url': 'http://www.videobankdigital.com'
}]}]
       )
def ml_gui():
    parser = GooeyParser(description='Control parameters of AI systems in one place. Use tools below to customize your AI.') #main parser
    subs = parser.add_subparsers(help='commands', dest='command') #created subparser parser to have multiple parsers for the tabs

    mapped_drives = get_mapped_drives()
    count = 0
    for drive in mapped_drives:
        mapped_drives[count] = mapped_drives[count] + '\\VideoBankSpeech_In'
        count += 1

    #Speech-to-Text
    stt_parser = subs.add_parser('AutoTranscribe', help='Transcribe video or audio files through an auto-transcribing agent')
    required_stt = stt_parser.add_argument_group("Required Parameters")
    required_stt.add_argument("--input_dir", default=r".\speech\input", help="Directory to be autotranscribed when files inserted", widget='DirChooser', metavar='Input Directory', required=True)
    required_stt.add_argument("--output_dir", default=r".\speech\output", help="Directory where transcripts will be dumped", widget='DirChooser',  metavar='Output Directory', required=True)
    required_stt.add_argument('--use_multiple', default=False,
                            help="Choose watch and run speech-to-text on multiple inputs", metavar='Use Multiple',
                            action="store_true", widget='CheckBox')
    required_stt.add_argument("--available_servers", help="Choose which mapped drives/servers to watch (automatically replaces 'In' with 'Out')",
                            metavar='Multiple server inputs', widget='Listbox', choices=mapped_drives, nargs='+')

    required_stt.add_argument("--speech_model", default='NeMo', help="Choose which model to use",  metavar='Model', widget='Dropdown', choices=['NeMo', 'Silero'], required=True)
    required_stt.add_argument("--wav_dir", default=r'.\speech\transcribed_audio',
                              help="Directory where atranscribed audio files will be stored", metavar='Audio Storage',
                              widget='DirChooser', required=True)
    required_stt.add_argument("--prev_input_dir", default=r".\speech\previous_inputs",
                              help="Directory where already transcribed files will be stored",
                              metavar='Previous Input Storage', widget='DirChooser', required=True)

    api_parser = stt_parser.add_argument_group("API Parameters")
    api_parser.add_argument('--use_api', default=False, help="Choose send files to speech-to-text API (api uses defaults)", metavar='Use API', action="store_true", widget='CheckBox')
    api_parser.add_argument("--api_endpoint", default="http://172.19.3.36:8080/speech", help="Endpoint address of speech-to-text API", action="store", metavar='API Endpoint')
    api_parser.add_argument("--api_servers", help="Choose which mapped drives/servers to watch and send to api", metavar='API server inputs', widget='Listbox', choices=mapped_drives, nargs='+')


    nemo = stt_parser.add_argument_group("NeMo Parameters")
    nemo.add_argument('--nemo_time', default=True,
                      help="Choose to output a json with time stamps attached to transcription", metavar='Time stamps',
                      action="store_true", widget='CheckBox')
    nemo.add_argument('--acro', default=True,
                      help="Choose have acronyms detected and capitalized (at least 3 letters)",
                      metavar='Capitalize Acronyms (e.g. UAV)', action="store_true", widget='CheckBox')
    nemo.add_argument('--remove_uhs', default=True,
                      help="Choose have transcribed output remove speech disfluency (e.g. uh) from final transcript",
                      metavar='Remove Speech Disfluency', action="store_true", widget='CheckBox')
    nemo.add_argument('--diarize', default=True,
                      help="Choose have transcribed output diarized - speaker seperation (i.e. who said what)",
                      metavar='Diarization', action="store_true", widget='CheckBox')
    nemo.add_argument('--to_esp', default=False,
                      help="Choose to translate text to spanish after transcription", action="store_true",
                      metavar="Translate to Spanish", widget="CheckBox")
    nemo.add_argument('--nemo_gpu', default=False,
                      help="Choose to utilize local gpu w/ CUDA (device 0)",
                      metavar='Enable GPU usage', action="store_true", widget='CheckBox')
    nemo.add_argument('--punc', default=False,
                      help="Choose have transcribed output with capitalization and punctuation",
                      metavar='Punctuation and Capitalization', action="store_true", widget='CheckBox')
    required_stt.add_argument('--spellcheck', default=False, help="Choose automatically apply spelling corrections transcription", metavar='Spell Check', action="store_true", widget='CheckBox')



    silero = stt_parser.add_argument_group("Silero Parameters")
    silero.add_argument('--device', default='cpu', help='Which device to use [only 1 gpu (GPU 0) and cpu supported]',  metavar='Computing Device', widget='Dropdown', choices=['cpu', 'cuda'])
    silero.add_argument('--lang', default='English', help='Speech-to-text for this chosen language',  metavar='Language', widget='Dropdown', choices=['English', 'Spanish', 'German'])
    silero.add_argument('--silero_time', default=False, help="Choose to output a json with time stamps attached to transcription", metavar='Time stamps', action="store_true", widget='CheckBox')
    silero.add_argument('--to_eng', default=False, help="Choose to translate Spanish transcription to English", metavar='Translate to English', action="store_true", widget='CheckBox')

    visualize_stt = stt_parser.add_argument_group("Visualizations Options")
    visualize_stt.add_argument('--audio_values', default=False, help="Choose to automatically output audio file values (uses 'Audio Storage' as input)", metavar='Return audio values',action="store_true", widget='CheckBox')
    visualize_stt.add_argument('--save_wav_graph', default=False, help="Choose to automatically save a graph of extracted audio", metavar='Save graph of audio', action="store_true", widget='CheckBox')
    visualize_stt.add_argument('--values_output_dir', default='./speech/audio_values/', help="Directories where audio values will be stored", metavar='Audio Values Output Directory', widget='DirChooser', required=False)
    visualize_stt.add_argument('--graph_output_dir', default='./speech/audio_values/', help="Directories where audio graphs will be stored", metavar='Audio Graph Output Directory', widget='DirChooser', required=False)

    dragon_stt = stt_parser.add_argument_group("Dragon Options")
    dragon_stt.add_argument('--dragon', default=False, help="Choose to automatically send extracted audio to a Dragon folder (needs to also be configured in Dragon)", metavar='Send to Dragon Folder', action="store_true", widget='CheckBox')
    dragon_stt.add_argument("--drag_dir", default=r'.\speech\dragon\in', help="Directory where audio will be sent for dragon transcription",  metavar='Dragon Directory', widget='DirChooser')
    dragon_stt.add_argument('--copy_drag', default=False, help="Choose to copy Dragon outputs (txt and wav) to a directory", metavar='Copy Dragon Output', action="store_true", widget='CheckBox')
    dragon_stt.add_argument("--copyfrom_dir", default=r'.\speech\dragon\out', help="Directory where Dragon output is (to copy from)",  metavar='Dragon Output Directory', widget='DirChooser')
    dragon_stt.add_argument("--copyto_dir", default=r'.\speech\dragon\\train', help="Directory where Dragon output will be copied",  metavar='Copy Dragon Directory', widget='DirChooser')

 
    #Wav Graph
    wav_parser = subs.add_parser('AutoSoundGraph', help='Visualize sound in a file')

    required_wav = wav_parser.add_argument_group("Required Parameters")
    required_wav.add_argument('--audio_values', default=False, help="Choose to automatically output audio file values (uses 'Audio Storage' as input)", metavar='Return audio values',action="store_true", widget='CheckBox')
    required_wav.add_argument('--save_wav_graph', default=False, help="Choose to automatically save a graph of extracted audio", metavar='Save graph of audio', action="store_true", widget='CheckBox')
    required_wav.add_argument('--compress', default=False, help="Choose to compress the sound of extracted audio", metavar='Compress sound', action="store_true", widget='CheckBox')

    required_wav.add_argument("--audio_input_dir", default='./sound/sound_in/', help = 'Path to input sound directory to be watched',  metavar='Sound Input Directoy', required=True, type=str, widget='DirChooser')

    required_wav.add_argument('--values_output_dir', default='./sound/sound_out/', help="Directories where audio values will be stored", metavar='Audio Values Output Directory', widget='DirChooser', required=False)
    required_wav.add_argument('--graph_output_dir', default='./sound/sound_out/', help="Directories where audio graphs will be stored", metavar='Audio Graph Output Directory', widget='DirChooser', required=False)


    #Object Detection (images)
    obj_parser = subs.add_parser('AutoObjectDetection', help='Run object detection on an image (YOLO 416 w/ COCO dataset)')

    required_obj = obj_parser.add_argument_group("Required Parameters")
    required_obj.add_argument('--source', type=str, default=r'./yolov4/inference/input', help = 'Path to input folder, "0" for webcam, or udp/rstp/http stream address', metavar = 'Input Source', required=True, widget='DirChooser')  # file/folder, 0 for webcam
    required_obj.add_argument('--output', type=str, default=r'./yolov4/inference/output', help='Path to folder where output will be stored',  metavar = 'Output Folder', required=True, widget='DirChooser')  # output folder
    required_obj.add_argument('--prev_in', type=str, default=r'./yolov4/inference/previous_input', help = 'Path to put previous inputs', metavar='Previous input folder', widget='DirChooser')
    required_obj.add_argument('--prev_out', type=str, default=r'./yolov4/inference/previous_output', help = 'Path to put previous outputs', metavar='Previous output folder', widget='DirChooser')


    file_obj = obj_parser.add_argument_group("Filepaths")
    file_obj.add_argument('--weights', nargs='+', type=str, default=r'./yolov4/weights/yolov4-pacsp-x.weights', help='model.pt path(s)', metavar='Weights file', widget='FileChooser')
    file_obj.add_argument('--cfg', type=str, default=r'./yolov4/cfg/yolov4-pacsp-x.cfg', help='*.cfg path', metavar='Model cfg file', widget='FileChooser')
    file_obj.add_argument('--names', type=str, default=r'./yolov4/names/coco.names', help='*.cfg path', metavar='Class names file', widget='FileChooser')

    tune_obj = obj_parser.add_argument_group("Tuneable Parameters")
    tune_obj.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold', metavar='Confidence threshold', widget='DecimalField')
    tune_obj.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS', metavar='IOU threshold', widget='DecimalField')
    tune_obj.add_argument('--img-size', type=int, default=640, help='inference size (pixels)', metavar='Size of images')
    tune_obj.add_argument('--device', default='cuda', help='Which device to use [only 1 gpu (cuda-enabled) or cpu supported]',  metavar='Computing Device', widget='Dropdown', choices=['cuda', 'cpu'])
    tune_obj.add_argument('--view-img', action='store_true', help='View object detected frames',  metavar='Display Results', widget='CheckBox')
    tune_obj.add_argument('--save-txt', action='store_true', help='Choose to save results to *.txt', metavar='Save results as .txt', widget='CheckBox')
    tune_obj.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS', metavar='Class agnostic NMS', widget='CheckBox')

    # AutoConcatenation
    cat_parser = subs.add_parser('AutoConcatenation', help='Visualize sound in a file')

    required_cat = cat_parser.add_argument_group("Required Parameters")
    required_cat.add_argument("--input_dir", default="./concat/input/",
                              help="Directory to be autoconcatenated when files inserted", widget='DirChooser',
                              metavar='Input Directory', required=True)
    required_cat.add_argument("--output_dir", default="./concat/output/",
                              help="Directory where concatenated videos will be dumped", widget='DirChooser',
                              metavar='Output Directory', required=True)

    
    # FaceDetection
    face_det_parser = subs.add_parser('AutoFaceDetection', help='Detect faces in an image or video')

    required_face_det = face_det_parser.add_argument_group("Required Parameters")
    required_face_det.add_argument("--input_dir", default="./facenet/input/",
                              help="Path in input folder", widget='DirChooser',
                              metavar='Input Directory', required=True)
    required_face_det.add_argument("--output_dir", default="./facenet/output/",
                              help="Path to folder where output will be stored", widget='DirChooser',
                              metavar='Output Directory', required=True)
    required_face_det.add_argument('--face_gpu', default=False, help="Choose to utilize local gpu w/ CUDA (device 0)",
                               metavar='Enable GPU usage', action="store_true", widget='CheckBox')
    required_face_det.add_argument("--draw_boxes", action='store_true', help='Create output with overlayed boxes on faces',
                               metavar='Draw boxes on faces', widget='CheckBox')
    
    
    paravision_dbs = os.listdir('./paravision/dbs/')
    server_choices =[]
    restore_choices = []
    for db in paravision_dbs:
        if db.endswith('.sqlite'):
            server_choices.append(db)
        else:
            restore_choices.append(db)

    # (paravision) Face Recognition
    face_rec_parser = subs.add_parser('AutoFaceRecognition', help='Recognize faces in an image or video')

    required_face_rec = face_rec_parser.add_argument_group("Required Parameters")
    required_face_rec.add_argument("--input_dir", default="./paravision/detect/input/",
                               help="Path in input folder", widget='DirChooser',
                               metavar='Input Directory', required=True)
    required_face_rec.add_argument("--output_dir", default="./paravision/detect/output/",
                               help="Path to folder where output will be stored", widget='DirChooser',
                               metavar='Output Directory', required=True)
    required_face_rec.add_argument("--server_db",
                               help="Choose server database with people embeddings", action="store",
                               metavar='Database/Server Name', required=True, widget='Dropdown', choices=server_choices)
    required_face_rec.add_argument("--confidence_thres", default=0.85,
                               help="Threshold at which faces will be accepted as a match", widget='DecimalField',
                               metavar='Confidence Threshold')


    # (paravision) Face Rec Enrollment
    enroll_parser = subs.add_parser('AutoFaceEnrollment', help='Enroll faces for AutoFaceRecognition')

    required_enroll = enroll_parser.add_argument_group("Required Parameters")
    required_enroll.add_argument("--input_dir", default="./paravision/enroll/",
                               help="Path in input folder", widget='DirChooser',
                               metavar='Input Directory', required=True)
    required_enroll.add_argument("--db_name",
                               help="Name of server/db for faces", action="store",
                               metavar='Database/Server Name', widget='Dropdown', choices=server_choices,
                               default=None)
    
    required_enroll.add_argument('--add_facedb', action='store_true', help='Choose to add database with specified name', 
                                 metavar='Add database', widget='CheckBox')
    required_enroll.add_argument("--add_db_name",
                               help="Name of server/db to be added", action="store",
                               metavar='Added Database/Server Name')
    
    required_enroll.add_argument('--restore_facedb', action='store_true', help='Choose to re-enroll previously used database', 
                                 metavar='Restore database', widget='CheckBox')
    required_enroll.add_argument("--restore_name",
                               help="Names of possible dbs to restore", action="store",
                               metavar='Restore Database/Server Name', widget='Dropdown', choices=restore_choices,
                               default=None)
    
    

    args = parser.parse_args()

    try:
        if sys.argv[1]=='AutoTranscribe':
           run_stt(args)
  
        elif sys.argv[1]=='AutoSoundGraph':
           run_autosound(args)

        elif sys.argv[1]=='AutoObjectDetection':
           run_object_detection(args)

        elif sys.argv[1] == 'AutoConcatenation':
           run_concat(args)
        
        elif sys.argv[1] == 'AutoFaceDetection':
            run_face_detection(args)
            
        elif sys.argv[1] == 'AutoFaceRecognition':
            face_detect.run_face_rec(args)

        elif sys.argv[1] == 'AutoFaceEnrollment':
            enroll.run_enrollment(args)
            
    except(RuntimeError):
           print(RuntimeError)


if __name__ == '__main__':
    ml_gui()
