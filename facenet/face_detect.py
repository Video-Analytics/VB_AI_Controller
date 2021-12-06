import os, json, gc, time, sys
import numpy as np
import mmcv, cv2
from PIL import Image, ImageDraw
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import torch
import filetype
from utils import *
from facenet_pytorch import MTCNN

def make_face_vars(arg_parser):
    global input_dir 
    global output_dir
    global face_gpu
    global draw_boxes
    global existing

    input_dir = arg_parser.input_dir           
    output_dir = arg_parser.output_dir            
    face_gpu = arg_parser.face_gpu
    draw_boxes = arg_parser.draw_boxes
    existing=False

    #correct and handle directory paths
    correct_dirs = resolve_directories(input_dir, output_dir)

    #rename dir to usable paths
    input_dir = correct_dirs[0]            
    output_dir = correct_dirs[1]
    

# Handles events of a directory
class face_Handler(FileSystemEventHandler):
    @staticmethod
    def on_any_event(event):

        if event.is_directory:
            print(event.src_path)
            return None

        elif event.event_type == 'created':
            event_path = event.src_path
            print('Received input event: ', event_path)
            time.sleep(2)
            try:
                detect_faces(event_path)
            except:
                time.sleep(2)
                detect_faces(event_path)



# Watches for events
class face_Watcher():
    def __init__(self, watched_dir):
        self.observer = Observer()
        self.watched_dir = watched_dir

    def run(self):
        event_handler = face_Handler()
        self.observer.schedule(event_handler, self.watched_dir, recursive=True)
        self.observer.start()

        try:
            while True:
                time.sleep(5)
        except:
            self.observer.stop()
            print("Stopped!")

        self.observer.join()


def check_existing():
    global existing
    while os.listdir(input_dir)!=[]:
        existing = True
        filelist = os.listdir(input_dir)
        for file in filelist:
            print('Existing file found: ' + file)
            file = input_dir + file
            detect_faces(file)
            
def init_face():
    global mtcnn
    device = torch.device('cuda:0' if face_gpu else 'cpu')
    print('Running on device: {}'.format(device))
    mtcnn = MTCNN(keep_all=True, device=device)
    prog_check=0
    up_to = 10
    for i in range(up_to-prog_check):
        print("progress: {}/100".format(prog_check + i))
        sys.stdout.flush()
        time.sleep(0.5)
    print('Model initialized.\n')

# IMAGE FACE DETECTION
def image_face_detection(input):
    #detect faces in images
    img = Image.open(input)
    boxes, _ = mtcnn.detect(img)
    
    
    print("progress: {}/100".format(50))
    
    if boxes is None:
        print('No faces detected')
        return
    
    face_dict = {}
    face_dict['0'] = boxes
    
    filename = os.path.basename(input)
    ext = filename.split('.')
    
    #export faces as json
    json_name = ext[0] + '.json'
    print('Saving faces as:', json_name)
    json_name = output_dir + json_name
    dumped = json.dumps(face_dict, cls=NumpyEncoder, indent=4)

    with open(json_name, 'a') as f:
        f.write(dumped + '\n') 
        f.close()
    
    if draw_boxes:
        # Draw faces
        frame_draw = img.copy()
        draw = ImageDraw.Draw(frame_draw)
        for box in boxes:
            draw.rectangle(box.tolist(), outline=(255, 0, 0), width=6)
        frame_draw.save(output_dir + ext[0] + '.png', 'PNG')
        print('Saved PNG as: ' + ext[0] + '.png in ' + output_dir)
        
    print("progress: {}/100".format(100))
    img.close()
    del face_dict, boxes, img
    gc.collect()
    
def video_face_detection(input):
    video = mmcv.VideoReader(input)
    frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in video]
    print("progress: {}/100".format(40))
    print('Frames imported.\n')
    
    frames_tracked = []
    draw_frames = []
    for i, frame in enumerate(frames):
        print('\rTracking frame: {}/{}\n'.format(i + 1, len(frames)), end='')
    
        # Detect faces
        boxes, _ = mtcnn.detect(frame)
    
        # Add to frame list
        frames_tracked.append(boxes)
        
        # Draw faces       
        if draw_boxes:
            frame_draw = frame.copy()
            draw = ImageDraw.Draw(frame_draw)
            try:
                for box in boxes:
                    draw.rectangle(box.tolist(), outline=(255, 0, 0), width=6)
                # Add to frame list
                draw_frames.append(frame_draw.resize((640, 360), Image.BILINEAR))
            except:
                continue
        
    print("progress: {}/100".format(85))
    
    face_dict = {}
    for frame in enumerate(frames_tracked):
        # each frame is a tuple with (frame#, boxes)
        temp_facelist = []
        if(frame[1] is not None):
            #get all bbox in each frame
            for box in frame[1]:
                temp_facelist.append(box)
        #add it to faces per frame dictionary
        face_dict[int(frame[0])] = temp_facelist
        
    filename = os.path.basename(input)
    ext = filename.split('.')
    
    #export faces as json
    json_name = ext[0] + '.json'
    print('Saving faces as:', json_name)
    json_name = output_dir + json_name

    dumped = json.dumps(face_dict, cls=NumpyEncoder, indent=4)

    with open(json_name, 'a') as f:
        f.write(dumped + '\n') 
        f.close()
        
    if draw_boxes:
        print('Drawing face detections on video.')
        dim = draw_frames[0].size
        fourcc = cv2.VideoWriter_fourcc(*'FMP4')    
        video_tracked = cv2.VideoWriter(output_dir+ext[0]+'.mp4', fourcc, 25.0, dim)
        for frame in draw_frames:
            video_tracked.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
        video_tracked.release()
        del video_tracked
    
    print("progress: {}/100".format(85))    
    del face_dict, boxes, frames_tracked, draw_frames
    gc.collect()
            
def detect_faces(input):
    if ('video' in filetype.guess(input).mime) or ('image' in filetype.guess(input).mime):
        if 'video' in filetype.guess(input).mime:
            is_video = True
        else:
            is_video = False
    else:
        #invalid input
        print('Invalid input type - removing.\n')
        os.remove(input)
        print("progress: {}/100".format(100))
        time.sleep(1.5)
        print("progress: {}/100".format(0))
        return
    
    # VIDEO FACE DETECTION
    if is_video:
        video_face_detection(input)
    # IMAGE FACE DETECTION
    else:
        image_face_detection(input)
    
    os.remove(input)
    print('Ready for next input\n')
    time.sleep(1.5)
    print("progress: {}/100".format(0))


def run_face_detection(args):
    make_face_vars(args)
    init_face()
    check_existing()
    w = face_Watcher(input_dir) 
    if existing!=True:
        print('Ready for input.\n')
    w.run()