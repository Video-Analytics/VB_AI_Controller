
import matplotlib.pyplot as plt
import numpy as np
import wave
import os
import shutil
import time
import subprocess
import sys
import json
from pydub import AudioSegment, effects
from ffmpy import FFmpeg, FFprobe
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from utils import *
import warnings
warnings.filterwarnings("ignore")



def make_audio_vars(arg_parser):
    global audio_values
    global save_wav_graph
    global compress

    global values_output_dir
    global graph_output_dir
    global input_dir
    global existing

    audio_values = arg_parser.audio_values
    save_wav_graph = arg_parser.save_wav_graph
    compress = arg_parser.compress
    existing=False

    values_output_dir = arg_parser.values_output_dir
    graph_output_dir = arg_parser.graph_output_dir
    input_dir = arg_parser.audio_input_dir


    #correct and handle directory paths
    correct_dirs = resolve_directories(input_dir, values_output_dir, graph_output_dir)

    #rename dir to usable paths
    input_dir = correct_dirs[0]            
    values_output_dir = correct_dirs[1]               
    graph_output_dir = correct_dirs[2]



def compress_sound(input_file):
    print('\nCompressing Audio...')
    rawsound = AudioSegment.from_file(input_file, "wav")
    compressedsound = effects.compress_dynamic_range(rawsound)
    compressedsound.export(input_file, format='wav')
    print('Audio Compressed\n')


# Handles events of a directory
class sound_Handler(FileSystemEventHandler):
    @staticmethod
    def on_any_event(event):

        if event.is_directory:
            print(event.src_path)
            return None

        elif event.event_type == 'created':
            event_path = event.src_path
            print('Received input event: ', event_path)
            auto_sound_grapher(event_path)

# Watches for events
class sound_Watcher():
    def __init__(self, watched_dir):
        self.observer = Observer()
        self.watched_dir = watched_dir

    def run(self):
        event_handler = sound_Handler()
        self.observer.schedule(event_handler, self.watched_dir, recursive=True)
        self.observer.start()
        try:
            while True:
                time.sleep(5)
        except:
            self.observer.stop()
            print("Stopped!")

        self.observer.join()


def audio_value_clean(x, y):
    x = x.tolist()
    clean_dict = {'count' : 0}
    data = {}
    count = 0
    for i in range(0,len(x)):
        clean_dict.update({'count': count})
        temp = {}
        data[x[i]] = y[i]
        count+=1
        data.update(temp)
    nest = {'data' : data}
    clean_dict.update(nest)
    return clean_dict





def save_audio_stuff(time, amplitude, graph, filename, audio_file):
    if save_wav_graph:
       graph_name = graph_output_dir + audio_file.replace('.wav', '_graph.png')
       plt.savefig(graph_name, bbox_inches = "tight")
       print('\nSaved audio graph to: ', graph_name)
    if audio_values:
       audio_values_json = values_output_dir + audio_file.replace('.wav', '_values.json') 
       audio_value_dict = {'x-axis': time, 'y-axis' : amplitude}
       cleaned = audio_value_clean(time, amplitude)
       with open(audio_values_json, 'w') as fp:
            json.dump(cleaned, fp, indent=4, cls=NpEncoder)
       print('\nAudio values saved to: {}\n'.format(audio_values_json))
    print('Removing input.')
    os.remove(audio_file)
    os.remove(input_dir + filename)
    os.remove(temp_dir + filename)
    print('\nReady for next input.\n')



def check_existing_graph():
    global existing
    while os.listdir(input_dir)!=[]:
        existing = True
        filelist = os.listdir(input_dir)
        for file in filelist:
            print('Existing file found: ' + file)
            file = input_dir + file
            auto_sound_grapher(file)






def get_audio_values(input):
    with wave.open(input, 'r') as wavfile:
        signal = wavfile.readframes(-1)
        signal = np.fromstring(signal, 'Int16')

    channels = [[] for channel in range (wavfile.getnchannels())]
    for index, datum in enumerate(signal):
        channels[index%len(channels)].append(datum)
    fs = wavfile.getframerate()
    timing = np.linspace(0, len(signal)/len(channels)/fs, num=int(len(signal)/len(channels)))
    print("progress: {}/100".format(50))

    plt.title('Graph of ' + input)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.tight_layout()

    for channel in channels:
        wav_graph = plt.plot(timing, channel)
    channels = channels[0]
    return timing, channels, wav_graph



def auto_sound_grapher(input):
    global temp_dir
    filename = input.replace(input_dir,"")
    temp_dir = "./sound/temp/"
    prog_check=0
    up_to = 5
    for i in range(up_to-prog_check):
        print("progress: {}/100".format(prog_check + i))
        sys.stdout.flush()
        time.sleep(0.5)
    local_copy = copy_to_local(input_dir, filename, temp_dir)
    has_audio = resolve_audio(local_copy, input)
    if has_audio==False:
        print("progress: {}/100".format(100))
        return
    wav_name, json_name = make_wav_json_name(filename)
    print("progress: {}/100".format(25))
    convert_to_wav(local_copy, wav_name)
    if compress:
        compress_sound(wav_name)

    #run function to get values
    timing, channels, wav_graph = get_audio_values(wav_name)

    print("progress: {}/100".format(70))

    save_audio_stuff(timing, channels, wav_graph, filename, wav_name)
    print("progress: {}/100".format(100))


def run_autosound(args):
    make_audio_vars(args)
    check_existing_graph()
    w = sound_Watcher(input_dir) 
    if existing!=True:
        print('Ready for sound input.\n')
    w.run()

