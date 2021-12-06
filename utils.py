import os
import shutil
from ffmpy import FFmpeg, FFprobe
import subprocess
import numpy as np
import time
import json

import torch
import tempfile
import warnings
import torchaudio
from typing import List
from itertools import groupby



def copy_to_local(input_dir, filename, temp_dir):
    tobemoved = input_dir + filename
    temp_file = temp_dir+filename
    newtemp = temp_file.replace(" ", "")
    #print("Temp File", temp_file)
    if not os.path.isfile(newtemp):
      shutil.copy2(tobemoved, temp_file)
      os.rename(temp_file, newtemp)
    return newtemp


#add slashes to directories to prevent path errors
def resolve_directories(*args):
    corrected = []
    for dir in args:
        if (dir[-1]=='/' or dir[-1]=='\\'):
            corrected.append(dir)
        elif '/' in dir:
            dir = dir + '/'
            corrected.append(dir)
        elif '\\' in dir:
            dir = dir + '\\'
            corrected.append(dir)
    return corrected



def resolve_filenames(*args):
    corrected = []
    for file in args:
        print('Before: ',  file)
        file = file + '\\'
        corrected.append(file)
        print('After: ', file)
    return corrected


def convert_to_wav(input, output):
    try:
        ## Use ffmpy to convert to correct .wav format
        convert_process = FFmpeg(inputs={input : None},
            outputs={output : '-loglevel warning -ac 1 -ar 16000 -y -f wav'})
        cmd = subprocess.Popen(["runas", "/noprofile", "/user:Administrator", "|", "echo", "Y", "|", "choco", "install", "dropbox"],stdin=subprocess.PIPE)
        cmd.stdin.write(b'vbpass12#')
        cmd.communicate()
        convert_process.run()
        print('\nFFmpeg executed in python')
    except:
        ## incase that fails
       cmd = subprocess.Popen(["runas", "/noprofile", "/user:Administrator", "|", "echo", "Y", "|", "choco", "install", "dropbox"],stdin=subprocess.PIPE)
       cmd.stdin.write(b'vbpass12#')
       cmd.communicate()
       convert_cmd = 'ffmpeg -i {} -loglevel warning -ac 1 -ar 16000 -f wav {}'.format(input, output)
       subprocess.call(convert_cmd, shell=True, stdout=subprocess.PIPE)

def resolve_audio(input, event_path):
    check_audio_process = FFprobe(inputs={input: '-show_streams -select_streams a -of json -loglevel quiet'})
    try:
        check_out = check_audio_process.run(stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        audio_info = check_out[0]
        audio_dict = json.loads(audio_info.decode('utf-8'))
        print('Audio channels in file:', audio_dict['streams'][0]['channels'])
        time.sleep(0.5)
        print('Sample Rate of file:', audio_dict['streams'][0]['sample_rate'])
        time.sleep(0.5)
        print('Audio Codec of file:', audio_dict['streams'][0]['codec_name'])
        time.sleep(0.5)
        check_out = audio_dict['streams'][0]['channels']
        if check_out == 0:
            print("\nNo audio in given input. Removing file out of input directory.")
            os.remove(event_path)
            print("progress: {}/100".format(100))
            print('Done.\n\nReady for next transciption.\n')
            print("progress: {}/100".format(0))
            os.remove(input)
            return False
    except Exception as e:
        print("\nNo audio in given input. Removing file out of input directory.")
        os.remove(event_path)
        print('Done.\n\nReady for next input.\n')
        print("progress: {}/100".format(0))
        os.remove(input)
        return False
    return True

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)
        
class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def make_names(filename):
    split_name = filename.split(".", 1)
    raw_name = split_name[0] #everything before the extension
    wav_name = raw_name + ".wav"
    json_name = raw_name + ".json"
    txt_name = raw_name + ".txt"
    return wav_name, json_name, txt_name








##Silero utils

torchaudio.set_audio_backend("soundfile")  # switch backend


def read_batch(audio_paths: List[str]):
    return [read_audio(audio_path)
            for audio_path
            in audio_paths]


def split_into_batches(lst: List[str],
                       batch_size: int = 10):
    return [lst[i:i + batch_size]
            for i in
            range(0, len(lst), batch_size)]


def read_audio(path: str,
               target_sr: int = 16000):
    assert torchaudio.get_audio_backend() == 'soundfile'
    wav, sr = torchaudio.load(path)

    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)

    if sr != target_sr:
        transform = torchaudio.transforms.Resample(orig_freq=sr,
                                                   new_freq=target_sr)
        wav = transform(wav)
        sr = target_sr

    assert sr == target_sr
    return wav.squeeze(0)


def prepare_model_input(batch: List[torch.Tensor],
                        device=torch.device('cpu')):
    max_seqlength = max(max([len(_) for _ in batch]), 12800)
    inputs = torch.zeros(len(batch), max_seqlength)
    for i, wav in enumerate(batch):
        inputs[i, :len(wav)].copy_(wav)
    inputs = inputs.to(device)
    return inputs


class Decoder():
    def __init__(self,
                 labels: List[str]):
        self.labels = labels
        self.blank_idx = self.labels.index('_')
        self.space_idx = self.labels.index(' ')

    def process(self,
                probs, wav_len, word_align):
        assert len(self.labels) == probs.shape[1]
        for_string = []
        argm = torch.argmax(probs, axis=1)
        align_list = [[]]
        for j, i in enumerate(argm):
            if i == self.labels.index('2'):
                try:
                    prev = for_string[-1]
                    for_string.append('$')
                    for_string.append(prev)
                    align_list[-1].append(j)
                    continue
                except:
                    for_string.append(' ')
                    warnings.warn('Token "2" detected a the beginning of sentence, omitting')
                    align_list.append([])
                    continue
            if i != self.blank_idx:
                for_string.append(self.labels[i])
                if i == self.space_idx:
                    align_list.append([])
                else:
                    align_list[-1].append(j)

        string = ''.join([x[0] for x in groupby(for_string)]).replace('$', '').strip()

        align_list = list(filter(lambda x: x, align_list))

        if align_list and wav_len and word_align:
            align_dicts = []
            linear_align_coeff = wav_len / len(argm)
            to_move = min(align_list[0][0], 1.5)
            for i, align_word in enumerate(align_list):
                if len(align_word) == 1:
                    align_word.append(align_word[0])
                align_word[0] = align_word[0] - to_move
                if i == (len(align_list) - 1):
                    to_move = min(1.5, len(argm) - i)
                    align_word[-1] = align_word[-1] + to_move
                else:
                    to_move = min(1.5, (align_list[i + 1][0] - align_word[-1]) / 2)
                    align_word[-1] = align_word[-1] + to_move

            for word, timing in zip(string.split(), align_list):
                align_dicts.append({'word': word,
                                    'start_ts': round(timing[0] * linear_align_coeff, 2),
                                    'end_ts': round(timing[-1] * linear_align_coeff, 2)})

            return string, align_dicts
        return string

    def __call__(self,
                 probs: torch.Tensor,
                 wav_len: float = 0,
                 word_align: bool = False):
        return self.process(probs, wav_len, word_align)


def init_jit_model(model_url: str,
                   device: torch.device = torch.device('cpu')):
    torch.set_grad_enabled(False)

    model_dir = os.path.join(os.path.dirname(__file__), "speech/models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, os.path.basename(model_url))

    if not os.path.isfile(model_path):
        torch.hub.download_url_to_file(model_url,
                                       model_path,
                                       progress=True)

    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    return model, Decoder(model.labels)






####### PROGRESS TIMER FOR REPETITION ####

from threading import Timer


class RepeatedTimer(object):
    def __init__(self, interval, function, *args, **kwargs):
        self._timer = None
        self.interval = interval
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.is_running = False
        self.start()

    def _run(self):
        self.is_running = False
        self.start()
        self.function(*self.args, **self.kwargs)

    def start(self):
        if not self.is_running:
            self._timer = Timer(self.interval, self._run)
            self._timer.start()
            self.is_running = True

    def stop(self):
        self._timer.cancel()
        self.is_running = False

def estimate_vid_progress(video_path, start_prog, end_prog):
    import cv2
    global prog_count
    global end_point
    end_point = end_prog
    prog_count = start_prog
    cap = cv2.VideoCapture(video_path)
    totalframecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    detect_rate = 9 #fps
    estimated_finish =  totalframecount/detect_rate
    print('\nEstimated inference time: {}s\n'.format(round(estimated_finish, 2) + 11))
    progress_chunks = ((end_prog - start_prog) / estimated_finish)
    return progress_chunks


def estimate_progress(audio, sample_rate, start_prog, end_prog):
    global prog_count
    global end_point
    end_point = end_prog
    prog_count = start_prog

    f = open("./speech/dependencies/rate.txt", "r")
    calc_rate = f.read()
    calc_rate = float(calc_rate)
    audio_length = len(audio) * (1 / sample_rate)

    estimated_finish = audio_length / calc_rate
    print('\nEstimated inference time: {}s\n'.format(round(estimated_finish, 2) + 11))
    progress_chunks = ((end_prog - start_prog) / estimated_finish)

    return progress_chunks


def print_model_progress(add_step):
    global prog_count
    prog_count = prog_count + add_step
    print("progress: {}/100".format(int(prog_count)))
    if prog_count > end_point:
        prog_count = end_point