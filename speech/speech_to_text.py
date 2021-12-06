#remove all warnings - comment out if unknown bugs
import warnings

from speech.dependencies.diarization import Diarizer

warnings.filterwarnings("ignore")

import logging
logging.getLogger('nemo_logger').setLevel(logging.ERROR)

import os, sys, gc, pathlib
import random
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import matplotlib.pyplot as plt
import torch

from speech.dependencies.speech_utils import *
from sound.sound_utils import get_audio_values
from utils import *

# Makes this so other classes and functions can use these variables
def make_stt_vars(arg_parser):
    global input_dir 
    global output_dir
    global wav_dir
    global prev_input_dir
    global existing

    global use_multiple
    global available_servers

    global speech_model
    global device
    global lang
    global silero_time
    global to_eng

    global spellcheck
    global to_esp
    global punc
    global nemo_time
    global acro
    global remove_uhs
    global nemo_gpu
    global diarize

    global audio_values
    global save_wav_graph
    global values_output_dir
    global graph_output_dir          

    global dragon
    global drag_dir
    global copy_drag
    global copyto_dir
    global copyfrom_dir

    global use_api
    global api_endpoint
    global api_servers

    input_dir = arg_parser.input_dir           
    output_dir = arg_parser.output_dir
    wav_dir = arg_parser.wav_dir                 
    prev_input_dir = arg_parser.prev_input_dir  
    existing = False                             

    use_multiple = arg_parser.use_multiple
    available_servers = arg_parser.available_servers

    speech_model = arg_parser.speech_model
    device = arg_parser.device
    lang = arg_parser.lang
    silero_time = arg_parser.silero_time
    to_eng = arg_parser.to_eng

    spellcheck = arg_parser.spellcheck
    to_esp = arg_parser.to_esp
    punc = arg_parser.punc
    nemo_time = arg_parser.nemo_time
    acro = arg_parser.acro
    remove_uhs = arg_parser.remove_uhs
    nemo_gpu = arg_parser.nemo_gpu
    diarize = arg_parser.diarize

    audio_values = arg_parser.audio_values
    save_wav_graph = arg_parser.save_wav_graph
    values_output_dir = arg_parser.values_output_dir
    graph_output_dir = arg_parser.graph_output_dir

    dragon = arg_parser.dragon
    drag_dir = arg_parser.drag_dir
    copy_drag = arg_parser.copy_drag
    copyto_dir = arg_parser.copyto_dir
    copyfrom_dir = arg_parser.copyfrom_dir

    use_api = arg_parser.use_api
    api_endpoint = arg_parser.api_endpoint
    api_servers = arg_parser.api_servers

    #correct and handle directory paths
    correct_dirs = resolve_directories(input_dir, output_dir, wav_dir, prev_input_dir, values_output_dir, graph_output_dir, drag_dir, copyto_dir, copyfrom_dir)

    #rename dir to usable paths
    input_dir = [correct_dirs[0]]
    output_dir = correct_dirs[1]               
    wav_dir = correct_dirs[2]                    
    prev_input_dir = correct_dirs[3]
    values_output_dir = correct_dirs[4]
    graph_output_dir = correct_dirs[5]
    drag_dir = correct_dirs[6]
    copyto_dir = correct_dirs[7]
    copyfrom_dir = correct_dirs[8]

    if use_api:
        input_dir.clear()
        del output_dir
        output_dir = []
        for i in range(0, len(api_servers)):
            api_servers[i] = resolve_directories(api_servers[i])[0]
            input_dir.append(api_servers[i])
            output_dir.append(api_servers[i].replace('In', 'Out'))
        print('API input dirs:', input_dir)
        print('API output dirs:', output_dir)

    if use_multiple:
        input_dir.clear()
        del output_dir
        output_dir = []
        for i in range(0, len(available_servers)):
            available_servers[i] = resolve_directories(available_servers[i])[0]
            input_dir.append(available_servers[i])
            output_dir.append(available_servers[i].replace('In', 'Out'))
        print('API input dirs:', input_dir)
        print('API output dirs:', output_dir)

    if lang=='English':
        lang = 'en'
    if lang=='Spanish':
        lang = 'es'
    if lang=='German':
        lang = 'de'

def silero_init():

    print('Loading model...')

    global silero_model, silero_decoder, silero_utils
    silero_model, silero_decoder, silero_utils = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                           model='silero_stt',
                                           language=lang,
                                           device=device)

    os.remove('latest_silero_models.yml')
    print('Model loaded successfully.\n')

    if spellcheck:
        print('Loading spellcheck model...')
        global nlp
        import contextualSpellCheck
        import spacy
        nlp = spacy.load("en_core_web_sm")
        contextualSpellCheck.add_to_pipe(nlp)
        print('Spellcheck model loaded successfully.\n')

    if to_eng:
        print('Loading translation model...\n')
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        global translation_tokenizer
        global translation_model
        translation_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-es-en")
        translation_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-es-en")
        print('Translation model loaded successfully.\n')


def silero_stt(wav, txt, json_name):

    # Silero
    from glob import glob

    print('\nRunning speech-to-text.')
    ## Initialize progress bar to start before transcribing
    rate, audio = read_wave(wav)
    progress_chunks = estimate_progress(audio, rate, 10, 90)
    rt = RepeatedTimer(1, print_model_progress, progress_chunks)  # it auto-starts, no need of rt.start()

    (read_batch, split_into_batches,
     read_audio, prepare_model_input) = silero_utils  # see function signature for details

    try:

        test_files = glob(wav)

        batches = split_into_batches(test_files, batch_size=10)
        input = prepare_model_input(read_batch(batches[0]), device=device)

        output = silero_model(input)
        results = ''
        for words in output:
            results = (silero_decoder(words.cpu()))
            print('Raw text: ', results)

        # SPELLCHECK
        if spellcheck:
            final_text = replace_topchecks(results, nlp=nlp)

        # translate
        if to_eng:
            print('\nTranslating to English...')

            src_text = '>>eng<< ' + final_text
            split_groups = text_to_groups(src_text, max_chars=250)

            # translate the groups
            translate_groups = []
            for word_group in split_groups:
                translated = translation_model.generate(
                    **translation_tokenizer.prepare_seq2seq_batch(word_group, return_tensors="pt"))
                tgt_text = [translation_tokenizer.decode(t, skip_special_tokens=True) for t in translated]
                tgt_text = tgt_text[0].replace(',', ' ').replace('[7]', '')
                translate_groups.append(tgt_text)

            translation = ''
            for word_group in translate_groups:
                translation = translation + word_group + " "

            print('Done translating.\n')

    finally:
        rt.stop()
        print("progress: {}/100".format(90))
    time.sleep(2)
    print('\nSpeech-to-text successful. Saving transcript.')
    text_file = open(txt, "w")
    text_file.write(final_text)
    time.sleep(2)
    text_file.close()
    time.sleep(2)
    shutil.copy2(txt, output_dir + "\\" + txt)
    os.remove(txt)

    if to_eng:
        text_file = open('translated_'+txt, "w")
        text_file.write(translation)
        time.sleep(1)
        text_file.close()
        time.sleep(1)
        shutil.copy2('translated_'+txt, output_dir + "\\" + 'translated_'+txt)
        os.remove('translated_'+txt)

    if silero_time:
        print('\nCreating time stamps for transcript.')
        batch = read_batch(random.sample(batches, k=1)[0])
        input = prepare_model_input(batch, device=device)

        wav_len = input.shape[1] / 16000

        output = silero_model(input)

        for i, example in enumerate(output):
            timed_results = silero_decoder(example.cpu(), wav_len, word_align=True)[-1]
            break

        with open(json_name, 'w') as fp:
           json.dump(timed_results, fp, indent=4)

        shutil.move(json_name, output_dir + "\\" + json_name)


def quartz_init():

    global nemo_gpu

    print('Enabled GPU:', nemo_gpu)

    device = torch.device('cpu')

    if nemo_gpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not torch.cuda.is_available():
            nemo_gpu = False
            print('GPU was not available to PyTorch for inference - using cpu.')
        else:
            torch.cuda.empty_cache()

    # Import Speech Recognition collection
    import nemo.collections.asr as nemo_asr

    global quartznet
    # Speech Recognition model - QuartzNet
    quartznet = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name="QuartzNet15x5Base-En").to(device)

    if punc:
        # Import Natural Language Processing colleciton
        import nemo.collections.nlp as nemo_nlp

        global punctuation
        # Punctuation and capitalization model
        punctuation = nemo_nlp.models.PunctuationCapitalizationModel.from_pretrained(model_name='Punctuation_Capitalization_with_BERT').to(device)

    print('\nLoaded QuartzNet model.\n')

    if spellcheck:
        print('Loading spellcheck model...')
        global nlp
        import contextualSpellCheck
        import spacy
        nlp = spacy.load("en_core_web_sm")
        contextualSpellCheck.add_to_pipe(nlp)
        print('Spellcheck model loaded successfully.\n')

    if to_esp:
        print('Loading translation model...\n')
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        global translation_tokenizer
        global translation_model
        translation_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")
        translation_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-es")
        print('Translation model loaded successfully.\n')

    if diarize:
        from speech.dependencies.diarization import Diarizer


def quartz_stt(wav_name, txt, json_name):

    '''
    from pydub import AudioSegment
    # if need to cut audio
    t1 = 0  # Works in milliseconds
    t2 = 45000
    newAudio = AudioSegment.from_wav(wav_name)
    newAudio = newAudio[t1:t2]
    newAudio.export('temp1.wav', format="wav")  # Exports to a wav file in the current path.
    print('\nPre-processed QuartzNet audio.\n')
    '''
    if nemo_gpu:
        torch.cuda.empty_cache()

    # Convert our audio sample to text
    audio = wav_name
    files = [audio]
    raw_text = ''
    for fname, transcription in zip(files, quartznet.transcribe(paths2audio_files=files)):
        raw_text = transcription
        print('\nRaw text recognized.')
    print("progress: {}/100".format(65))

    if nemo_time:
        with_times = nemo_timestamps(files, quartznet, raw_text)

    #disfluency
    if remove_uhs:
        common_text = remove_stutters(raw_text)
        common_text = common_corrections(common_text)
        if nemo_time:
            new_timestamps = remove_stutters(with_times, time_format=True)
            new_timestamps = common_corrections(new_timestamps, time_format=True)
    else:
        common_text = raw_text
        if nemo_time:
            new_timestamps = with_times


    #acronyms
    try:
        if acro:
            acro_text, acros = acronym_fix(common_text)
            if nemo_time:
                acro_stamps, acros = acronym_fix(common_text, time_format=True, input_stamps=new_timestamps)
        else:
            acro_text = common_text
            if nemo_time:
                acro_stamps = new_timestamps
    except:
        print('Acronym replacement failed. Continuing...\n')
        pass
        acro_text = common_text
        if nemo_time:
            acro_stamps = new_timestamps


    #punc+cap
    if punc:
        punctuation._cfg.dataset.max_seq_length = 1000

        split_groups = text_to_groups(acro_text)

        # Add capitalization and punctuation
        restored_groups = []
        for word_group in split_groups:
            res = punctuation.add_punctuation_capitalization(queries=[word_group])
            text = res[0]
            restored_groups.append(text)
        restored_text = ''
        for word_group in restored_groups:
            restored_text = restored_text + decapitalize(word_group[:-1]) + " "
        if acro:
            restored_text = recapitalize_acros(restored_text, acros)
    else:
        restored_text = acro_text

    #spellcheck
    if spellcheck:
        final_text = replace_topchecks(restored_text, nlp=nlp)
    else:
        final_text=restored_text

    #reconnect stamps w/ corrected text
    if nemo_time:
        final_stamps = reconnect_timetext(final_text, acro_stamps)

    #translate
    if to_esp:
        print('\nTranslating to Spanish...')

        src_text = '>>esp<< '+final_text
        split_groups = text_to_groups(src_text, max_chars=500)

        # translate the groups
        translate_groups = []
        for word_group in split_groups:
            translated = translation_model.generate(**translation_tokenizer.prepare_seq2seq_batch(word_group, return_tensors="pt"))
            tgt_text = [translation_tokenizer.decode(t, skip_special_tokens=True) for t in translated]
            tgt_text = tgt_text[0].replace(',', ' ').replace('[7]', '')
            translate_groups.append(tgt_text)

        translation = ''
        for word_group in translate_groups:
            translation = translation + word_group +  " "

        print('Done translating.\n')

    print("progress: {}/100".format(85))

    out_dir=current_dir.replace('In','Out')
    print('Outdir:', out_dir)


    time.sleep(1.2)
    print('\nSpeech-to-text successful. Saving transcript...')
    text_file = open(txt, "w")
    text_file.write(final_text)
    time.sleep(1)
    text_file.close()
    print('Saved transcipt.')

    if to_esp:
        text_file = open('translated_'+txt, "w")
        text_file.write(translation)
        time.sleep(1)
        text_file.close()
        time.sleep(1)
        shutil.copy2('translated_'+txt, out_dir + "\\" + 'translated_'+txt)
        os.remove('translated_'+txt)

    time.sleep(1)
    shutil.copy2(txt, out_dir + "\\" + txt)
    os.remove(txt)

    global diarize
    diarize = False
    if diarize:
        print('\nDiarizing audio with transcript...')
        diarizer = Diarizer(audio, final_stamps)
        diarizer.create_dia_json()
        diarizer.create_speaker_segments()
        diarizer.align_timed_trans()
        diarized_segments = diarizer.speaker_segments
        newest_trans = diarizer.get_timed_trans()
        final_stamps = newest_trans

        non_stamped_dia = False
        if non_stamped_dia:
            dia_json = 'diarized_' + json_name
            with open(dia_json, 'w') as fp:
                json.dump(diarized_segments, fp, indent=4)
            shutil.move(dia_json, out_dir + "\\" + dia_json)
            print('Diarization successful. Saved diarized transcript.')

    if nemo_time:
        print('\nSaving timestamps into .json...')
        with open(json_name, 'w') as fp:
            json.dump(final_stamps, fp, indent=4)
        shutil.move(json_name, out_dir + "\\" + json_name)
        print('Saved timestamps.')


    if nemo_gpu:
        torch.cuda.empty_cache()

    del raw_text, restored_text, acro_text, final_text

    if nemo_time:
        del with_times, new_timestamps, acro_stamps, final_stamps
    if diarize:
        del diarizer, diarized_segments

    gc.collect()


def speech_to_text(event_path, input_directory):

    # 1) Initialize in system
    print("progress: {}/100\n".format(0))
    if existing!=True:
        print("Received input event: %s.\n" % event_path)
    filename = event_path.replace(input_directory, "")
    print("Name of input file: " + event_path + "\n")
    prog_check=0
    up_to = 8
    for i in range(up_to-prog_check):
        print("progress: {}/100".format(prog_check + i))
        sys.stdout.flush()
        time.sleep(1)

    local_copy = event_path
        ## Make local copy on C: drive
    temp_dir = "./speech/cache/"
    print('Copying to cache...')
    local_copy = copy_to_local(event_path, temp_dir)
    filename = local_copy.replace(temp_dir, "")

        ## Verify input has valid audio for speech-to-text
    print('\nChecking for audio in the input...\n')
    audio_verify = resolve_audio(local_copy, event_path)
    if audio_verify:
        print('\nAudio Verifired.')
    else:
        return


    # 2) Data Preprocessing
        ## Get name of file and give file extension, wav name, and json name
    wav_name, json_name, txt_name = make_names(filename)

        ## Convert file to .wav format
    convert_to_wav(local_copy, wav_name)
    print('Successfully converted to .wav\n')


    print("progress: {}/100".format(15))

        ## Save audio values if requested
    if save_wav_graph or audio_values:
        print('Creating Wav Signal Graph')
        x_time, y_amplitude, wav_graph = get_audio_values(wav_name)
        print('\nGraph created.')
        if save_wav_graph:
            graph_name = graph_output_dir + wav_name.replace('.wav', '_graph.png')
            plt.savefig(graph_name, bbox_inches = "tight")
            print('\nSaved audio graph to: ', graph_name)
        if audio_values:
            audio_values_json = values_output_dir + wav_name.replace('.wav', '_values.json') 
            audio_value_dict = {'x-axis': x_time, 'y-axis' : y_amplitude}
            with open(audio_values_json, 'w') as fp:
                json.dump(audio_value_dict, fp, indent=4, cls=NpEncoder)
            print('\nAudio values saved to: {}\n'.format(audio_values_json))

    if speech_model == 'Silero':
        silero_stt(wav_name, txt_name, json_name)
    if speech_model == 'NeMo':
        quartz_stt(wav_name, txt_name, json_name)

    ## Move files into correct areas
    if not use_multiple:
        shutil.move(event_path, prev_input_dir + "\\" + filename)
        try:
            shutil.copy2(wav_name, wav_dir + "\\" + wav_name)
        except:
            shutil.move(wav_name, wav_dir + "\\" + wav_name)

    if dragon:
        shutil.copy2(wav_dir + "\\" + wav_name, drag_dir + "\\" + wav_name)
        print('\nMoved audio to Dragon folder.\n')
    if copy_drag:
        max_wait = 90
        shutil.copy2(wav_dir + "\\" + wav_name, copyto_dir + "\\" + wav_name)
        print('\nWaiting for Dragon to transcribe to copy its output...({}s max)\n'.format(max_wait))
        count = 0
        while os.path.exists(drag_dir + "\\"+ wav_name):
            time.sleep(1)
            count += 1
            if count % 10 == 0:
                print('\nWaited for {}s..\n'.format(count))
            if count == max_wait:
                break
        try:
            shutil.copy2(copyfrom_dir + "\\"+ wav_name.replace('.wav','_wav.txt'), copyto_dir + "\\"+ wav_name.replace('.wav','_wav.txt'))
        except:
            print('\nRan out of time or file does not exist.\n')
            pass
    os.remove(local_copy)
    os.remove(wav_name)
    os.remove(event_path)
    print("progress: {}/100\n".format(100))
    print('Done.\n\nReady for next transciption.\n')


def send_to_api(file_name):
    import requests
    print('Preparing to send to API...')
    wav_name, json_name, txt_name = make_names(file_name)
    time.sleep(10)

    print('Sending:', file_name)
    with open(file_name, 'rb') as f:
        files = {'file': f}
        session = requests.Session()
        del session.headers['User-Agent']
        del session.headers['Accept-Encoding']
        response = session.post(api_endpoint, files=files)
        f.close()
    response = response.json()
    final_stamps = response['data']['timed_transcript']

    print('\nSaving timestamps into .json...', json_name)
    with open(json_name.replace('VideoBankSpeech_In', 'VideoBankSpeech_Out'), 'w') as fp:
        json.dump(final_stamps, fp, indent=4)
        fp.close()
    os.remove(file_name)
    print('Saved timestamps.\nReady for next input')


def check_existing_stt():
    global existing
    global current_dir
    for dir in input_dir:
        while os.listdir(dir)!=[]:
            print('Existing file found in: ' + dir)
            existing = True
            filelist = os.listdir(dir)
            existingfile = dir + filelist[0]
            if use_api:
                send_to_api(existingfile)
            else:
                current_dir = dir
                speech_to_text(existingfile, dir)




# Handles events of a directory
class stt_Handler(FileSystemEventHandler):

    def on_any_event(self, event):

        if event.is_directory:
            print(event.src_path)
            return None

        elif event.event_type == 'created':
            global current_dir
            event_path = event.src_path
            if use_api:
                send_to_api(event_path)
            else:
                dir = str(pathlib.Path(event_path).parent.absolute())
                current_dir = dir
                speech_to_text(event_path, dir)

# Watches for events
class stt_Watcher:
    def __init__(self, watched_dirs):
        self.observer = Observer()
        self.watched_dirs = watched_dirs

    def run(self):
        print('Watching directories:', self.watched_dirs)
        event_handler = stt_Handler()
        for i in range(0, len(self.watched_dirs)):
            self.observer.schedule(event_handler, self.watched_dirs[i], recursive=True)
        self.observer.start()
        try:
            while True:
                time.sleep(5)
        except:
            self.observer.stop()
            print("Stopped!")

        self.observer.join()


# Concized it altogether
def run_stt(args):
    print("progress: {}/100".format(0))
    make_stt_vars(args)
    if args.speech_model=='NeMo' and not args.use_api:
        print('Loading models...')
        quartz_init()
    if args.speech_model=='Silero' and not args.use_api:
        silero_init()
    check_existing_stt()
    w = stt_Watcher(input_dir) #uses speech_to_text method on creation events of directory
    if existing!=True:
        print('Ready for first transcription.\n')
    w.run()
