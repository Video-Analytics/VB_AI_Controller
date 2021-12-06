import os, time, shutil, csv, subprocess, sys, psutil
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from utils import *
from moviepy.editor import VideoFileClip, concatenate_videoclips



def make_concat_vars(arg_parser):
    global input_dir 
    global output_dir
    global existing
    global vidlist
    global start

    input_dir = arg_parser.input_dir           
    output_dir = arg_parser.output_dir            
    existing = False 
    start = False
    vidlist = []

    #correct and handle directory paths
    correct_dirs = resolve_directories(input_dir, output_dir)

    #rename dir to usable paths
    input_dir = correct_dirs[0]            
    output_dir = correct_dirs[1]



# Handles events of a directory
class concat_Handler(FileSystemEventHandler):
    @staticmethod
    def on_any_event(event):
        global start

        if event.event_type == 'created':
            if event.is_directory:
                print('Folder Recieved.', event.src_path)
                event_path = event.src_path
                cache_check(True, event_path.replace(input_dir, './concat/cache/'))

                time.sleep(4)
                while os.listdir(input_dir):
                    try:
                        movetocache(event_path)
                    except:
                        print('Folder still loading. Retrying in 5 sec...')
                        time.sleep(5)
                        shutil.rmtree((event_path.replace(input_dir, "./concat/cache/")))
                        pass
                concatenation(event_path)


# Watches for events
class concat_Watcher():
    def __init__(self, watched_dir):
        self.observer = Observer()
        self.watched_dir = watched_dir

    def run(self):
        event_handler = concat_Handler()
        self.observer.schedule(event_handler, self.watched_dir, recursive=True)
        self.observer.start()

        try:
            while True:
                time.sleep(5)
        except:
            self.observer.stop()
            print("Stopped!")

        self.observer.join()


def addtovidlist(item, folder):
    global vidlist
    time.sleep(2)
    vidlist.append(item.replace(" ", ''))
    try:
        vidlist.remove(folder)
    except:
        pass


def movetocache(item):
    if item == input_dir:
        return
    destination = item.replace(input_dir, "./concat/cache/")
    print('\nCopying {} to {}'.format(item, destination))
    try:
        shutil.copytree(item, destination)
    except shutil.Error:
        shutil.rmtree(destination, ignore_errors=True)
        shutil.copytree(item, destination)
    if os.path.isfile(destination+'start.csv'):
        shutil.rmtree(item)
    else:
        shutil.rmtree(destination, ignore_errors=True)
        shutil.copytree(item, destination)
        shutil.rmtree(item)
    print('\nDone moving items to cache.')



def clearvidlist(folder):
    global vidlist
    vidlist.clear()
    print('Attempting to remove folder: ', folder)
    shutil.rmtree(folder)
    print('Cache cleared.\n')


def cache_check(end, folder):
    if not end:
        if os.listdir('./concat/cache/'):
            print('Items in cache detected. Clearing.')
            try:
                str_error = None
                clearvidlist(folder)
            except Exception as str_error:
                print('Failed to clear cache. Continuing with concatenation.')
                pass
    if end:
        count = 0
        while os.listdir('./concat/cache/') and count < 11:
            count+=1
            try:
                str_error = None
                clearvidlist(folder)
            except Exception as str_error:
                print(str_error)
                time.sleep(2)
                print('Failed to clear cache. Retrying in 2s...')
                pass


def getfilenames(folder):
    filenames = []
    for vid in vidlist:
        vid.replace(folder, '')
        filenames.append(vid)
    return filenames


def ordervidlist(folder):
    try:
        with open(folder+"start.csv", newline='') as f:
            reader = csv.reader(f)
            order = list(reader)
    except:
        print('Failed to find csv. Trying again in 2 sec...')
        time.sleep(2)
        ordervidlist(folder)
    order = order[0]
    for vid in order:
        vid = folder + vid.replace(" ", '')
        addtovidlist(vid, folder)


def print_concat_progress(final_frames, progress_folder, final_folder):
    try:
        with open(final_folder+'output.mp4.log', 'r') as f:
            last_line = f.readlines()[-1]
            if 'frame=' in last_line:
                current_frame = int(float(last_line.split('fps')[0].replace('frame=', '').replace(" ", "")))
            print('')  # so progress below can print cleanly
            percent_done = int((round((current_frame/final_frames), 2)*100))
            print("progress: {}/100".format(percent_done))

            if (10 <= percent_done < 20) and not os.path.isfile(progress_folder+"1.txt"):
                print('Making 1.txt')
                file = open(progress_folder+"1.txt", "w")
                file.close()
            if 20 <= percent_done < 30 and not os.path.isfile(progress_folder+"2.txt"):
                print('Making 2.txt')
                file = open(progress_folder+"2.txt", "w")
                file.close()
            if 30 <= percent_done < 40 and not os.path.isfile(progress_folder+"3.txt"):
                print('Making 3.txt')
                file = open(progress_folder+"3.txt", "w")
                file.close()
            if 40 <= percent_done < 50 and not os.path.isfile(progress_folder+"4.txt"):
                print('Making 4.txt')
                file = open(progress_folder+"4.txt", "w")
                file.close()
            if 50 <= percent_done < 60 and not os.path.isfile(progress_folder+"5.txt"):
                print('Making 5.txt')
                file = open(progress_folder+"5.txt", "w")
                file.close()
            if 60 <= percent_done < 70 and not os.path.isfile(progress_folder+"6.txt"):
                print('Making 6.txt')
                file = open(progress_folder+"6.txt", "w")
                file.close()
            if 70 <= percent_done < 80 and not os.path.isfile(progress_folder+"7.txt"):
                print('Making 7.txt')
                file = open(progress_folder+"7.txt", "w")
                file.close()
            if 80 <= percent_done < 90 and not os.path.isfile(progress_folder+"8.txt"):
                print('Making 8.txt')
                file = open(progress_folder+"8.txt", "w")
                file.close()
            if 90 <= percent_done < 100 and not os.path.isfile(progress_folder+"9.txt"):
                print('Making 9.txt')
                file = open(progress_folder+"9.txt", "w")
                file.close()
            #print('Frame count: ', current_frame, '/', final_frames)
    except:
        print('No build-video log found.')
        return


def create_dirs(cached_folder):
    final_folder = cached_folder.replace('./concat/cache/', output_dir)
    error_folder = cached_folder.replace('cache', 'repair')
    pro_folder = final_folder+'progress/'
    try:
        os.mkdir(final_folder)
        os.mkdir(pro_folder)
    except:
        shutil.rmtree(final_folder)
        os.mkdir(final_folder)
        os.mkdir(pro_folder)
    try:
        os.mkdir(error_folder)
    except:
        shutil.rmtree(error_folder)
        os.mkdir(error_folder)
    return final_folder, pro_folder, error_folder


def vid_check_and_correct(error_folder):
    print('\nChecking integrity of videos.')
    for vid in vidlist:
        check_cmd = 'ffprobe -v error -i {} 2>./concat/repair/error.log'.format(vid)
        subprocess.call(check_cmd, shell=True, stdout=subprocess.PIPE)
        with open('./concat/repair/error.log') as f:
            error = f.read()
            f.close()
        if error == '' or 'No such file or directory' in error:
            continue
        else:
            print('\nVideo error found in {}'.format(vid.replace('./concat/cache/','')))
            print('Repairing video.')
            #move video to repair
            dest = vid.replace('cache', 'repair')
            shutil.copy2(vid, dest)
            os.remove(vid)
            #output repaired video back into cache
            check_cmd = 'ffmpeg -hide_banner -loglevel quiet -nostats -i {} {}'.format(dest, vid)
            subprocess.call(check_cmd, shell=True, stdout=subprocess.PIPE)
    print('\nIntegrity check and repair complete.')
    os.remove('./concat/repair/error.log')
    shutil.rmtree(error_folder)


def remove_namespaces(folder):
    replaced = False
    for f in os.listdir(folder):
        r = f.replace(" ", "")
        if r != f:
            os.rename(folder + f, folder + r)
            replaced = True
    return replaced
	
def kill_ffmpeg():
    PROCNAME = "ffmpeg-win64-v4.2.2.exe"
    for proc in psutil.process_iter():
        # check whether the process name matches
        if proc.name() == PROCNAME:
            proc.kill()


def concatenation(folder):
    folder = folder.replace(input_dir, "./concat/cache/") + '/'
    print('Working in cached folder: ', folder)
    final_folder, pro_folder, error_folder = create_dirs(cached_folder=folder)
    ordervidlist(folder)
    vid_check_and_correct(error_folder)
    did_replace = remove_namespaces(folder)
    print('\nRunning concatenation on videos:', vidlist)
    time.sleep(1)
    clips=[]
    if did_replace:
        time.sleep(2)
    for vid in vidlist:
        print(vid)
        clip = VideoFileClip(vid)
        clips.append(clip)
        clip.close
    final_clip = concatenate_videoclips(clips, method='compose')

    frames = int(24 * final_clip.duration)
    print('\nFinal frame count in concatenated clip: ', frames)
    rt = RepeatedTimer(1, print_concat_progress, frames, pro_folder, final_folder)  # it auto-starts, no need of rt.start()
    try:
        final_clip.write_videofile(final_folder + 'output.mp4', codec='libx264', preset='ultrafast', write_logfile=True,
                                   fps=29.970, ffmpeg_params=['-profile:v', 'baseline'])
    finally:
        time.sleep(1)
        rt.stop()
        try:
            print('Making 10.txt')
            file = open(pro_folder+"10.txt", "w")
            file.close()
        except:
            print('Output already deleted.')
            pass
    final_clip.close

    print("Clips concatenated. Waiting to clear cache...")
    time.sleep(5)
    kill_ffmpeg()
    cache_check(True, folder)
    try:
        os.remove(final_folder+'output.mp4.log')
    except:
        pass
    print('\nReady for next concatenation.\n')


def check_existing_vids():
    global existing
    if os.listdir(input_dir):
        existing = True
        filelist = os.listdir(input_dir)
        print('Existing folder found: ', filelist)
        movetocache(input_dir + filelist[0])
        concatenation(input_dir + filelist[0])
		

def run_concat(args):
    print("progress: {}/100".format(0))
    make_concat_vars(args)
    check_existing_vids()
    w = concat_Watcher(input_dir) #uses speech_to_text method on creation events of directory
    if not existing:
        print('Ready for first concatenation.\n')
    w.run()