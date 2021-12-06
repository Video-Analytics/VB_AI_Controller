import gc, os, sys, time, json, shutil
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from utils import resolve_directories, RepeatedTimer, estimate_vid_progress, print_model_progress
import filetype
import cv2
import sqlite3
from typing import Optional
import paravision_processor as processor
import paravision_identity as identity


def make_variables():
    global input_dir
    global output_dir
    global server_db
    global raw_db_name
    global confidence_thres

    input_dir = args.input_dir
    output_dir = args.output_dir
    server_db = './paravision/dbs/' + args.server_db
    raw_db_name = args.server_db.replace('.sqlite', '')
    confidence_thres = float(args.confidence_thres)

    correct_dirs = resolve_directories(input_dir, output_dir)
    input_dir = correct_dirs[0]
    output_dir = correct_dirs[1]
    

def initialize_face_rec():
    global processor_client
    global identity_client
    global conn
    global group_id

    host_ip = "172.19.3.36"
    PROCESSOR_URI = f"{host_ip}:50051"
    IDENTITY_URI = f"{host_ip}:5656"
    
    print("Connecting to IP:", host_ip)

    # Create both client instances
    processor_client = processor.Client(PROCESSOR_URI)
    identity_client = identity.Client(IDENTITY_URI)

    # Connect to both services
    processor_client.connect()
    identity_client.connect()

    # Prepare a local database
    conn = local_db__prepare()

    # Find the working group used by this demo
    group_id = local_db__get_working_group_id(conn)
    if not group_id:
        group = identity_client.create_group(raw_db_name)
        group_id = group.id
        local_db__store_working_group_id(conn, group_id)
    print('Successfully connected.\n')

def make_names(video):
    base_name = video.replace(input_dir, '')
    sep_ext = base_name.split('.')
    base_name = sep_ext[0]
    
    progress_folder = output_dir + base_name + '/' + 'progress/'
    print('Progress directory:', progress_folder, '\n')
    
    json_name = base_name + '.json'
    json_name = output_dir + base_name + '/' + json_name

    if os.path.exists(progress_folder):
        shutil.rmtree(progress_folder, ignore_errors=True)
    os.makedirs(progress_folder)

    return progress_folder, json_name
    

def face_recognition(video):
    global group_id
    
    cap = cv2.VideoCapture(video)
    detected_faces = []
    cached_faces = []
    time_and_faces = {}
    # count = 0
    start_time = 0
    current_time = 0
    
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_time = total_frames/fps
    # f_target = fps * 15
    # print('Target frame:', f_target)
    print('Calculated video length:', total_time)
    
    progress_folder, json_name = make_names(video)
    time.sleep(1)

    while (cap.isOpened()):
        ret, frame = cap.read()
        current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        current_time = round((current_frame*total_time)/total_frames, 2)
        
        print_facerec_progress(current_frame, total_frames, progress_folder)
        

        # get a frame
        if ret == True:
            is_success, im_buf_arr = cv2.imencode(".jpg", frame)
            image_data = im_buf_arr.tobytes()

            # get embeddings, quality, and bounding box from frame
            response = processor_client.process_full_image(
                image_data, [
                    processor.ProcessFullImageOptions.EMBEDDING,
                    processor.ProcessFullImageOptions.QUALITY
                ], True)

            faces = response.faces

            # if there are any faces draw bounding box, give quality, and recognize
            if len(faces) > 0:
                for i, face in enumerate(faces):
                    # give quality
                    quality = face.quality

                    ## Recognize Face ##
                    # Multi-face lookup
                    try:
                        ci_matrix = identity_client.lookup([face.embedding], [group_id], 1)
                    except:
                        try:
                            group = identity_client.create_group(raw_db_name)
                            group_id = group.id
                            print('Group was not in paravision docker db - group added.')
                        except:
                            ci_matrix = identity_client.lookup([face.embedding], [group_id], 1)
                    for cis in ci_matrix:
                        if len(cis) == 0:
                            # No matches for this face, but maybe other have some.
                            continue
                        name = local_db__get_person_name(conn, cis[0][0].external_id)
                        confidence = cis[0][1]

                    #    print("Face ", i + 1, "(out of " + str(len(faces)) + "): "
                    #          + "Name: " + str(name) + ", "
                    #          + "Confidence: " + str(confidence) + ", "
                    #          + "Quality: " + str(quality) + ", ")

                        # accept face if higher than confidence threshold
                        if (confidence > confidence_thres) and (name not in detected_faces):
                            detected_faces.append(name)
                        else:
                            print('Unknown face detected.')
                            
            print(f'Frame {current_frame} / {total_frames} : {detected_faces}')
            if sorted(detected_faces) != sorted(cached_faces) and ((current_time-start_time) > 0.5):
                time_and_faces[str(start_time) + '-' + str(current_time)] = [name for name in sorted(cached_faces)]
                cached_faces.clear()
                for name in detected_faces:
                    cached_faces.append(name)
                start_time = current_time
            detected_faces.clear()

            # if current_frame >= f_target:
                # time_and_faces[f'{count} - {count+15}'] = detected_faces
                # detected_faces.clear()
                # detected_faces = []
                # f_target = f_target*2
                # print('Target frame:', f_target)
        else:
            break

    # get leftovers
    time_and_faces[f'{start_time} - {current_time}'] = cached_faces


    print('Finished - removing video and creating output.\n')
    # Release everything if job is finished
    cap.release()
    cv2.destroyAllWindows()
    os.remove(video)
    
    print('Json output name:', json_name)
    with open(json_name, 'w') as fp:
        json.dump(time_and_faces, fp, indent=4)
        
    file = open(progress_folder+"10.txt", "w")
    file.close()

    local_db__close(conn)
    identity_client.close()
    processor_client.close()
    initialize_face_rec()

    print('Ready for next face recognition.')


def local_db__prepare() -> sqlite3.Connection:
    connect = sqlite3.connect(server_db, check_same_thread=False)
    connect.executescript("""
        create table if not exists people (
            name text,
            created_at timestamp default current_timestamp
        );
        create table if not exists metadata (
            key text unique primary key,
            value text
        );""")
    connect.commit()
    return connect


def local_db__get_working_group_id(
        connect: sqlite3.Connection) -> Optional[str]:
    c = connect.cursor()
    c.execute("select value from metadata where key = 'working_group_id'")
    res = c.fetchone()
    return res[0] if res else "Persons"


def local_db__store_working_group_id(
        connect: sqlite3.Connection, group: str) -> None:
    connect.execute("insert or replace into metadata (key, value) values "
                 "('working_group_id', ?)", [group])
    connect.commit()


def local_db__get_person_name(
        connect: sqlite3.Connection, person_id: str) -> Optional[str]:
    c = connect.cursor()
    c.execute("select name from people where rowid = ?", [person_id])
    res = c.fetchone()
    return res[0] if res else None


def local_db__close(connect: sqlite3.Connection):
    connect.close()


# Handles events of a directory
class face_rec_Handler(FileSystemEventHandler):
    @staticmethod
    def on_any_event(event):

        if event.is_directory:
            print(event.src_path)
            return None

        elif event.event_type == 'created':
            print('\nInput recieved. Running face recognition in 5 sec...')
            time.sleep(5)
            event_path = event.src_path
            event_type = filetype.guess(event_path)
            if ('video' in event_type.mime) or (event_path.endswtih('.ts')) and (event_path):
                face_recognition(event_path)
            else:
                print('Input type not a video. Not running face detection')
                try:
                    os.remove(event_path)
                except:
                    pass



# Watches for events
class face_rec_Watcher():
    def __init__(self, watched_dir):
        self.observer = Observer()
        self.watched_dir = watched_dir

    def run(self):
        event_handler = face_rec_Handler()
        self.observer.schedule(event_handler, self.watched_dir, recursive=True)
        self.observer.start()
        try:
            while True:
                time.sleep(5)
        except:
            self.observer.stop()
            print("Stopped!")

        self.observer.join()
        
def print_facerec_progress(current_frame, final_frames, progress_folder):
    percent_done = int((round((current_frame/final_frames), 2)*100))
    print("progress: {}/100".format(percent_done))

    if (10 <= percent_done < 20) and not os.path.isfile(progress_folder+"1.txt"):
        file = open(progress_folder+"1.txt", "w")
        file.close()
    if 20 <= percent_done < 30 and not os.path.isfile(progress_folder+"2.txt"):
        file = open(progress_folder+"2.txt", "w")
        file.close()
    if 30 <= percent_done < 40 and not os.path.isfile(progress_folder+"3.txt"):
        file = open(progress_folder+"3.txt", "w")
        file.close()
    if 40 <= percent_done < 50 and not os.path.isfile(progress_folder+"4.txt"):
        file = open(progress_folder+"4.txt", "w")
        file.close()
    if 50 <= percent_done < 60 and not os.path.isfile(progress_folder+"5.txt"):
        file = open(progress_folder+"5.txt", "w")
        file.close()
    if 60 <= percent_done < 70 and not os.path.isfile(progress_folder+"6.txt"):
        file = open(progress_folder+"6.txt", "w")
        file.close()
    if 70 <= percent_done < 80 and not os.path.isfile(progress_folder+"7.txt"):
        file = open(progress_folder+"7.txt", "w")
        file.close()
    if 80 <= percent_done < 90 and not os.path.isfile(progress_folder+"8.txt"):
        file = open(progress_folder+"8.txt", "w")
        file.close()
    if 90 <= percent_done < 100 and not os.path.isfile(progress_folder+"9.txt"):
        file = open(progress_folder+"9.txt", "w")
        file.close()


def check_existing_face_rec():
    global existing
    existing = False
    while os.listdir(input_dir)!=[]:
        existing = True
        filelist = os.listdir(input_dir)
        existingfile = input_dir + filelist[0]
        print('Existing file found: ' + existingfile)
        event_type = filetype.guess(existingfile)
        if type(existingfile) is None:
            existing = False
            return
        if 'Thumbs.db' in existingfile:
            existing = False
            print('We hate Thumbs.db - no detection for Thumbs.db\n')
            return
        if ('video' in event_type.mime) or (existingfile.endswtih('.ts')):
            time.sleep(5)
            face_recognition(existingfile)
        else:
            print('Input type not a video. Not running face detection')
            try:
                os.remove(existingfile)
            except:
                pass


def run_face_rec(opt):
    global args
    args = opt
    print('Initializing connections...')
    make_variables()
    initialize_face_rec()
    check_existing_face_rec()
    w = face_rec_Watcher(input_dir)
    if existing!=True:
        print('Ready for first face recognition.')
    w.run()
