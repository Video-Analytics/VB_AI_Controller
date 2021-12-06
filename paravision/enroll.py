import gc, os, sys, time, json, shutil, re
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from utils import resolve_directories, RepeatedTimer, estimate_vid_progress, print_model_progress

import sqlite3
from typing import Optional
import paravision_processor
import paravision_identity

def make_variables():
    global input_dir
    global db_name
    global raw_db_name
    
    global add_db_name
    global add_facedb
    
    global restore_db
    global restore_name
    
    input_dir = args.input_dir
    db_name = args.db_name
    
    add_db_name = args.add_db_name
    add_facedb = args.add_facedb
    
    restore_db = args.restore_facedb
    restore_name = args.restore_name
    
    if add_facedb:
        db_name = './paravision/dbs/' + add_db_name.lower() + '.sqlite'
        raw_db_name = add_db_name.lower()
    elif restore_db:
        print("Database restoration process intiated.")
        raw_db_name = str(restore_name)
        db_name = './paravision/dbs/' + raw_db_name.lower() + '.sqlite'
    else:
        db_name = './paravision/dbs/' + args.db_name.lower()
        raw_db_name = args.db_name.lower()

    restore_name = './paravision/dbs/' + raw_db_name + '/'
    if not os.path.isdir(restore_name):
        try:
            os.mkdir(restore_name)
        except FileExistsError as e:
            print("Backup directory already exists.")
    restore_name.replace(".sqlite", '')

    correct_dirs = resolve_directories(input_dir)
    input_dir = correct_dirs[0]

def initiate_enrollment():
    global processor_client
    global identity_client
    global conn
    global group_id

    host_ip = "172.19.3.36"
    DEFAULT_PROCESSOR_URI = f"{host_ip}:50051"
    DEFAULT_IDENTITY_URI = f"{host_ip}:5656"
    
    print('Group name:', raw_db_name)
    GROUP_NAME = raw_db_name

    # Create both client instances
    processor_client = paravision_processor.Client(DEFAULT_PROCESSOR_URI)
    identity_client = paravision_identity.Client(DEFAULT_IDENTITY_URI)

    # Connect to both services
    processor_client.connect()
    identity_client.connect()

    # Prepare a local database
    conn = local_db__prepare()

    # Find the working group used by this demo
    group_id = local_db__get_working_group_id(conn)
    if group_id:
        print('Group ID:', group_id)
    # Create the group, if it does not exist
    if not group_id:
        group = identity_client.create_group(GROUP_NAME)
        group_id = group.id
        print('Group ID:', group_id)
        local_db__store_working_group_id(conn, group_id)

    if restore_db:
        restore_facedb()

        
def restore_facedb():
    for file in os.listdir(restore_name):
        print('Found enroll file:', restore_name + file)
        enroll(restore_name + file)
        
    print("Restoration complete. Please restart enrollment service with restored database.")
    
def store_restore(enroll_file):
    if not restore_db:
        base_name = enroll_file.replace(input_dir, '').replace('.sqlite', '')
        restore_file = restore_name + base_name.replace('.sqlite', '')
        restore_file = restore_file.replace('.sqlite', '')
        print('Restore file:', restore_file)
        if not os.path.isfile(restore_file):
            shutil.copy2(enroll_file, restore_file)
        print('\nBackup file created.\n')
    

# Register a new identity from a photo
def enroll(enroll_file):
    store_restore(enroll_file)
    with open(enroll_file) as json_file:
        data = json.load(json_file)
        name = data['name']
        image = data['image']
        img = r"{}".format(image)
        print('IMAGE LOCATION:', img)
    # Get the face from the image
    with open(img, "rb") as image:
        data = image.read()
    faces = processor_client.process_full_image(
        data, [paravision_processor.ProcessFullImageOptions.EMBEDDING,
               paravision_processor.ProcessFullImageOptions.QUALITY],
        False).faces
    if len(faces) == 0:
        print("Rejected image - No faces found on the image")
        return
    if len(faces) > 1:
        print("Rejected image - More than one face found in the image")
        return

    # Register a new identity. Using a non-zero confidence threshold
    # ensures we do not register the same person mutliple times.
    person_id = local_db__create_person(conn, name)
    try:
        identity = identity_client.create_identity(
            faces[0].embedding, 0.9, faces[0].quality, [group_id],
            external_id=person_id)
    except paravision_identity.ClientException as e:
        print(f"Failed to register a new identity - Error: {e}\nRetrying in 2 sec...")
        if str(e) == "StatusCode.ALREADY_EXISTS: create identities: " \
                "embedding is too similar to an existing identites' embedding":
            print('Already exists, so moving to next person.')
            pass
        else:
            time.sleep(2)
            check_waiting_enrollment()

    print(f"Added a new person: {name} \nTo '{raw_db_name}' database")
    if not restore_db:
        os.remove(enroll_file)
    
    local_db__close(conn)
    identity_client.close()
    processor_client.close()
    initiate_enrollment()
    print('\nReady for next enrollment.')


def local_db__prepare() -> sqlite3.Connection:
    conn = sqlite3.connect(db_name, check_same_thread=False)
    conn.executescript("""
        create table if not exists people (
            name text,
            created_at timestamp default current_timestamp
        );
        create table if not exists metadata (
            key text unique primary key,
            value text
        );""")
    conn.commit()
    return conn


def local_db__clear() -> None:
    if os.path.exists(db_name):
        os.remove(db_name)


def local_db__get_working_group_id(
        conn: sqlite3.Connection) -> Optional[str]:
    c = conn.cursor()
    c.execute("select value from metadata where key = 'working_group_id'")
    res = c.fetchone()
    return res[0] if res else None


def local_db__store_working_group_id(
        conn: sqlite3.Connection, group_id: str) -> None:
    conn.execute("insert or replace into metadata (key, value) values "
                 "('working_group_id', ?)", [group_id])
    conn.commit()


def local_db__get_person_name(
        conn: sqlite3.Connection, person_id: str) -> Optional[str]:
    c = conn.cursor()
    c.execute("select name from people where rowid = ?", [person_id])
    res = c.fetchone()
    return res[0] if res else None


def local_db__create_person(
        conn: sqlite3.Connection, name: str) -> str:
    cur = conn.cursor()
    cur.execute("insert or replace into people (name) values (?)",
                [name])
    conn.commit()
    return str(cur.lastrowid)


def local_db__close(conn: sqlite3.Connection):
    conn.close()


# Handles events of a directory
class face_rec_Handler(FileSystemEventHandler):
    @staticmethod
    def on_any_event(event):

        if event.is_directory:
            print(event.src_path)
            return None

        elif event.event_type == 'created':
            print('\nInput recieved. Running enrollment in 2 sec...')
            time.sleep(2)
            file = event.src_path
            try:
                enroll(file)
            except:
                print('\nError occurred.\nRestarting...')
                w = face_rec_Watcher(input_dir)
                check_waiting_enrollment()
                print('\nReady for enrollment.')
                w.run()



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

def check_waiting_enrollment():
    global existing
    existing = False
    while os.listdir(input_dir)!=[]:
        existing = True
        filelist = os.listdir(input_dir)
        existingfile = input_dir + filelist[0]
        print('Existing file found: ' + existingfile)
        enroll(existingfile)

def run_enrollment(opt):
    global args
    args = opt
    print('Initializing connections...')
    make_variables()
    initiate_enrollment()
    check_waiting_enrollment()
    w = face_rec_Watcher(input_dir)
    if existing!=True:
        print('Ready for first enrollment.')
    w.run()