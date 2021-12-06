import gc, os, sys, math
from numpy import random
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import torch
import torch.backends.cudnn as cudnn
from utils import resolve_directories, RepeatedTimer, estimate_vid_progress, print_model_progress

from yolov4.utils.torch_utils import select_device, time_synchronized
from yolov4.models.models import *
from yolov4.utils.datasets import *
from yolov4.utils.general import *


def make_variables():
    global weights
    global source
    global output
    global prev_in
    global prev_out
    global img_size
    global conf_thres
    global iou_thres
    global device
    global view_img
    global save_txt
    global agnostic_nms
    global cfg
    global names

    weights = args.weights
    weights = weights[0]
    source = args.source
    output = args.output
    prev_in = args.prev_in
    prev_out = args.prev_out
    img_size = args.img_size
    conf_thres = args.conf_thres
    iou_thres = args.iou_thres
    device = args.device
    view_img = args.view_img
    save_txt = args.save_txt
    agnostic_nms = args.agnostic_nms
    cfg = args.cfg
    names = args.names

    #correct and handle directory paths
    if os.path.isdir(source):
        correct_dirs = resolve_directories(source, output, prev_in, prev_out)
        #rename dir to usable paths
        source = correct_dirs[0]
        output = correct_dirs[1]
        prev_in = correct_dirs[2]
        prev_out = correct_dirs[3]
    else:
        correct_dirs = resolve_directories(output, prev_in, prev_out)
        output = correct_dirs[0]
        prev_in = correct_dirs[1]
        prev_out = correct_dirs[2]


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def initiate_yolo():
    print('\nInitializing model.')

    global cfg
    global weights
    global names
    global device
    global model

    # Initialize
    if device == 'cuda':
        device = '0'
    device = select_device(device)
    if os.path.exists(output):
        print('Clearing output folder in previous outputs.\n')
        torch.cuda.empty_cache()
        for item in os.listdir(output):
            shutil.copy2(output + item, prev_out + item)
            os.remove(output + item)


    else:
        os.makedirs(output)  # make new output folder

    # Load model
    model = Darknet(cfg, img_size).cuda()

    try:
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    except:
        load_darknet_weights(model, weights)

    model.to(device).eval()

    # Get names and colors
    names = load_classes(names)

    print('Initialization finished!')


def detect(save_img=False):
    torch.cuda.empty_cache()

    global source, vid_writer
    global output
    global prev_in
    global prev_out


    global img_size

    global view_img
    global save_txt

    global conf_thres
    global iou_thres
    global agnostic_nms


    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt') or source.startswith('udp')
    if source.startswith('udp'):
        source=source+'?overrun_nonfatal=1&fifo_size=50000000'

    # Set Dataloader
    with HiddenPrints():
        torch.cuda.empty_cache()
        vid_path, vid_writer = None, None
        if webcam:
            view_img = True
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=img_size)
        else:
            save_img = True
            dataset = LoadImages(source, img_size=img_size)

        # Run inference
        t0 = time.time()

        img = torch.zeros((1, 3, img_size, img_size), device=device)  # init img

        _ = model(img) if device.type != 'cpu' else None  # run once
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    if not webcam:
        cap = cv2.VideoCapture(os.path.abspath(str(source+video_path)))
        totalframecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        current_frame = 1

    print('Running object detection and building output...')
    for path, img, im0s, vid_cap in dataset:
        try:
            current_prog = (current_frame/totalframecount)*100
            print()
            print("progress: {}/100".format(math.ceil(current_prog)))
            current_frame+=1
        except:
            pass
        with HiddenPrints():
            torch.cuda.empty_cache()

            img = torch.from_numpy(img).to(device)
            img = img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=False)[0]

            # Apply NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, agnostic=agnostic_nms)
            t2 = time_synchronized()

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                torch.cuda.empty_cache()
                if webcam:  # batch_size >= 1
                    p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
                else:
                    p, s, im0 = path, '', im0s

                save_path = str(Path(output) / Path(p).name)
                txt_path = str(Path(output) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].detach().unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, names[int(c)])  # add to string

                    # Write results
                    for *xyxy, conf, cls in det:
                        torch.cuda.empty_cache()
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh

                            with open(txt_path + '.txt', 'a') as f:
                                line = ('%g ' * 5 + '- {} \n').format(names[int(cls)]) % (cls, *xywh)
                                f.write(line)  # label format
                            del xywh, f

                        if save_img or view_img:  # Add bbox to image
                            label = '%s %.2f' % (names[int(cls)], conf)
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                    gc.collect()
                    del n, c, xyxy, conf, cls

                # Print time (inference + NMS)
                print('%sDone. (%.3fs)' % (s, t2 - t1))

                if webcam:
                    torch.cuda.empty_cache()
                    gc.collect()

                # Stream results
                if view_img:
                    cv2.imshow(p, im0)
                    if cv2.waitKey(1) == ord('q'):  # q to quit
                        raise StopIteration

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'images':
                        cv2.imwrite(save_path, im0)
                    else:
                        if vid_path != save_path:  # new video
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # release previous video writer

                            fourcc = 'mp4v'  # output video codec
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                        vid_writer.write(im0)


#           #delete temp vars to help clear device cache
                gc.collect()
                torch.cuda.empty_cache()
                del det, p, s, gn, im0, im0s, vid_cap, pred, img

    if save_txt or save_img:
        print('Results saved to %s' % Path(output))

    print('Done. (%.3fs)' % (time.time() - t0))

    print("progress: {}/100".format(100))

    print('\nMoving inputs to previous input folder.')
    def stubborn_rm(filepath,count):
        try:
            if count<10:
                os.remove(filepath)
            else:
                subprocess.Popen(['del', '/f', filepath])
        except:
            print("Couldn't remove input, trying again in 2 sec...")
            count+=1
            time.sleep(2)
            stubborn_rm(filepath, count)

    time.sleep(2)
    count=0
    for item in os.listdir(source):
        shutil.copy2(source + item, prev_in + item)
        stubborn_rm(source + item, count)

    print('\nDone.')




# Handles events of a directory
class obj_Handler(FileSystemEventHandler):
    @staticmethod
    def on_any_event(event):

        if event.is_directory:
            print(event.src_path)
            return None

        elif event.event_type == 'created':
            if os.listdir(source)!=[]:
                print('\nInput recieved. Running object detection in 5 sec...')
                time.sleep(5)
                with torch.no_grad():
                    global video_path
                    video_path = event.src_path
                    detect()



# Watches for events
class obj_Watcher():
    def __init__(self, watched_dir):
        self.observer = Observer()
        self.watched_dir = watched_dir

    def run(self):
        event_handler = obj_Handler()
        self.observer.schedule(event_handler, self.watched_dir, recursive=True)
        self.observer.start()
        try:
            while True:
                time.sleep(5)
        except:
            self.observer.stop()
            print("Stopped!")

        self.observer.join()


def check_existing_obj():
    if not os.path.isdir(source):
        detect()
    while os.listdir(source)!=[]:
        print('\nExisting files found.')
        global video_path
        video_path = os.listdir(source)[0]
        detect()


def run_object_detection(opt):
    global args
    args = opt
    make_variables()
    initiate_yolo()
    check_existing_obj()
    print('Ready for object detection.')
    w = obj_Watcher(source)
    w.run()

