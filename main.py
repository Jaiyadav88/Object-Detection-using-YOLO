# import dependencies

import math
import datetime
import cv2
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog
from matplotlib.figure import Figure
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def update_frame():
    confidence = []
    ret, frame = cap.read()
    if ret:
        # displaying date and time on the window
        dt = datetime.datetime.now()
        text = str(dt)
        font = cv2.FONT_HERSHEY_TRIPLEX
        cv2.putText(frame, text, (10, 50), font, 1, (0, 255, 255), 2, cv2.LINE_AA)

        # initializing YOLO object
        result = model(frame, stream=True)
        # based of xy-xy values drawing bounding boxes
        for r in result:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (168, 50, 60), 2, cv2.LINE_AA)

                # getting confidence values
                conf = math.ceil((box.conf[0]) * 100) / 100
                # storing confidence inside the list
                confidence.append(conf)
                # getting class values
                cls = box.cls[0]
                # text along with confidence and its class
                txt = f'{conf} {classNames[int(cls)]}'
                cv2.putText(frame, txt, (max(0, x1), max(35, y1)), font, 1, (56, 78, 90), 2, cv2.LINE_AA)

                ax.clear()
                ax.bar(range(len(confidence)), confidence, color='blue')
                ax.set_title('Confidence Graph')
                ax.set_xlabel('Object Detection Index')
                ax.set_ylabel('Confidence Values')
                fig_canvas.draw()

                # converting frames to pixels and then displaying it on the label using the tkinter library
                cv2_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(cv2_img)
                imgtk = ImageTk.PhotoImage(image=img)
                label.imgtk = imgtk
                label.config(image=imgtk)
                label.after(10, update_frame)
    else:
        cap.release()
if __name__ == '__main__':
    # list of all the classes of object detection
    classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                  "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                  "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                  "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                  "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                  "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                  "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                  "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                  "teddy bear", "hair drier", "toothbrush"
                  ]
    # initializing the YOLO model
    model = YOLO('../YOLO-Weights/yolov8l.pt')

    # capture object to add the video
    cap = cv2.VideoCapture('./Video/Test Video.mp4')
    root = tk.Tk()
    root.title('Object Detection Window')

    # making label to display frames
    label = tk.Label(root)
    label.pack()


    def openvideo():
        filepath = filedialog.askopenfilename()
        cap = cv2.VideoCapture(filepath)
        update_frame()


    # creating a open button using tkinter
    open_button = tk.Button(root, text='Open Video', command=openvideo)
    open_button.pack()

    # creating an exit button
    exit_button = tk.Button(root, text='Exit', command=root.destroy)
    exit_button.pack()

    # creating a window for the graph and defining graph characteristics
    graph_tk = tk.Toplevel()
    graph_tk.title('Confidence Graph')
    fig = Figure(figsize=(5, 4), dpi=100)
    ax = fig.add_subplot(111)
    ax.set_title('Confidence Graph')
    ax.set_xlabel('Object Detection Index')
    ax.set_ylabel('Confidence score')
    fig_canvas = FigureCanvasTkAgg(fig, master=graph_tk)
    fig_widget= fig_canvas.get_tk_widget()
    fig_widget.pack()
    update_frame()
    root.mainloop()
