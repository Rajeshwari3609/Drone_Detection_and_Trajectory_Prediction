import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import threading

# Load YOLOv8 model
model = YOLO("drone.pt")  # Replace with your model path

# Create the main window
root = tk.Tk()
root.title("Deep Learning Based Drone Tracking System")
root.geometry("600x400")  # Set a fixed size for the window
root.config(bg="#f0f0f0")  # Set background color

# Center the title at the top
title_label = tk.Label(root, text="Deep Learning Based Drone Tracking System", font=("Helvetica", 16, "bold"), bg="#f0f0f0")
title_label.pack(pady=20)

# Global variable for video path and video capture
video_path = ""
cap = None
out = None

# Store the track history
track_history = defaultdict(lambda: [])
future_frames = 10

# Function to upload the video file
def upload_video():
    global video_path
    video_path = filedialog.askopenfilename(filetypes=[("MP4 Files", "*.mp4"), ("All Files", "*.*")])
    if video_path:
        messagebox.showinfo("Success", f"Video '{video_path}' loaded successfully!")
    else:
        messagebox.showerror("Error", "Failed to load video.")

# Function to process and track the video
def process_video():
    global cap, out

    if not video_path:
        messagebox.showerror("Error", "Please upload a video first.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open video.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 file
    out = cv2.VideoWriter('output_drone_tracking.mp4', fourcc, fps, (frame_width, frame_height))

    # Process video frames in a separate thread to keep the GUI responsive
    def track():
        while cap.isOpened():
            success, frame = cap.read()
            if success:
                results = model.track(frame, persist=True)

                if results[0].boxes.id is not None:
                    boxes = results[0].boxes.xywh.cpu()
                    track_ids = results[0].boxes.id.int().cpu().tolist()

                    annotated_frame = results[0].plot()

                    # Plot the tracks and predict future path
                    for box, track_id in zip(boxes, track_ids):
                        x, y, w, h = box
                        track = track_history[track_id]
                        track.append((float(x), float(y)))  # x, y center point
                        if len(track) > 30:  # retain 30 tracks for 30 frames
                            track.pop(0)

                        points = np.array(track).astype(np.int32).reshape((-1, 1, 2))
                        cv2.polylines(annotated_frame, [points], isClosed=False, color=(25, 150, 230), thickness=2)

                        # Predict future positions
                        if len(track) > 1:
                            times = np.arange(len(track))
                            x_coords = np.array([p[0] for p in track])
                            y_coords = np.array([p[1] for p in track])

                            poly_x = np.polyfit(times, x_coords, 1)
                            poly_y = np.polyfit(times, y_coords, 1)

                            future_points = []
                            for i in range(1, future_frames + 1):
                                next_time = len(track) + i
                                pred_x = np.polyval(poly_x, next_time)
                                pred_y = np.polyval(poly_y, next_time)
                                future_points.append((pred_x, pred_y))

                            for j in range(len(future_points) - 1):
                                cv2.line(annotated_frame,
                                         (int(future_points[j][0]), int(future_points[j][1])),
                                         (int(future_points[j + 1][0]), int(future_points[j + 1][1])),
                                         color=(0, 0, 255),
                                         thickness=2)

                    out.write(annotated_frame)
                    cv2.imshow("YOLOv8 Tracking", annotated_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

    # Start video processing in a separate thread to keep the GUI responsive
    threading.Thread(target=track, daemon=True).start()

# Create the buttons for upload and process
upload_button = tk.Button(root, text="Upload Video", command=upload_video, font=("Helvetica", 12), bg="#4CAF50", fg="white", relief="raised", width=20)
upload_button.pack(pady=10)

track_button = tk.Button(root, text="Track", command=process_video, font=("Helvetica", 12), bg="#008CBA", fg="white", relief="raised", width=20)
track_button.pack(pady=20)

# Run the GUI loop
root.mainloop()
