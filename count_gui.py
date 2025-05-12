import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from ultralytics import YOLO, solutions
import threading

# Initialize YOLO model
model = YOLO("drone_f.pt")  # Ensure you specify the correct model path
names = model.model.names

# Global variables
cap = None
video_writer = None
video_path = ""
output_path = "object_counting_output.avi"

# Define line points for object counting
line_points = [(0, 200), (400, 200)]

# Initialize Object Counter
counter = solutions.ObjectCounter(
    view_img=True,
    reg_pts=line_points,
    classes_names=model.names,
    draw_tracks=True,
    line_thickness=1
)

# Function to upload video
def upload_video():
    global video_path
    video_path = filedialog.askopenfilename(filetypes=[("MP4 Files", "*.mp4"), ("All Files", "*.*")])
    if video_path:
        messagebox.showinfo("Success", f"Video '{video_path}' loaded successfully!")
    else:
        messagebox.showerror("Error", "Failed to load video.")

# Function to process the video with object counting
def process_video():
    global cap, video_writer, video_path

    if not video_path:
        messagebox.showerror("Error", "Please upload a video first.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open video.")
        return

    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

    # Initialize the video writer
    video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    # Process video frames
    def track_and_count_objects():
        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
                print("Video frame is empty or video processing has been successfully completed.")
                break

            # Run YOLO object tracking
            tracks = model.track(im0, persist=True, show=False)

            # Start object counting
            im0 = counter.start_counting(im0, tracks)

            # Write the processed frame to the output video
            video_writer.write(im0)

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()

    # Run video processing in a separate thread to avoid blocking the GUI
    threading.Thread(target=track_and_count_objects, daemon=True).start()

    # Display message once the processing starts
    messagebox.showinfo("Processing", "Object counting started. The video will be saved as 'object_counting_output.avi'.")

# Function to display the processed video
def show_processed_video():
    # Open the output video file
    output_cap = cv2.VideoCapture(output_path)
    if not output_cap.isOpened():
        messagebox.showerror("Error", "Could not open the processed video.")
        return

    while True:
        ret, frame = output_cap.read()
        if not ret:
            break

        # Display the processed frame
        cv2.imshow("Processed Video with Object Counting", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    output_cap.release()
    cv2.destroyAllWindows()

# Create the main window
root = tk.Tk()
root.title("Deep Learning Based Drone Object Counting System")
root.geometry("600x400")
root.config(bg="#f0f0f0")

# Add a title label
title_label = tk.Label(root, text="Deep Learning Based Drone Object Counting System", font=("Helvetica", 16, "bold"), bg="#f0f0f0")
title_label.pack(pady=20)

# Create buttons
upload_button = tk.Button(root, text="Upload Video", command=upload_video, font=("Helvetica", 12), bg="#4CAF50", fg="white", relief="raised", width=20)
upload_button.pack(pady=10)

track_button = tk.Button(root, text="Start Counting Objects", command=process_video, font=("Helvetica", 12), bg="#008CBA", fg="white", relief="raised", width=20)
track_button.pack(pady=10)

# show_button = tk.Button(root, text="Show Processed Video", command=show_processed_video, font=("Helvetica", 12), bg="#FF5722", fg="white", relief="raised", width=20)
# show_button.pack(pady=10)

# Start the GUI loop
root.mainloop()
