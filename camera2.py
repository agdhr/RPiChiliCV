import cv2, os, csv, time
from datetime import datetime
import chilicv
import numpy as np
import pandas as pd

class ImageCapture:
    def __init__(self, temperature, minute, camera_index=1):
        self.temperature = temperature
        self.minute = minute
        self.camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)  
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 2592) 
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1944)

        self.save_dir = f"images/chili_{self.temperature}_{self.minute}"
        os.makedirs(self.save_dir, exist_ok=True)

        self.csv_file = os.path.join(self.save_dir, "capture_log.csv")
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "File Path"])

    def capture_image(self):
        ret, frame = self.camera.read()
        if not ret:
            print("Error: Could not read from webcam.")
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'chili_{self.temperature}_{timestamp}.TIFF'
        file_path = os.path.join(self.save_dir, filename)

        cv2.imwrite(file_path, frame)
        print(f"Image captured and saved as '{file_path}'.")

        with open(self.csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, file_path])

        return file_path

    def process_image(self, file_path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'chili_{self.temperature}_{timestamp}.TIFF'
        img = chilicv.readimage(file_path)

        img = cv2.GaussianBlur(img, (5, 5), 0)
        b = chilicv.rgb2gray_lab(img, 'b')
        tri_img = chilicv.triangle(b, 'light', 35)
        b_fill = chilicv.fill(tri_img, 100)
        masked = chilicv.apply_mask(img, b_fill, 'white')

        masked_file_path = os.path.join(self.save_dir, f'{filename}_masked.TIFF')
        cv2.imwrite(masked_file_path, masked)
        print(f"Masked image saved as '{masked_file_path}'.")

        masked_rgb = cv2.cvtColor(masked, cv2.COLOR_BGR2RGB)
        background_color = np.array([255, 255, 255])
        non_background_pixels = np.any(masked_rgb != background_color, axis=-1)
        non_background_values = masked_rgb[non_background_pixels]
        mean_rgb = np.mean(non_background_values, axis=0)
        print(f"Mean RGB values (R, G, B): {mean_rgb}")

        mean_L, mean_a, mean_b = chilicv.rgb_to_lab(mean_rgb[0], mean_rgb[1], mean_rgb[2])
        print(f"Mean LAB values (L, a, b): {mean_L}, {mean_a}, {mean_b}")

        lab_csv_file = os.path.join(self.save_dir, "lab_values.csv")
        with open(lab_csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, mean_rgb[0], mean_rgb[1], mean_rgb[2], mean_L, mean_a, mean_b])
        print(f"Logged LAB values: {timestamp}, {mean_L}, {mean_a}, {mean_b}")

    def release(self):
        self.camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    temperature = 50
    minute = 30

    image_capture = ImageCapture(temperature, minute)
    try:
        capture_count = 0
        max_capture = 3
        while capture_count < max_capture:
            file_path = image_capture.capture_image()
            if file_path:
                image_capture.process_image(file_path)
                capture_count += 1
                print(f"Capture {capture_count}/{max_capture} at temp {temperature} minute {minute}")
            time.sleep(15)  # wait 15 seconds between captures
        print(f"Capture stopped automatically after {max_capture} images.")
    except KeyboardInterrupt:
        print("Capture stopped by user.")
    finally:
        image_capture.release()


# import subprocess
# import time
# import schedule

# def job():
#     # Replace 'gdrive' with your rclone remote name 
#     # and local/path/to/data with your source folder
#     source = "/local/path/to/data"
#     destination = "gdrive:backup_folder"
    
#     try:
#         print("Starting rclone sync...")
#         result = subprocess.run(
#             ["rclone", "copy", source, destination, "--progress"],
#             check=True,
#             text=True,
#             capture_output=True
#         )
#         print("Backup complete:", result.stdout)
#     except subprocess.CalledProcessError as e:
#         print("Backup failed:", e.stderr)

# # Schedule the job to run every day at a specific time or interval
# schedule.every().day.at("02:00").do(job)
# # Or run every X hours: schedule.every(2).hours.do(job)

# print("Backup scheduler is running...")
# while True:
#     schedule.run_pending()
#     time.sleep(1)
