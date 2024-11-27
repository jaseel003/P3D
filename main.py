from flask import Flask, request, jsonify, render_template, send_from_directory
import zipfile
import os
import cv2
import json
import pandas as pd
import numpy as np
from transformers import AutoModelForImageSegmentation
import openvino as ov
import mysql.connector
from mysql.connector import Error
from pathlib import Path
from werkzeug.utils import secure_filename
import boto3

app = Flask(__name__)

# Database connection setup
def update_database_with_urls(sku_id, p3d_urls_json):
    try:
        # Establish database connection
        conn = mysql.connector.connect(
            host=os.getenv('DB_HOST'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            database=os.getenv('DB_NAME')
        )
        cursor = conn.cursor()

        p3d_urls_json_str = json.dumps(p3d_urls_json, ensure_ascii=False)

        # Update or insert URLs and process status in the database
        query = """
        UPDATE all_platform_products 
        SET p3d_urls = %s
        WHERE sku = %s
        """
        cursor.execute(query, (p3d_urls_json_str, sku_id))

        # Commit the transaction
        conn.commit()
        print(f"Database updated for SKU {sku_id} with P3D URLs: {p3d_urls_json}")

    except Exception as e:
        print(f"Failed to update the database: {e}")
    finally:
        cursor.close()
        conn.close()

# Configure allowed file types
ALLOWED_EXTENSIONS = {'csv'}  # Only allow CSV file uploads for SKU information

# S3 configuration
S3_BUCKET_NAME = 'your-s3-bucket-name'
S3_REGION_NAME = 'your-region-name'  
S3_ACCESS_KEY = 'your-access-key'
S3_SECRET_KEY = 'your-secret-key'

# Load the OpenVINO model
model_input_size = [1024, 1024]
ov_model_path = Path("models/rmbg-1.4.xml")

# Load the segmentation model
net = AutoModelForImageSegmentation.from_pretrained("briaai/RMBG-1.4", trust_remote_code=True)

if not ov_model_path.exists():
    # Convert the model and save it
    example_input = np.zeros((1, 3, *model_input_size), dtype=np.uint8)
    ov_model = ov.convert_model(net, example_input, input=[1, 3, *model_input_size])
    ov.save_model(ov_model, ov_model_path)

core = ov.Core()
device = "AUTO"
ov_compiled_model = core.compile_model(ov_model_path, device)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def apply_bleed_effect(image_np):
    """Apply a bleed effect to the image."""
    x_scale = 20
    y_scale = 20
    alpha_channel = image_np[:, :, 3]
    kernel_size_x = max(1, x_scale)
    kernel_size_y = max(1, y_scale)
    kernel = np.ones((kernel_size_y, kernel_size_x), np.uint8)

    adjusted_mask = cv2.dilate(alpha_channel, kernel, iterations=1)
    bleeded_image = image_np.copy()
    bleeded_image[:, :, 3] = adjusted_mask
    return bleeded_image

def center_align_subject(image):
    """Center the subject within a new blank canvas."""
    alpha_channel = image[:, :, 3]
    coords = cv2.findNonZero(alpha_channel)
    x, y, w, h = cv2.boundingRect(coords)
    subject_region = image[y:y + h, x:x + w]

    centered_image = np.zeros_like(image)
    center_x = (image.shape[1] - w) // 2
    center_y = (image.shape[0] - h) // 2
    centered_image[center_y:center_y + h, center_x:center_x + w] = subject_region
    return centered_image

def add_white_background(image):
    """Add a white background to an image with a transparent background."""
    white_background = np.ones_like(image) * 255
    for c in range(3):  
        white_background[:, :, c] = image[:, :, c] * (image[:, :, 3] / 255.0) + 255 * (1 - (image[:, :, 3] / 255.0))

    return white_background    

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'csv' not in request.files:
        return jsonify({"error": "No CSV file part"}), 400

    csv_file = request.files['csv']
    seller_id = request.form.get('sellerId')
    partner_id = request.form.get('partnerId')
    num_frames = request.form.get('numFrames')

    if not seller_id or not partner_id:
        return jsonify({"error": "Seller ID and Partner ID are required"}), 400

    if not (csv_file and allowed_file(csv_file.filename)):
        return jsonify({"error": "Invalid CSV file type"}), 400

    # Save the uploaded CSV file temporarily
    csv_filename = secure_filename(csv_file.filename)
    csv_path = os.path.join('uploads', csv_filename)
    csv_file.save(csv_path)

    try:
        # Load the CSV file
        data = pd.read_csv(csv_path)
        if 'sku_id' not in data.columns or 'process_id' not in data.columns:
            return jsonify({"error": "CSV file must contain 'sku_id' and 'process_id' columns"}), 400

        # Loop through each row in the CSV
        for _, row in data.iterrows():
            sku_id = row['sku_id']
            process_id = row['process_id']

            # Construct the folder paths based on seller ID and SKU ID
            base_folder = rf"E:\pf\igo\batch_process_output\{seller_id}\{sku_id}\raw"
            video_file_path = None
            additional_csv_path = None

            # Check for video and CSV file in the raw folder
            for file_name in os.listdir(base_folder):
                if file_name.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv')):
                    video_file_path = os.path.join(base_folder, file_name)
                elif file_name.lower().endswith('.csv'):
                    additional_csv_path = os.path.join(base_folder, file_name)

            if process_id == "3D360":
                if not video_file_path or not additional_csv_path:
                    return jsonify({"error": f"Video or CSV file missing for SKU {sku_id}"}), 400

                try:
                    num_frame = 360 if num_frames == "36 x 10" else 3600
                    processed_folder = os.path.join(rf"E:\pf\igo\batch_process_output\{seller_id}\{sku_id}\processed")
                    os.makedirs(processed_folder, exist_ok=True)

                    processed_images_36, processed_images = process_video(
                        video_file_path, additional_csv_path, processed_folder,
                        file_counter=1, num_frame=num_frame, sku_id=sku_id
                    )

                    p3d_file_36 = create_p3d_file(processed_images_36, processed_folder, sku_id, "36")
                    p3d_file_360 = create_p3d_file(processed_images, processed_folder, sku_id, "360")

                    s3_file_url_36 = upload_to_s3(p3d_file_36, f"igo_python_files/3d_360/{seller_id}/{sku_id}/{sku_id}_36.p3d")
                    s3_file_url_360 = upload_to_s3(p3d_file_360, f"igo_python_files/3d_360/{seller_id}/{sku_id}/{sku_id}_360.p3d")

                    p3d_urls_json = {"p3d_36": s3_file_url_36, "p3d_360": s3_file_url_360}

                    # Update the database with URLs
                    update_database_with_urls(sku_id, p3d_urls_json)

                    print(f"Completed processing for SKU {sku_id}. P3D URLs: {p3d_urls_json}")

                except Exception as e:
                    return jsonify({"error": f"Processing failed for SKU {sku_id}: {str(e)}"}), 500

            elif process_id == "bg_elimination":
                image_file_paths = [os.path.join(base_folder, f) for f in os.listdir(base_folder)
                                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'))]

                if not image_file_paths:
                    return jsonify({"error": f"No image files found in the raw folder for SKU {sku_id} background elimination."}), 400

                processed_folder = os.path.join(rf"E:\pf\igo\batch_process_output\{seller_id}\{sku_id}\processed")
                os.makedirs(processed_folder, exist_ok=True)

                processed_images = []
                for image_path in image_file_paths:
                    try:
                        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                        if image is None:
                            print(f"Error: Failed to read image at {image_path}. Skipping.")
                            continue

                        input_image = cv2.resize(image, (model_input_size[1], model_input_size[0]))
                        input_image = input_image.transpose(2, 0, 1)
                        input_image = np.expand_dims(input_image, axis=0).astype(np.float32) / 255.0

                        result = ov_compiled_model(input_image)[0]

                        mask = result[0] > 0.5
                        mask = mask.astype(np.uint8) * 255
                        mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

                        kernel = np.ones((5, 5), np.uint8)
                        mask_resized = cv2.morphologyEx(mask_resized, cv2.MORPH_CLOSE, kernel)

                        no_bg_image = cv2.bitwise_and(image, image, mask=mask_resized)
                        no_bg_image = cv2.cvtColor(no_bg_image, cv2.COLOR_BGR2BGRA)
                        no_bg_image[:, :, 3] = mask_resized

                        centered_image = center_align_subject(no_bg_image)

                        output_filename = os.path.basename(image_path)
                        output_path = os.path.join(processed_folder, output_filename)
                        cv2.imwrite(output_path, centered_image)

                        processed_images.append(output_path)
                        print(f"Processed image saved as {output_path}")

                    except Exception as e:
                        print(f"Unexpected error while processing image {image_path}: {e}")

                print(f"Background elimination completed successfully for SKU {sku_id}.")

            elif process_id == "bg_elimination with bleed":
                image_file_paths = [os.path.join(base_folder, f) for f in os.listdir(base_folder)
                                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.jfif'))]

                if not image_file_paths:
                    return jsonify({"error": f"No image files found in the raw folder for SKU {sku_id} background elimination with bleed."}), 400

                processed_folder = os.path.join(rf"E:\pf\igo\batch_process_output\{seller_id}\{sku_id}\processed")
                os.makedirs(processed_folder, exist_ok=True)

                processed_images = []
                for image_path in image_file_paths:
                    try:
                        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                        if image is None:
                            print(f"Error: Failed to read image at {image_path}. Skipping.")
                            continue

                        input_image = cv2.resize(image, (model_input_size[1], model_input_size[0]))
                        input_image = input_image.transpose(2, 0, 1)
                        input_image = np.expand_dims(input_image, axis=0).astype(np.float32) / 255.0

                        result = ov_compiled_model(input_image)[0]

                        mask = result[0] > 0.5
                        mask = mask.astype(np.uint8) * 255
                        mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

                        kernel = np.ones((5, 5), np.uint8)
                        mask_resized = cv2.morphologyEx(mask_resized, cv2.MORPH_CLOSE, kernel)

                        no_bg_image = cv2.bitwise_and(image, image, mask=mask_resized)
                        no_bg_image = cv2.cvtColor(no_bg_image, cv2.COLOR_BGR2BGRA)
                        no_bg_image[:, :, 3] = mask_resized

                        bleeded_image = apply_bleed_effect(no_bg_image)
                        centered_image = center_align_subject(bleeded_image)

                        output_filename = os.path.basename(image_path)
                        output_path = os.path.join(processed_folder, output_filename)
                        cv2.imwrite(output_path, centered_image)

                        processed_images.append(output_path)
                        print(f"Processed image saved as {output_path}")

                    except Exception as e:
                        print(f"Unexpected error while processing image {image_path}: {e}")

                print(f"Background elimination with bleed completed successfully for SKU {sku_id}.")

            else:
                print(f"Unsupported process_id: {process_id} for SKU {sku_id}")

        print("All processes completed for all SKUs in CSV.")
        return jsonify({"message": "Batch processing completed for all SKUs in CSV."}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(csv_path):
            os.remove(csv_path)

def process_video(video_path, csv_path, save_directory, file_counter, num_frame, sku_id):
    # Load CSV data
    data = pd.read_csv(csv_path)
    num_rows = len(data)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        cap.release()
        raise ValueError("Could not determine the FPS of the video")

    processed_images = []  # List to hold paths of all processed images
    processed_images_36 = []  # List to hold paths of the first 36 processed images
    frames_per_interval = num_frame // 10

    for i in range(0, num_rows, 2):
        if i + 1 >= num_rows:
            print(f"Error: Incomplete data for pair starting at row {i}.")
            break

        start_time_ms = data.iloc[i]['time']
        end_time_ms = data.iloc[i + 1]['time']
        start_turntable_angle = data.iloc[i]['turntable_angle']
        end_turntable_angle = data.iloc[i + 1]['turntable_angle']

        start_frame = int((start_time_ms / 1000) * fps)
        end_frame = int((end_time_ms / 1000) * fps)
        total_frames_time_range = end_frame - start_frame

        angle_range = end_turntable_angle - start_turntable_angle
        angle_interval = angle_range / frames_per_interval

        for j in range(frames_per_interval):
            try:
                current_angle = start_turntable_angle + j * angle_interval
                frame_position = start_frame + int((j / frames_per_interval) * total_frames_time_range)

                if frame_position > end_frame or frame_position >= int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
                    print(f"Warning: Frame position {frame_position} exceeds valid frame range.")
                    continue

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_position)
                ret, frame = cap.read()

                if not ret or frame is None:
                    print(f"Error: Failed to read frame at position {frame_position}. Skipping.")
                    continue

                # Prepare frame for OpenVINO
                input_image = cv2.resize(frame, (model_input_size[1], model_input_size[0]))
                input_image = input_image.transpose(2, 0, 1)
                input_image = np.expand_dims(input_image, axis=0).astype(np.float32) / 255.0

                # Perform inference
                result = ov_compiled_model(input_image)[0]

                # Generate binary mask
                mask = result[0] > 0.5
                mask = mask.astype(np.uint8) * 255
                mask = np.squeeze(mask)
                mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

                # Optional: Morphological operations to improve mask
                kernel = np.ones((5, 5), np.uint8)
                mask_resized = cv2.morphologyEx(mask_resized, cv2.MORPH_CLOSE, kernel)

                # Create output image with background removed
                no_bg_image = cv2.bitwise_and(frame, frame, mask=mask_resized)
                no_bg_image = cv2.cvtColor(no_bg_image, cv2.COLOR_BGR2BGRA)
                no_bg_image[:, :, 3] = mask_resized

                white_bg_image = add_white_background(no_bg_image)
                centered_image = center_align_subject(white_bg_image)
                # bleeded_image = apply_bleed_effect(centered_image)

                # Crop to a 1:1 aspect ratio (center square)
                height, width = centered_image.shape[:2]
                if width > height:
                    offset = (width - height) // 2
                    cropped_image = white_bg_image[:, offset:offset + height]
                else:
                    offset = (height - width) // 2
                    cropped_image = white_bg_image[offset:offset + width, :]

                # Resize to 830 x 830 pixels
                final_image = cv2.resize(cropped_image, (830, 830), interpolation=cv2.INTER_AREA)

                # Save the image
                output_filename = f"image_f{file_counter}.jpg"
                output_path = os.path.join(save_directory, output_filename)
                cv2.imwrite(output_path, final_image, [cv2.IMWRITE_JPEG_QUALITY, 80])

                # Add to processed images lists
                processed_images.append(output_path)
                if file_counter <= 36:
                    processed_images_36.append(output_path)  # Save the first 36 frames separately

                print(f"Frame {file_counter} processed: Background removed and saved as {output_path}")
                file_counter += 1

            except Exception as e:
                print(f"Unexpected error while processing frame {file_counter}: {e}")

    cap.release()
    print("Frame extraction completed.")
    return processed_images_36, processed_images  # Return both lists

def create_p3d_file(processed_images, save_directory, sku_id, suffix):
    p3d_filename = f"{sku_id}_{suffix}.p3d"
    p3d_file_path = os.path.join(save_directory, p3d_filename)

    with zipfile.ZipFile(p3d_file_path, 'w') as zipf:
        for img_path in processed_images:
            img_filename = os.path.basename(img_path)
            zipf.write(img_path, img_filename)

    # # Upload the P3D file to S3
    # try:
    #     upload_to_s3(p3d_file_path, p3d_filename)  # Use the dynamically created filename for S3
    # except NameError:
    #     print("S3 upload function not defined. Skipping upload.")

    return p3d_file_path

def upload_to_s3(file_path, s3_key):
    # Initialize a session using Amazon S3
    s3_client = boto3.client(
        's3',
        region_name="us-east-1",
        aws_access_key_id="AKIAQLSIVUGCCDXXZXFC",
        aws_secret_access_key="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"
    )

    try:
        # Upload the file to the S3 bucket with specified content type
        s3_client.upload_file(
            file_path,
            "igo-media-dev",
            s3_key,
            ExtraArgs={'ContentType': 'application/zip'}  # Specify the MIME type
        )
        s3_url = f"https://igo-media-dev.s3.us-east-1.amazonaws.com/{s3_key}"
        print(f"File {file_path} uploaded to S3 as {s3_key}.")
        return s3_url

    except Exception as e:
        print(f"Failed to upload {file_path} to S3: {e}")
        return None

if __name__ == '__main__':
    app.run(debug=True)
