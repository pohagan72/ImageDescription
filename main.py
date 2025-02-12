import os
import time
import json
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, render_template, request, redirect, url_for, flash
from PIL import Image, UnidentifiedImageError
import ollama

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Configure logging
logging.basicConfig(
    filename=os.path.join(os.path.dirname(__file__), 'processing.log'),
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Define file paths relative to the script's directory
SCRIPT_DIR = os.path.dirname(__file__)
CHECKPOINT_FILE = os.path.join(SCRIPT_DIR, 'processed_images.json')
OUTPUT_FILE = os.path.join(SCRIPT_DIR, 'Output_Descriptions.txt')

# Rest of your existing functions remain the same
def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
            return set(json.load(f))
    return set()

def save_checkpoint(processed_set):
    with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
        json.dump(list(processed_set), f)

def write_output(line):
    with open(OUTPUT_FILE, 'a', encoding='utf-8') as outfile:
        outfile.write(line + "\n")

def process_single_image(image_path, max_retries=3, timeout=30):
    retries = 0
    while retries < max_retries:
        try:
            with Image.open(image_path) as img:
                img.verify()
            description = get_description(image_path, timeout)
            return {"status": "success", "description": description}
        except (IOError, UnidentifiedImageError) as e:
            logging.error(f"IOError processing image {image_path}: {str(e)}")
            return {"status": "error", "message": f"IOError: {str(e)}"}
        except ollama.ResponseError as e:
            logging.error(f"Ollama ResponseError for image {image_path}: {str(e)}")
            return {"status": "error", "message": f"Ollama ResponseError: {str(e)}"}
        except Exception as e:
            logging.error(f"Unexpected error for image {image_path}: {str(e)}\n{traceback.format_exc()}")
            retries += 1
            if retries < max_retries:
                time.sleep(2)
            else:
                return {"status": "error", "message": f"Unexpected error: {str(e)}"}

def get_description(image_path, timeout=30):
    res = ollama.chat(
        model="llama3.2-vision:11b",
        messages=[
            {
                'role': 'user',
                'content': 'Please generate a description of no more than 5 sentences of this image.',
                'images': [image_path]
            }
        ],
        timeout=timeout
    )
    return res['message']['content']

def process_images(folder_path):
    images = [file for file in os.listdir(folder_path) if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'))]
    total_images = len(images)
    processed_images = load_checkpoint()
    images_to_process = [img for img in images if img not in processed_images]
    remaining = len(images_to_process)

    if remaining == 0:
        flash("All images have already been processed.", "info")
        return

    start_time = time.time()
    total_time = 0

    if not os.path.exists(OUTPUT_FILE):
        open(OUTPUT_FILE, 'w', encoding='utf-8').close()

    max_workers = min(10, os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_image = {
            executor.submit(process_single_image, os.path.join(folder_path, image)): image
            for image in images_to_process
        }

        for count, future in enumerate(as_completed(future_to_image), start=1):
            image = future_to_image[future]
            try:
                result = future.result()
                if result["status"] == "success":
                    line = f"Processing image: {image}\nDescription: {result['description']}\n--------------"
                    write_output(line)
                    processed_images.add(image)
                else:
                    error_message = f"Error processing image: {image}. Error: {result['message']}"
                    logging.error(error_message)
                    line = f"{error_message}\n--------------"
                    write_output(line)
            except Exception as e:
                error_message = f"Unhandled exception for image {image}: {str(e)}"
                logging.error(f"{error_message}\n{traceback.format_exc()}")
                line = f"Unhandled exception for image: {image}. Error: {str(e)}\n--------------"
                write_output(line)

            current_time = time.time()
            elapsed = current_time - start_time
            total_time += elapsed
            start_time = current_time
            avg_time = total_time / count

            if count % 100 == 0:
                save_checkpoint(processed_images)
                logging.info(f"Checkpoint saved at {count} images processed.")

    save_checkpoint(processed_images)
    flash("Image descriptions have been generated and saved to Output_Descriptions.txt", "success")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        folder_path = request.form['folder_path']
        if folder_path and os.path.isdir(folder_path):
            process_images(folder_path)
        else:
            flash("Please enter a valid folder path.", "error")
        return redirect(url_for('index'))
    return render_template('index.html')

# Ensure the app runs only when executed directly
if __name__ == '__main__':
    # Add explicit host and port configuration
    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000,
        use_reloader=False,  # Disable reloader in debug mode
        threaded=True        # Enable threaded mode for better handling
    )