import os
import streamlit as st
import ollama
from PIL import Image, UnidentifiedImageError
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback
import logging
import json

# Configure logging to record processing details and errors
logging.basicConfig(
    filename='processing.log',  # Log file name
    filemode='a',                # Append mode
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log message format
    level=logging.INFO           # Log level
)

# File to keep track of processed images to enable checkpointing
CHECKPOINT_FILE = 'processed_images.json'

def load_checkpoint():
    """
    Load the set of already processed images from the checkpoint file.
    
    Returns:
        set: A set containing filenames of processed images.
    """
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
            return set(json.load(f))  # Load JSON list and convert to set for faster lookups
    return set()

def save_checkpoint(processed_set):
    """
    Save the set of processed images to the checkpoint file.
    
    Args:
        processed_set (set): The set of processed image filenames.
    """
    with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
        json.dump(list(processed_set), f)  # Convert set to list for JSON serialization

def write_output(line):
    """
    Append a line of text to the output file incrementally.
    
    Args:
        line (str): The line of text to append.
    """
    with open('Q3_Descriptions.txt', 'a', encoding='utf-8') as outfile:
        outfile.write(line + "\n")  # Write the line followed by a newline character

def process_single_image(image_path, max_retries=3, timeout=30):
    """
    Process a single image by generating its description using the Ollama API.
    
    Args:
        image_path (str): The full path to the image file.
        max_retries (int, optional): Maximum number of retries for unexpected errors. Defaults to 3.
        timeout (int, optional): Timeout in seconds for the Ollama API call. Defaults to 30.
    
    Returns:
        dict: A dictionary containing the status and either the description or an error message.
    """
    retries = 0
    while retries < max_retries:
        try:
            # Open and verify the image to ensure it's not corrupted
            with Image.open(image_path) as img:
                img.verify()

            # Generate description for the image
            description = get_description(image_path, timeout)
            return {"status": "success", "description": description}

        except (IOError, UnidentifiedImageError) as e:
            # Handle image-related errors
            logging.error(f"IOError processing image {image_path}: {str(e)}")
            return {"status": "error", "message": f"IOError: {str(e)}"}

        except ollama.ResponseError as e:
            # Handle specific Ollama API response errors
            logging.error(f"Ollama ResponseError for image {image_path}: {str(e)}")
            return {"status": "error", "message": f"Ollama ResponseError: {str(e)}"}

        except Exception as e:
            # Handle any other unexpected exceptions
            logging.error(f"Unexpected error for image {image_path}: {str(e)}\n{traceback.format_exc()}")
            retries += 1
            if retries < max_retries:
                time.sleep(2)  # Wait before retrying
            else:
                return {"status": "error", "message": f"Unexpected error: {str(e)}"}

def get_description(image_path, timeout=30):
    """
    Generate a description for an image using the Ollama API.
    
    Args:
        image_path (str): The full path to the image file.
        timeout (int, optional): Timeout in seconds for the API call. Defaults to 30.
    
    Returns:
        str: The generated description of the image.
    
    Raises:
        ollama.ResponseError: If the Ollama API returns an error.
    """
    # Make a chat request to Ollama with the image and prompt
    res = ollama.chat(
        model="llava:latest",  # Specify the model to use
        messages=[
            {
                'role': 'user',
                'content': 'Please generate a description of no more than 5 sentences of this image.',
                'images': [image_path]  # Attach the image to the message
            }
        ],
        timeout=timeout  # Set the timeout for the API call
    )
    return res['message']['content']  # Extract the description from the response

def process_images(folder_path):
    """
    Process all images in a specified folder by generating descriptions using the Ollama API.
    
    Args:
        folder_path (str): The path to the folder containing images.
    """
    # List all image files with supported extensions in the folder
    images = [file for file in os.listdir(folder_path) if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'))]
    total_images = len(images)
    
    # Load the set of already processed images to avoid re-processing
    processed_images = load_checkpoint()
    images_to_process = [img for img in images if img not in processed_images]
    remaining = len(images_to_process)

    if remaining == 0:
        # Inform the user if there are no new images to process
        st.info("All images have already been processed.")
        return

    # Initialize Streamlit UI components for progress tracking
    progress_bar = st.progress(0)           # Progress bar
    status_text = st.empty()                # Status text
    total_time_text = st.empty()            # Total processing time
    avg_time_text = st.empty()              # Average time per image

    start_time = time.time()  # Record the start time for processing
    total_time = 0            # Initialize total processing time

    # Initialize the output file if it doesn't exist
    if not os.path.exists('Q3_Descriptions.txt'):
        open('Q3_Descriptions.txt', 'w', encoding='utf-8').close()

    # Determine the number of worker threads based on the system's CPU count
    max_workers = min(10, os.cpu_count() or 1)  # Limit to a maximum of 10 workers
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all image processing tasks to the thread pool
        future_to_image = {
            executor.submit(process_single_image, os.path.join(folder_path, image)): image
            for image in images_to_process
        }

        # Iterate over completed futures as they finish
        for count, future in enumerate(as_completed(future_to_image), start=1):
            image = future_to_image[future]  # Get the image associated with the future
            try:
                result = future.result()  # Retrieve the result of the processing

                if result["status"] == "success":
                    # If processing was successful, write the description to the output file
                    line = f"Processing image: {image}\nDescription: {result['description']}\n--------------"
                    write_output(line)
                    processed_images.add(image)  # Mark the image as processed

                else:
                    # If there was an error, log it and write the error message to the output file
                    error_message = f"Error processing image: {image}. Error: {result['message']}"
                    logging.error(error_message)
                    line = f"{error_message}\n--------------"
                    write_output(line)

            except Exception as e:
                # Handle any unexpected exceptions during future result retrieval
                error_message = f"Unhandled exception for image {image}: {str(e)}"
                logging.error(f"{error_message}\n{traceback.format_exc()}")
                line = f"Unhandled exception for image: {image}. Error: {str(e)}\n--------------"
                write_output(line)

            # Update the progress bar based on the number of processed images
            progress = count / remaining
            progress_bar.progress(progress)
            status_text.text(f"Processed {count} of {remaining} files.")

            # Update timing information
            current_time = time.time()
            elapsed = current_time - start_time
            total_time += elapsed
            start_time = current_time
            avg_time = total_time / count
            total_time_text.text(f"Total processing time: {int(total_time)} seconds")
            avg_time_text.text(f"Average processing time per image: {int(avg_time)} seconds")

            # Periodically save a checkpoint every 100 images processed
            if count % 100 == 0:
                save_checkpoint(processed_images)
                logging.info(f"Checkpoint saved at {count} images processed.")

    # Save the final checkpoint after all images have been processed
    save_checkpoint(processed_images)
    st.success("Image descriptions have been generated and saved to Q3_Descriptions.txt")

# Streamlit User Interface Configuration

# Set the title of the Streamlit app
st.title("Q3 Image Description Generator")

# Provide a brief description of the application
st.markdown("""
This application processes images in a specified folder and generates descriptions using the Ollama API. 
It supports resuming from the last checkpoint in case of interruptions.
""")

# Input field for the user to specify the folder containing images
folder_path = st.text_input("Enter the path to the folder containing images:")

# Button to initiate the image processing
if st.button("Process Images"):
    if folder_path and os.path.isdir(folder_path):
        process_images(folder_path)  # Start processing if the folder path is valid
    else:
        st.error("Please enter a valid folder path.")  # Show an error if the path is invalid
