# Object Detection using YOLOv8 Video Analysis

This project is a Streamlit application that performs object detection on a video file using the YOLOv8 model. The detected objects are analyzed, and a story is generated from the detected information using a language model from HuggingFace.

## Features

- Upload a video file in various formats (e.g., MP4, AVI, MOV).
- Perform object detection on the uploaded video using the YOLOv8 model.
- Extract and display the detected classes for each frame.
- Generate a story from the detected information using a language model from HuggingFace.
- Display the video and annotated frames within the Streamlit application.

## Installation

1. Install the necessary packages:
    ```bash
    pip install streamlit pyngrok opencv-python-headless ultralytics transformers langchain-huggingface huggingface_hub accelerate bitsandbytes langchain langchain_community
    ```

2. Save the application code to `app.py`:
    ```python
    %%writefile app.py
    import streamlit as st
    import cv2
    import os
    import json
    from google.colab.patches import cv2_imshow
    from ultralytics import YOLO
    from langchain import PromptTemplate, LLMChain
    from langchain.llms import HuggingFaceEndpoint

    # Function to process video and extract class information
    def process_video(video_path, output_folder):
        # Load the YOLOv8 model
        model = YOLO("yolov8n.pt")

        # Open the video file
        cap = cv2.VideoCapture(video_path)

        # Check if the video file opened successfully
        if not cap.isOpened():
            st.error("Error: Could not open video file.")
            return None

        # Get the video properties
        fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_seconds = total_frames / fps
        interval = 1  # Interval in seconds

        frame_index = 0
        max_frames = int(duration_seconds)  # Number of frames to process

        # Create a dictionary to store the detected information
        detected_info_dict = {}

        # Loop through the video frames at fixed intervals
        while cap.isOpened() and frame_index < max_frames:
            # Set the position of the next frame to read
            frame_position = int(fps * frame_index * interval)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_position)

            # Read a frame from the video
            success, frame = cap.read()

            if success:
                # Run YOLOv8 tracking on the frame, persisting tracks between frames
                results = model.track(frame, persist=True)

                # Extract detection results
                detections = results[0].boxes
                detected_info = []

                for detection in detections:
                    class_id = int(detection.cls)
                    class_name = results[0].names[class_id]  # Use class names from YOLO output

                    # Ensure proper extraction of confidence and coordinates
                    confidence = detection.conf.item()  # Convert to Python float
                    coordinates = detection.xyxy.cpu().numpy().tolist()  # Convert to list

                    detected_info.append({
                        "class": class_name,
                        "confidence": confidence,
                        "coordinates": coordinates
                    })

                # Store the detected information in the dictionary
                detected_info_dict[f"frame_{frame_index}"] = detected_info

                # Visualize the results on the frame
                annotated_frame = results[0].plot()

                # Display the annotated frame in Google Colab
                cv2_imshow(annotated_frame)

                frame_index += 1
            else:
                # Break the loop if the end of the video is reached
                break

        # Release the video capture object
        cap.release()

        # Save the detected information dictionary to a file
        with open(os.path.join(output_folder, "detected_info.json"), "w") as file:
            json.dump(detected_info_dict, file, indent=2)

        return detected_info_dict

    # Function to extract classes from detected information
    def extract_classes(detected_info_dict):
        class_dict = {}
        frame_count = 0  # Initialize frame count

        for frame_number, detected_objects in detected_info_dict.items():
            frame_count += 1  # Increment frame count
            # Extract class names for the current frame
            class_names = [obj['class'] for obj in detected_objects]
            class_dict[frame_number] = class_names  # Store in the dictionary

        return class_dict, frame_count

    # Function to generate a story from detected information
    def generate_story(classes):
        import os
        from langchain_huggingface import HuggingFaceEndpoint

        sec_key  = "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"
        os.environ["HUGGINGFACEHUB_API_TOKEN"]=sec_key

        # Initialize the language model
        repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
        llm = HuggingFaceEndpoint(repo_id=repo_id, max_length=128, temperature=0.7, token=sec_key)

        # Prepare the frames for the prompt
        frames = "\n".join([f"{key}: {', '.join(value)}" for key, value in classes.items()])

        template = """Given the following frames, which are snapshots taken per second from a video, create a story that describes the scene and what is happening in the video:

    {frames}

    The story should incorporate all the elements from each frame in a coherent narrative, assuming the video is {frames} seconds long.
    """

        # Create the prompt with the formatted frames
        prompt = PromptTemplate(template=template, input_variables=["frames"])
        formatted_prompt = prompt.format(frames=frames)



        # Create the LLM chain
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        story = llm_chain.run({"frames": frames})

        return story

    # Streamlit app
    st.title("Object Detection using YOLOv8 Video Analysis")

    # Upload video file
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        # Create temporary directories for input and output
        input_folder = "temp_input"
        output_folder = "temp_output"
        os.makedirs(input_folder, exist_ok=True)
        os.makedirs(output_folder, exist_ok=True)
        
        # Save uploaded video to temporary file
        video_path = os.path.join(input_folder, uploaded_file.name)
        with open(video_path, "wb") as f:
            f.write(uploaded_file.read())
        
        # Display the video using Streamlit's video player
        st.video(video_path)

        # Process video when submit button is clicked
        if st.button("Submit"):
            st.write("Processing video...")
            detected_info_dict = process_video(video_path, output_folder)
            
            if detected_info_dict is not None:
                class_values, frame_count = extract_classes(detected_info_dict)
                st.write(f"Total number of frames: {frame_count}")
                st.write("List of detected classes:")
                st.json(class_values)
                st.subheader("Generated Story:")
                story = generate_story(class_values)
                st.write(story)
            else:
                st.error("Error processing the video.")


    from pyngrok import ngrok

    # Set authentication token if you haven't already done so
    ngrok.set_auth_token("2jqxm0w9Pr8ShCaQJModCkc9nfR_5ZhZdLfz7aYcRV8GSzYoY")

    # Start Streamlit server on a specific port
    !nohup streamlit run app.py --server.port 5011 &

    # Start ngrok tunnel to expose the Streamlit server
    ngrok_tunnel = ngrok.connect(addr='5011', proto='http', bind_tls=True)

    # Print the URL of the ngrok tunnel
    print(' * Tunnel URL:', ngrok_tunnel.public_url)
    ```

## Usage

1. Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```

2. The application will start on a local server. If you are using Google Colab, the server will be exposed using Ngrok, and the public URL will be printed in the console.

3. Open the provided URL in your browser to access the application.

4. Upload a video file and click "Submit" to start the object detection and story generation process.

## Requirements

- Python 3.x
- Streamlit
- Pyngrok
- OpenCV
- YOLOv8 model
- HuggingFace API Token

## Notes

- Ensure you have a valid HuggingFace API token and update the `sec_key` variable in the code with your token.
- The application uses the YOLOv8n model, which should be downloaded and placed in the appropriate directory.
- The application is designed to run in a Google Colab environment, and the `cv2_imshow` function is used to display images. If running locally, you might need to adjust the display logic accordingly.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
