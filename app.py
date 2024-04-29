import os
import json
import tempfile
import time

from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin

from whisper_models.whisper_timestamped import transcribe

def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    CORS(app)
    app.config.from_mapping(
        SECRET_KEY='dev',
    )
    app.config["CORS_HEADERS"] = "Content-Type"

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    @app.route('/hello', methods=['GET'])
    def hello():
        print("Hello, World on BACKEND!")
        return json.dumps({"message": "Hello, World!"})

    @app.route('/transcribe', methods=['POST'])
    def transcribe_local():
        start_time = time.time()
        video_file = request.files.get('video_file')
        file_type = request.form.get("type")
        print(video_file)
        print(file_type)
        filename = "video." + file_type
  
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()
        
        # Construct the target path
        target_path = os.path.join(temp_dir, filename)
        
        try:
            # Save the audio file to the temporary directory
            video_file.save(target_path)
            
            # Run whisper
            start_time = time.time()
            print("Running whisper...")
            transcript = transcribe(target_path)
            whisper_stt_time = time.time() - start_time

            print("Whisper STT Time: ", whisper_stt_time)
            print("Total Time: ", time.time() - start_time)
            
            # TODO add transcript type
            return json.dumps({"transcript": transcript})
        
        except Exception as e:
            print(f"Error processing audio: {e}")
            return jsonify({"message": "Error processing audio"}), 500
        
        finally:
            # delete the temporary directory to clean up
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

    return app