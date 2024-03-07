from flask import Flask, request, jsonify , send_file
from moviepy.editor import VideoFileClip
from controllers.utils import process_video
app = Flask(__name__)
def convert_video_to_mp4(video_path):
    # Load the video clip
    clip = VideoFileClip(video_path)
    
    # Define the output path for the MP4 file
    output_path = 'output.mp4'
    
    # Write the video clip to the output file in MP4 format
    clip.write_videofile(output_path, codec='libx264')
    
    # Close the clip
    clip.close()
    
    return output_path
@app.route('/video', methods=['POST'])
async def video():
    print("request.form")
    print(request.files)
    video = request.files['video']
    video.save('video.mp4')
    
    process_video()
    return {"message": "Video processed successfully"}



@app.route('/video', methods=['GET'])
def getVideo():
    print("Hello")
    video_path = 'svr.avi'
    mp4_path = convert_video_to_mp4(video_path)
    return send_file(mp4_path, as_attachment=True, mimetype='video/mp4')


if __name__ == '__main__':
    app.run(debug=True) 
