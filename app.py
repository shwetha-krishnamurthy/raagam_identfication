from flask import Flask, request, render_template, jsonify
import predict

app = Flask(__name__, template_folder="template")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'audio' in request.files:
        audio_file = request.files['audio']
        audio_file.save("received_audio.mp3")
        
        result = predict.process_audio("received_audio.mp3")
        
        return jsonify(result)  # Convert list to JSON and return

    return "No audio received"

if __name__ == '__main__':
    app.run(debug=True)
