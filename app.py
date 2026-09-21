from flask import Flask, request, Response
import edge_tts
import asyncio
import os
import uuid

app = Flask(__name__)

async def generate_audio(text, voice, output_file):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_file)

# Naya Health Check Route - Taki bot ko pata chale ki server zinda hai
@app.route('/', methods=['GET'])
def health_check():
    return "OK", 200

@app.route('/tts', methods=['POST'])
def tts():
    data = request.json
    if not data or 'text' not in data:
        return "Text not found", 400
    
    text = data['text']
    voice = data.get('voice', 'hi-IN-MadhurNeural')
    filename = f"temp_{uuid.uuid4().hex}.mp3"
    
    try:
        asyncio.run(generate_audio(text, voice, filename))
        
        with open(filename, 'rb') as f:
            audio_data = f.read()
            
        os.remove(filename)
        return Response(audio_data, mimetype="audio/mpeg")
    except Exception as e:
        if os.path.exists(filename):
            os.remove(filename)
        return str(e), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
