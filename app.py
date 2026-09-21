from flask import Flask, request, Response
import edge_tts
import asyncio
import os
import uuid

app = Flask(__name__)

async def generate_audio(text, voice, output_file):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_file)

@app.route('/tts', methods=['POST'])
def tts():
    data = request.json
    if not data or 'text' not in data:
        return "Text not found", 400
    
    text = data['text']
    # यदि कोई आवाज़ नहीं चुनी गई है, तो डिफ़ॉल्ट रूप से हिंदी आवाज़ का उपयोग होगा
    voice = data.get('voice', 'hi-IN-MadhurNeural')
    filename = f"temp_{uuid.uuid4().hex}.mp3"
    
    asyncio.run(generate_audio(text, voice, filename))
    
    with open(filename, 'rb') as f:
        audio_data = f.read()
        
    os.remove(filename)
    return Response(audio_data, mimetype="audio/mpeg")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
