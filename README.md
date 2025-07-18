# Indian Sign Language Recognition

**Realtime ISL gesture detection & translation**

Detects hand signs (A–Z, 1–9) via MediaPipe & a TensorFlow model, then corrects and translates the sequence using Google Gemini.

## Repository
```
app.py          # Main script
model.h5        # Trained Keras model
gemini_api.py   # Gemini client setup
``` 

## Setup
1. **Clone**
   ```bash
   git clone https://github.com/PadmeshxK/Indian-Sign-Language-Recognition.git
   cd Indian-Sign-Language-Recognition
   ```

   
3. **Install**
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure**
   - In `gemini_api.py`, set `api_key=<gemini-api-key>`
   - Place `model.h5` in root

## Run
```bash
python app.py
```
- **S**: translate buffer
- **ESC**: quit


