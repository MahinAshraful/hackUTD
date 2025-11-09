# Parkinson's Disease Voice Detector - Frontend

A modern React + Vite frontend for analyzing voice recordings to detect early signs of Parkinson's disease.

## Features

- 🎙️ **Live Recording**: Record 5-second voice samples directly in the browser
- 📁 **File Upload**: Upload existing audio files (WAV, MP3, OGG, WEBM)
- 🔬 **Real-time Analysis**: Get instant predictions from the ML backend
- 📊 **Visual Results**: Color-coded risk levels and feature importance charts
- 📱 **Responsive Design**: Works on desktop and mobile devices

## Tech Stack

- **React 18.2** - UI library
- **Vite 5** - Fast build tool and dev server
- **Web Audio API** - Browser-based audio recording
- **Fetch API** - Backend communication

## Getting Started

### Prerequisites

- Node.js 16+ and npm

### Installation

```bash
# Install dependencies
npm install
```

### Running the App

```bash
# Start development server on http://localhost:3000
npm run dev
```

The app will automatically proxy API requests to the Flask backend at `http://localhost:5001`.

### Building for Production

```bash
# Create optimized production build
npm run build

# Preview production build
npm run preview
```

## How to Use

1. **Start the Backend**: Make sure the Flask API is running on port 5001
   ```bash
   cd ../backend
   python3 app.py
   ```

2. **Start the Frontend**:
   ```bash
   npm run dev
   ```

3. **Access the App**: Open http://localhost:3000 in your browser

4. **Provide Audio**:
   - Upload an existing audio file, OR
   - Click "Record Voice" and say "Ahhhhh" for 5 seconds

5. **Analyze**: Click "Analyze Audio" to get results

## API Integration

The frontend communicates with the Flask backend via:

- **Endpoint**: `POST /api/predict`
- **Input**: Audio file (multipart/form-data)
- **Output**: JSON with prediction results

Example response:
```json
{
  "success": true,
  "prediction": 1,
  "pd_probability": 0.87,
  "healthy_probability": 0.13,
  "risk_level": "HIGH",
  "recommendation": "Significant indicators detected. Consult a neurologist.",
  "feature_importance": {
    "MFCC2": 0.243,
    "RPDE": 0.235,
    "PPE": 0.214
  }
}
```

## Project Structure

```
frontend/
├── src/
│   ├── App.jsx          # Main React component
│   ├── App.css          # Component styles
│   ├── main.jsx         # React entry point
│   └── index.css        # Global styles
├── index.html           # HTML entry point
├── vite.config.js       # Vite configuration
└── package.json         # Dependencies and scripts
```

## Development

The Vite dev server provides:
- ⚡️ Lightning-fast HMR (Hot Module Replacement)
- 🔄 Automatic page reload on file changes
- 🔌 API proxy to backend (no CORS issues)
- 📦 Optimized bundle size

## Browser Support

- Chrome/Edge 90+
- Firefox 88+
- Safari 14+

## License

MIT
