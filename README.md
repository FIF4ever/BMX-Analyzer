# BMX Analyzer

A Python tool for analyzing BMX trick videos using computer vision and machine learning techniques.

## Features

- Video processing and frame extraction
- Rider pose detection and tracking
- Trick pattern recognition
- Performance metrics calculation
- Visualization of trick analysis

## Requirements

- Python 3.8+
- OpenCV
- NumPy
- MediaPipe
- Matplotlib
- SciPy
- scikit-learn

## Installation

1. Clone this repository:
```
git clone https://github.com/YOUR_USERNAME/bmx-analyzer.git
cd bmx-analyzer
```

2. Install required packages:
```
pip install opencv-python numpy mediapipe matplotlib scipy scikit-learn
```

## Usage

### Basic Usage

```python
from bmx_analyzer import BMXAnalyzer

# Initialize the analyzer
analyzer = BMXAnalyzer()

# Analyze a video file
results = analyzer.analyze_video("path/to/your/bmx_video.mp4")

# Generate and save visualization
analyzer.visualize_results(results, "output_analysis.mp4")

# Print detected tricks and metrics
for trick in results['detected_tricks']:
    print(f"Detected trick: {trick['name']} at time {trick['timestamp']}")
    print(f"Confidence score: {trick['confidence']}")
    print(f"Technical difficulty: {trick['difficulty']}")
```

### Advanced Configuration

You can customize the analyzer settings:

```python
analyzer = BMXAnalyzer(
    confidence_threshold=0.75,  # Minimum confidence for trick detection
    track_smoothing=True,       # Apply trajectory smoothing
    detect_rotations=True,      # Detect rotational tricks
    detect_flips=True,          # Detect flip tricks
    output_mode="detailed"      # "basic" or "detailed" analysis
)
```

## Example Analysis Output

The analyzer returns a dictionary with the following information:

```
{
    'video_info': {
        'duration': 45.2,
        'frame_count': 1356,
        'fps': 30
    },
    'detected_tricks': [
        {
            'name': 'Barspin',
            'timestamp': 12.4,
            'confidence': 0.92,
            'difficulty': 3.5
        },
        {
            'name': 'Tailwhip',
            'timestamp': 28.7,
            'confidence': 0.87,
            'difficulty': 4.2
        }
    ],
    'performance_metrics': {
        'trick_count': 2,
        'average_difficulty': 3.85,
        'trick_variety': 2.0
    }
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
