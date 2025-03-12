#!/usr/bin/env python3
"""
Example usage of the BMX Analyzer
"""

import argparse
import os
from bmx_analyzer import BMXAnalyzer

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='BMX Video Trick Analysis')
    parser.add_argument('input_video', help='Path to input BMX video file')
    parser.add_argument('--output', '-o', help='Path to output analysis video', default=None)
    parser.add_argument('--confidence', '-c', type=float, default=0.7, 
                        help='Confidence threshold (0.0-1.0)')
    parser.add_argument('--save-frames', action='store_true', 
                        help='Save individual frames with annotations')
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.isfile(args.input_video):
        print(f"Error: Input video file '{args.input_video}' not found")
        return
    
    # Set default output filename if not specified
    if args.output is None:
        base_name = os.path.splitext(os.path.basename(args.input_video))[0]
        args.output = f"{base_name}_analysis.mp4"
    
    print(f"Analyzing BMX video: {args.input_video}")
    print(f"Confidence threshold: {args.confidence}")
    
    # Initialize the analyzer with custom settings
    analyzer = BMXAnalyzer(
        confidence_threshold=args.confidence,
        track_smoothing=True,
        detect_rotations=True,
        detect_flips=True,
        output_mode="detailed"
    )
    
    # Analyze the video
    try:
        results = analyzer.analyze_video(args.input_video)
        
        # Generate visualization
        analyzer.visualize_results(results, args.output)
        
        # Print results summary
        print("\n===== ANALYSIS RESULTS =====")
        print(f"Video duration: {results['video_info']['duration']:.1f} seconds")
        print(f"Detected tricks: {len(results['detected_tricks'])}")
        
        print("\nTrick breakdown:")
        for i, trick in enumerate(results['detected_tricks'], 1):
            print(f"  {i}. {trick['name']} at {trick['timestamp']:.1f}s " +
                  f"(confidence: {trick['confidence']:.2f}, difficulty: {trick['difficulty']:.1f})")
        
        print("\nPerformance metrics:")
        metrics = results['performance_metrics']
        print(f"  - Total tricks: {metrics['trick_count']}")
        print(f"  - Average difficulty: {metrics['average_difficulty']:.2f}")
        print(f"  - Trick variety: {metrics['trick_variety']:.2f}")
        
        print(f"\nVisualization saved to: {args.output}")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")

if __name__ == "__main__":
    main()
