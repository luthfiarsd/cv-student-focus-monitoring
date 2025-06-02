import streamlit as st
import cv2
import time
from datetime import datetime
import plotly.express as px
import pandas as pd
import os

import sys
if 'torch' in sys.modules:
    import torch
    torch.multiprocessing.set_start_method('spawn', force=True)

try:
    from ultralytics import YOLO
except ImportError as e:
    st.error(f"Error importing YOLO: {e}")
    st.stop()

st.set_page_config(
    page_title="Distraction Detection Monitor",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        backdrop-filter: blur(4px);
        border: 1px solid rgba(255, 255, 255, 0.18);
    }
    
    .alert-box {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ff6b6b;
        margin: 1rem 0;
    }
    
    .success-box {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #4ecdc4;
        margin: 1rem 0;
    }
    
    .stats-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class DistractionDetector:
    def __init__(self, model_path="best.pt"):
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            self.model = YOLO(model_path)
            self.names = self.model.model.names
            self.total_distracted_count = 0
            self.session_start_time = time.time()
            self.detection_history = []
            self.current_detections = 0
            self.is_running = False
            
        except Exception as e:
            st.error(f"Error initializing detector: {str(e)}")
            raise e
        
    def count_distracted(self, results):
        count = 0
        current_detections = []
        seen_faces = set()
        now = datetime.now()

        last_detection_time = None
        if self.detection_history:
            last_detection_time = self.detection_history[-1]['timestamp']

        for r in results:
            if hasattr(r, 'boxes') and r.boxes is not None:
                for xyxy, conf, cls_id in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
                    coords = tuple(map(lambda x: round(float(x), 1), xyxy))
                    label = self.names[int(cls_id)]
                    confidence = float(conf)

                    if coords not in seen_faces:
                        if label.lower() == "distracted":
                            if (
                                last_detection_time is None or
                                (now - last_detection_time).total_seconds() > 2
                            ):
                                count += 1
                                current_detections.append({
                                    'coordinates': coords,
                                    'confidence': confidence,
                                    'timestamp': now
                                })
                                last_detection_time = now 
                        seen_faces.add(coords)

        if count > 0:
            self.total_distracted_count += count
            self.detection_history.extend(current_detections)
            
    def detect_from_camera(self):
        """Run detection on camera feed"""
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Cannot open camera")
                return
            
            self.is_running = True
            
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    break
                
                results = self.model(frame, conf=0.5, verbose=False)

                current_count = self.count_distracted(results)
                self.current_detections = current_count

                annotated_frame = self.draw_detections(frame, results)

                frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

                yield frame_rgb
                
            cap.release()
            
        except Exception as e:
            st.error(f"Camera error: {str(e)}")
            self.is_running = False
    
    def draw_detections(self, frame, results):
        """Draw bounding boxes and labels on frame"""
        for r in results:
            if hasattr(r, 'boxes') and r.boxes is not None:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    label = self.names[cls_id]

                    if label.lower() == "distracted":
                        color = (0, 0, 255)  # Red
                        thickness = 3
                    else:
                        color = (0, 255, 0)  # Green
                        thickness = 2
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                    
                    # Draw label
                    label_text = f"{label} ({conf:.2f})"
                    cv2.putText(frame, label_text, (x1, y1 - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw counter on frame
        counter_text = f"Total Distracted: {self.total_distracted_count}"
        cv2.putText(frame, counter_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, counter_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
        
        return frame
    
    def stop_detection(self):
        """Stop the detection process"""
        self.is_running = False
        session_duration = time.time() - self.session_start_time
        avg_detections_per_minute = (self.total_distracted_count / session_duration) * 60 if session_duration > 0 else 0
        
    def get_statistics(self):
        session_duration = time.time() - self.session_start_time
        avg_detections_per_minute = (self.total_distracted_count / session_duration) * 60 if session_duration > 0 else 0
        
        return {
            'total_count': self.total_distracted_count,
            'current_detections': self.current_detections,
            'session_duration': session_duration,
            'avg_per_minute': avg_detections_per_minute,
            'last_detection': self.detection_history[-1]['timestamp'] if self.detection_history else None
        }

def run_camera_detection():
    """Function to run camera detection in Streamlit"""
    if 'detector' not in st.session_state or st.session_state.detector is None:
        st.error("Please initialize the detector first")
        return
    
    detector = st.session_state.detector
    
    # Camera display
    frame_placeholder = st.empty()
    
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Cannot open camera. Please check your camera connection.")
            return
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        while st.session_state.camera_active:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to read from camera")
                break
            
            # Run YOLO detection
            try:
                results = detector.model(frame, conf=0.5, verbose=False)
                
                # Count distractions
                current_count = detector.count_distracted(results)
                
                # Draw detections on frame
                annotated_frame = detector.draw_detections(frame, results)
                
                # Convert BGR to RGB for Streamlit
                frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                
                # Display frame
                frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                
                # Small delay to prevent overwhelming the system
                time.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                st.error(f"Detection error: {str(e)}")
                break
        
        cap.release()
        
    except Exception as e:
        st.error(f"Camera initialization error: {str(e)}")

def draw_detection_overlay(frame, results, names):
    """Draw bounding boxes and labels on the frame"""
    for r in results:
        if hasattr(r, 'boxes') and r.boxes is not None:
            for xyxy, conf, cls_id in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
                x1, y1, x2, y2 = map(int, xyxy)
                label = names[int(cls_id)]
                confidence = float(conf)
                
                # Choose color based on label
                if label.lower() == "distracted":
                    color = (0, 0, 255)  # Red for distracted
                    thickness = 3
                else:
                    color = (0, 255, 0)  # Green for normal
                    thickness = 2
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                
                # Draw label with confidence
                label_text = f"{label} ({confidence:.2f})"
                (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                
                # Background rectangle for text
                cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
                cv2.putText(frame, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return frame

def main():
    # Header
    st.markdown('<h1 class="main-header">Distraction Detection Monitor</h1>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'detector' not in st.session_state:
        st.session_state.detector = None
    if 'camera_active' not in st.session_state:
        st.session_state.camera_active = False
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("## 🎛️ Configuration")
        
        model_path = st.text_input("Model Path", value="best.pt", help="Path to your YOLO model file")
        confidence_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.1)
        
        st.markdown("---")
        st.markdown("## 📊 Session Controls")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🎬 Start Detection", type="primary"):
                try:
                    if not os.path.exists(model_path):
                        st.error(f"Model file not found: {model_path}")
                    else:
                        st.session_state.detector = DistractionDetector(model_path)
                        st.session_state.camera_active = True
                        st.success("Detection started!")
                        st.rerun()
                except Exception as e:
                    st.error(f"Error loading model: {str(e)}")
        
        with col2:
            if st.button("⏹️ Stop Detection"):
                st.session_state.camera_active = False
                if st.session_state.detector:
                    st.session_state.detector.stop_detection()
                st.info("Detection stopped!")
                st.rerun()
        
        if st.button("🔄 Reset Counter", type="secondary"):
            if st.session_state.detector:
                st.session_state.detector.total_distracted_count = 0
                st.session_state.detector.detection_history = []
                st.session_state.detector.session_start_time = time.time()
                st.success("Counter reset!")
                st.rerun()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## 📹 Live Camera Feed")
        
        if st.session_state.camera_active and st.session_state.detector:
            # Run camera detection
            run_camera_detection()
        else:
            st.info("👆 Click 'Start Detection' to begin monitoring")
            st.image("https://via.placeholder.com/640x480/1a1a1a/ffffff?text=Camera+Feed+Placeholder", 
                    use_container_width=True)
    
    with col2:
        
        if st.session_state.detector:
            stats = st.session_state.detector.get_statistics()
            
            # Total distracted count
            st.markdown(f"""
            <div class="metric-container">
                <h2 style="margin: 0; font-size: 2.5rem;">{stats['total_count']}</h2>
                <p style="margin: 0; font-size: 1.2rem;">Total Distracted Count</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Session duration
            duration_minutes = int(stats['session_duration'] // 60)
            duration_seconds = int(stats['session_duration'] % 60)
            
            st.markdown(f"""
            <div class="stats-card">
                <h4>⏱️ Session Duration</h4>
                <p>{duration_minutes}m {duration_seconds}s</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Average detections per minute
            st.markdown(f"""
            <div class="stats-card">
                <h4>📊 Avg Detections/Min</h4>
                <p>{stats['avg_per_minute']:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Last detection time
            if stats['last_detection']:
                time_diff = (datetime.now() - stats['last_detection']).total_seconds()
                st.markdown(f"""
                <div class="stats-card">
                    <h4>🕐 Last Detection</h4>
                    <p>{int(time_diff)}s ago</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Alert system
            if stats['total_count'] > 0:
                if stats['avg_per_minute'] > 5: 
                    st.markdown(f"""
                    <div class="alert-box">
                        <h4>⚠️ High Distraction Alert!</h4>
                        <p>Frequent distraction detected. Consider taking a break.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="success-box">
                        <h4>✅ Monitoring Active</h4>
                        <p>System is tracking attention levels.</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Detection history chart
    if st.session_state.detector and st.session_state.detector.detection_history:
        st.markdown("## 📊 Detection Timeline")
        
        # Create timeline data
        timeline_data = []
        for i, detection in enumerate(st.session_state.detector.detection_history):
            timeline_data.append({
                'Detection': i + 1,
                'Time': detection['timestamp'],
                'Confidence': detection['confidence']
            })
        
        df = pd.DataFrame(timeline_data)
        
        # Plot timeline
        fig = px.scatter(df, x='Time', y='Confidence', 
                        title='Distraction Detection Timeline',
                        color='Confidence',
                        color_continuous_scale='Reds')
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()