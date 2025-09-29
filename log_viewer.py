"""
VideoLagFix Log Viewer (PyQt6)

Requirements:
  pip install av PyQt6 numpy

Controls:
 - Space toggles play/pause.
 - ',' steps one frame backward.
 - '.' steps one frame forward.
 - L to toggle between local and global view while paused.
 - Click the seek bar to seek to a frame.

Notes: Almost completely vibe coded (couldn't be bothered)
"""

import sys
import math
import re
import argparse

import numpy as np

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor, QFont
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,
    QWidget,
    QPushButton,
    QMainWindow,
)

import av

# Hardcoded video path. Replace with a real file on your system.
VIDEO_PATH = r"trim.mkv"

# Example color ranges: list of tuples (start_frame, end_frame, (r,g,b))
# These are precise frame indices inclusive start, exclusive end.
color_ranges = [
    (0, 150, (200, 50, 50)),
    (150, 300, (50, 200, 50)),
    (300, 600, (50, 50, 200)),
    (600, 1200, (200, 200, 50)),
]

DUPLICATE_COLOR = (200, 50, 50)
REPLACED_COLOR = (50, 255, 50)
SKIP_COLOR = (210, 200, 30)
ADDITIONAL_COLOR = (50, 150, 150)
FAILED_COMP_COLOR = (20, 150, 20)

_PRIORITY = {
    REPLACED_COLOR: 5,
    ADDITIONAL_COLOR: 4,
    FAILED_COMP_COLOR: 3,
    SKIP_COLOR: 2,
    DUPLICATE_COLOR: 1,
}


# Vertical margin in pixels to subtract from video height to account for title/task bars
VERTICAL_MARGIN = 68

LOCAL_MODE_WINDOW_SIZE = 6

class SeekBar(QWidget):
    seekRequested = pyqtSignal(int)

    def __init__(self, total_frames, color_ranges, parent=None):
        super().__init__(parent)
        self.total_frames = total_frames
        self.color_ranges = color_ranges
        self.current_frame = 0
        self.fps = 60.0
        self.force_local = False
        self.playing = False
        self.setMinimumHeight(18)
        self.setMaximumHeight(18)

    def setPlaying(self, playing: bool, fps: float):
        self.playing = playing
        self.fps = fps
        self.update()

    def setForceLocal(self, enable: bool):
        self.force_local = enable
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        w = self.width()
        h = self.height()
        painter.fillRect(0, 0, w, h, QColor(0, 0, 0, 0))

        if self.total_frames <= 0:
            return

        use_local = self.force_local or self.playing
        if use_local:
            # Show 20s window centered around current frame
            half_window = int(self.fps * LOCAL_MODE_WINDOW_SIZE)
            start_frame = max(0, self.current_frame - half_window)
            end_frame = min(self.total_frames, self.current_frame + half_window)
            window_frames = max(1, end_frame - start_frame)
        else:
            # Show entire video
            start_frame = 0
            end_frame = self.total_frames
            window_frames = self.total_frames

        # Draw color ranges relative to current window
        for start, end, color in self.color_ranges:
            s = max(start, start_frame)
            e = min(end, end_frame)
            if e <= s:
                continue
            x1 = int(((s - start_frame) / window_frames) * w)
            x2 = int(((e - start_frame) / window_frames) * w)
            painter.fillRect(x1, 0, max(1, x2 - x1), h, QColor(*color))

        # Draw progress overlay
        progress_x = int(((self.current_frame - start_frame) / window_frames) * w)
        progress_x = max(0, min(progress_x, w))
        painter.fillRect(0, 0, progress_x, h, QColor(0, 0, 0, 140))

        # Draw current position marker
        marker_x = progress_x
        painter.setPen(QColor(255, 20, 255))
        painter.drawLine(marker_x, 0, marker_x, h)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.total_frames > 0:
            x = event.position().x() if hasattr(event, "position") else event.x()
            use_local = self.force_local or self.playing
            if use_local:
                half_window = int(self.fps * LOCAL_MODE_WINDOW_SIZE)
                start_frame = max(0, self.current_frame - half_window)
                end_frame = min(self.total_frames, self.current_frame + half_window)
                window_frames = max(1, end_frame - start_frame)
                frac = x / max(1, self.width())
                frame = int(start_frame + frac * window_frames)
            else:
                frac = x / max(1, self.width())
                frame = int(frac * self.total_frames)
            frame = max(0, min(frame, self.total_frames - 1))
            self.seekRequested.emit(frame)

    def setCurrentFrame(self, frame):
        self.current_frame = frame
        self.update()


class VideoPlayer(QMainWindow):
    def __init__(self, args):
        super().__init__()

        self.video_path = None
        self.color_ranges = []
        self.args = args

        self.container = None
        self.stream = None
        self.fps = None
        self.time_base = None
        self.total_frames = 0
        self.duration = None

        self.frame_iterator = None
        self.current_frame_number = 0

        self.playing = False
        self.force_local = False
        self.timer = QTimer()
        self.timer.timeout.connect(self.play_step)

        self.video_label = None
        self.frame_label = None
        self.bottom_overlay = None
        self.play_button = None
        self.seekbar = None

        if self.args.log == None:
            self.pick_log()
        else:
            self.load_from_logs(self.args.log)

        self.init_ui()
        self.seek_to_frame(0)


    def pick_log(self):
        dialog = QFileDialog(self)
        dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        dialog.setNameFilter("Log (*.log)")
        if dialog.exec() == QFileDialog.DialogCode.Accepted:
            log_file_path = dialog.selectedFiles()[0]
            self.load_from_logs(log_file_path)
        else:
            sys.exit(-1)

    def load_from_logs(self, log_file_path: str):
        with open(log_file_path, mode="r", encoding="utf-8") as f:
            lines = f.readlines()

        # Find video path for open_video to work
        process_regex = re.compile(r"Processing: \"(?P<in>[^\"]+)\" to \"(?P<out>[^\"]+)\"")
        for line in lines:
            if m := process_regex.search(line):
                self.video_path = m.group("out") if self.args.show_output else m.group("in")
                break

        # Open video first to get width/height/total_frames
        self.open_video()
        color_array = [None] * self.total_frames

        def add_range(start: int, end: int, color):
            for i in range(start, end):
                if i >= len(color_array):
                    break
                current = color_array[i]
                if current is None or _PRIORITY[color] > _PRIORITY[current]:
                    color_array[i] = color

        duplicate_regex = re.compile(r"Found duplicate at #([0-9]+)")
        chain_regex = re.compile(r"Duplicate chain (?P<chain_index>[0-9]+) found #(?P<start>[0-9]+)-#(?P<end>[0-9]+), length: (?P<length>[0-9]+), (?P<extra>.+), chain_motion: (?P<chain_motion>[0-9]+\.[0-9]+), state: (?P<state>(CompensateMotion { n_additional: (?P<n_additional>[0-9]+) }|[a-zA-Z]+|))")
        skip_regex = re.compile(r"Skipping duplicate chain #(?P<start>[0-9]+)-#(?P<end>[0-9]+): (?P<result>[a-zA-Z#]+), state: (?P<state>[a-zA-Z#]+)")
        for line in lines:
            if m := duplicate_regex.search(line):
                frame_idx = int(m.group(1))
                add_range(frame_idx, frame_idx+1, DUPLICATE_COLOR)
            if m := chain_regex.search(line):
                start = int(m.group("start"))
                end = int(m.group("end"))
                length = int(m.group("length"))
                state = m.group("state")
                n_additional = m.group("n_additional")
                n_additional = int(n_additional) if n_additional else 0
                replaced_len = length - n_additional

                # Choose base color depending on state
                if state.startswith("FailedCompensate"):
                    base_color = FAILED_COMP_COLOR
                else:
                    base_color = REPLACED_COLOR

                if replaced_len > 0:
                    add_range(start + 1, start + replaced_len, base_color)
                if n_additional > 0:
                    add_range(start + replaced_len, end, ADDITIONAL_COLOR)
            if m := skip_regex.search(line):
                start = int(m.group("start"))
                end = int(m.group("end"))
                add_range(start + 1, end, SKIP_COLOR)

        # Convert to real ranges
        self.color_ranges = []
        i = 0
        while i < len(color_array):
            c = color_array[i]
            if c is not None:
                start = i
                while i < len(color_array) and color_array[i] == c:
                    i += 1
                self.color_ranges.append((start, i, c))
            else:
                i += 1

    def init_ui(self):
        # Native video resolution
        orig_w = getattr(self.stream, 'width', 640)
        orig_h = getattr(self.stream, 'height', 360)

        # Subtract vertical margin to account for title bar and taskbar
        display_h = max(120, orig_h - VERTICAL_MARGIN)
        # Adjust width to preserve aspect ratio
        display_w = int(round((orig_w * display_h) / float(orig_h)))

        self.setWindowTitle("Video Visualization Prototype")
        # Fix window size so aspect ratio remains correct and user cannot resize
        self.setFixedSize(display_w, display_h)

        central = QWidget(self)
        central.setGeometry(0, 0, display_w, display_h)
        self.setCentralWidget(central)

        # Video label fills entire window
        self.video_label = QLabel(central)
        self.video_label.setGeometry(0, 0, display_w, display_h)
        self.video_label.setStyleSheet("background-color: black;")

        # Top-left frame index overlay: give it enough width so numbers do not clip
        self.frame_label = QLabel(central)
        self.frame_label.setStyleSheet(
            "color: white; font-weight: bold; background-color: rgba(0,0,0,0); padding-left:6px; padding-right:6px;"
        )
        self.frame_label.setFont(QFont('SansSerif', 14))
        self.frame_label.move(8, 8)
        # Provide ample width so large frame indices do not clip
        self.frame_label.setFixedWidth(220)
        self.frame_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        # Bottom overlay: semi-transparent rectangle occluding bottom of video
        overlay_h = 72
        self.bottom_overlay = QWidget(central)
        overlay_y = display_h - overlay_h
        self.bottom_overlay.setGeometry(0, overlay_y, display_w, overlay_h)
        self.bottom_overlay.setStyleSheet('background-color: rgba(0, 0, 0, 160);')

        # Play button
        self.play_button = QPushButton('Play', self.bottom_overlay)
        self.play_button.setFixedSize(80, 36)
        self.play_button.move(8, (overlay_h - 36) // 2)
        self.play_button.clicked.connect(self.toggle_play)
        self.play_button.setStyleSheet(
            'QPushButton { color: white; background: rgba(255,255,255,10); border: 1px solid rgba(255,255,255,40); border-radius: 4px; }'
        )

        # Seekbar placed to the right of the button, leaving small margins
        seek_x = 8 + 80 + 12
        seek_w = display_w - seek_x - 8
        self.seekbar = SeekBar(max(1, self.total_frames), self.color_ranges, self.bottom_overlay)
        self.seekbar.setGeometry(seek_x, (overlay_h - 18) // 2, seek_w, 18)
        self.seekbar.seekRequested.connect(self.seek_to_frame)

        # Ensure key events are received
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setFocus()

    def open_video(self):
        try:
            # If no path was found in logs, fall back to the hardcoded path
            if not self.video_path:
                self.video_path = VIDEO_PATH
            self.container = av.open(self.video_path, options={'hwaccel': 'cuda', 'hwaccel_device': '0', 'hwaccel_output_format': 'cuda'})
        except Exception as e:
            raise RuntimeError(f"Failed to open video with PyAV: {e}")

        # pick first video stream
        self.stream = next((s for s in self.container.streams if s.type == 'video'), None)
        if self.stream is None:
            raise RuntimeError("No video stream found")

        # FPS estimation
        if self.stream.average_rate:
            self.fps = float(self.stream.average_rate)
        elif getattr(self.stream, "rate", None):
            self.fps = float(self.stream.rate)
        else:
            self.fps = 60.0

        self.time_base = float(self.stream.time_base)

        # duration in seconds
        if getattr(self.stream, "duration", None):
            self.duration = float(self.stream.duration) * self.time_base
        elif getattr(self.container, "duration", None):
            self.duration = float(self.container.duration) / av.time_base
        else:
            self.duration = None

        if self.duration:
            self.total_frames = int(math.floor(self.duration * self.fps))
        elif getattr(self.stream, "frames", None):
            self.total_frames = int(self.stream.frames)
        else:
            self.total_frames = 0

        # Prepare iterator from start
        try:
            self.container.seek(0)
        except Exception:
            pass
        self.frame_iterator = self.container.decode(self.stream)

    def toggle_play(self):
        if self.playing:
            self.pause()
        else:
            self.play()

    def play(self):
        self.playing = True
        self.play_button.setText('Pause')
        interval = int(round(1000.0 / max(1.0, self.fps)))
        self.timer.start(interval)
        self.seekbar.setPlaying(True, self.fps)

    def pause(self):
        self.playing = False
        self.play_button.setText('Play')
        self.timer.stop()
        self.seekbar.setPlaying(False, self.fps)

    def play_step(self):
        frame = self.decode_next_frame()
        if frame is None:
            self.pause()
            return
        self.show_frame(frame)

    def decode_next_frame(self):
        if not self.frame_iterator:
            self.frame_iterator = self.container.decode(self.stream)

        for _ in range(1000000):
            try:
                frame = next(self.frame_iterator)
            except StopIteration:
                return None
            if frame is None or frame.pts is None:
                continue
            frame_time = float(frame.pts) * self.time_base
            est_idx = int(round(frame_time * self.fps))
            self.current_frame_number = max(0, est_idx)
            return frame
        return None

    def show_frame(self, frame):
        arr = frame.to_ndarray(format='rgb24')
        h, w, _ = arr.shape
        # Build QImage and scale to the label size preserving aspect ratio
        # Convert to bytes to ensure buffer lifetime is safe
        img = QImage(arr.tobytes(), w, h, 3 * w, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(img).scaled(
            self.video_label.width(),
            self.video_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        # Center the image by setting aligned pixmap; label is full window size
        self.video_label.setPixmap(pix)
        self.info_update()

    def info_update(self):
        # Update frame index overlay and seekbar
        self.frame_label.setText(str(self.current_frame_number))
        # ensure text is vertically centered
        self.frame_label.repaint()
        self.seekbar.setCurrentFrame(self.current_frame_number)

    def seek_to_frame(self, target_frame):
        target_frame = int(target_frame)
        target_frame = max(0, target_frame)
        if self.total_frames > 0:
            target_frame = min(target_frame, self.total_frames - 1)

        target_seconds = target_frame / float(self.fps)
        try:
            target_pts = int(math.floor(target_seconds / self.time_base))
        except Exception:
            target_pts = None

        try:
            if target_pts is not None:
                self.container.seek(target_pts, any_frame=False, backward=True, stream=self.stream)
            else:
                self.container.seek(int(target_seconds * av.time_base))
        except Exception:
            try:
                if target_pts is not None:
                    self.container.seek(target_pts, self.stream)
                else:
                    self.container.seek(int(target_seconds * av.time_base))
            except Exception as e:
                print('Seek failed:', e)
                return

        # Reset decode iterator
        self.frame_iterator = self.container.decode(self.stream)

        # Decode until exact frame index or closest
        found = False
        max_decode = int(self.fps * 5)
        decoded = 0
        last_frame = None
        while decoded < max_decode:
            try:
                frame = next(self.frame_iterator)
            except StopIteration:
                break
            decoded += 1
            if frame is None or frame.pts is None:
                continue
            frame_time = float(frame.pts) * self.time_base
            frame_idx = int(round(frame_time * self.fps))
            last_frame = frame
            self.current_frame_number = max(0, frame_idx)
            if frame_idx == target_frame:
                self.show_frame(frame)
                found = True
                break
            if frame_time > target_seconds + (0.5 / self.fps):
                self.show_frame(frame)
                found = True
                break

        if not found and last_frame is not None:
            self.current_frame_number = max(0, int(round(float(last_frame.pts) * self.time_base * self.fps)))
            self.show_frame(last_frame)

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key.Key_Comma:
            self.step_frame(-1)
        elif key == Qt.Key.Key_Period:
            self.step_frame(1)
        elif key == Qt.Key.Key_Space:
            self.toggle_play()
        elif key == Qt.Key.Key_L:
            self.seekbar.setForceLocal(not self.seekbar.force_local)
        else:
            super().keyPressEvent(event)

    def step_frame(self, step):
        was_playing = self.playing
        if was_playing:
            self.pause()
        target = self.current_frame_number + step
        if self.total_frames > 0:
            target = max(0, min(target, self.total_frames - 1))
        self.seek_to_frame(target)
        if was_playing:
            self.play()


def main():
    parser = argparse.ArgumentParser(prog="VideoLagFix Log Viewer", description="Visualizes VideoLagFix logs")
    parser.add_argument("-l", "--log", help="The path to the logfile")
    parser.add_argument("-o", "--show_output", action="store_true", help="Show the patched output instead of the laggy input video")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    player = VideoPlayer(args)
    player.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
