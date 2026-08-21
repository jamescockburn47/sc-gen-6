"""Futuristic generation animation with pulsing rings and particle stream."""

import math
import random
from dataclasses import dataclass
from typing import List

from PySide6.QtCore import Qt, QTimer, QPointF, Property, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QPainter, QColor, QPen, QRadialGradient, QLinearGradient, QPainterPath
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout


@dataclass
class Particle:
    """A single flowing particle."""
    x: float
    y: float
    speed: float
    size: float
    alpha: float
    trail_length: float


class GenerationAnimation(QWidget):
    """Futuristic pulsing rings with flowing particle stream animation."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(120)
        self.setMaximumHeight(160)
        
        # Animation state
        self._active = False
        self._opacity = 0.0
        self._phase = 0.0
        self._ring_phases = [0.0, 0.33, 0.66]  # 3 rings with offset phases
        self._particles: List[Particle] = []
        
        # Metrics
        self._tokens_per_sec = 0.0
        self._total_tokens = 0
        self._elapsed_time = 0.0
        
        # Colors (purple/violet theme)
        self._accent_color = QColor(139, 124, 246)  # #8b7cf6
        self._accent_light = QColor(167, 139, 250)  # Lighter accent
        self._accent_dark = QColor(99, 102, 241)   # Darker accent
        
        # Animation timer
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_animation)
        
        # Fade animation
        self._fade_anim = None
        
        # Initialize particles
        self._init_particles()
        
        # Setup UI
        self._setup_overlay()
    
    def _setup_overlay(self):
        """Setup metrics overlay."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Metrics row at bottom
        metrics_layout = QHBoxLayout()
        metrics_layout.setContentsMargins(16, 0, 16, 12)
        
        self._status_label = QLabel("Generating...")
        self._status_label.setStyleSheet("""
            color: #a78bfa;
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 1px;
            text-transform: uppercase;
        """)
        metrics_layout.addWidget(self._status_label)
        
        metrics_layout.addStretch()
        
        self._speed_label = QLabel("")
        self._speed_label.setStyleSheet("""
            color: #8b7cf6;
            font-size: 13px;
            font-weight: 700;
            font-family: 'Consolas', 'Monaco', monospace;
        """)
        metrics_layout.addWidget(self._speed_label)
        
        layout.addStretch()
        layout.addLayout(metrics_layout)
    
    def _init_particles(self):
        """Initialize particle pool."""
        self._particles = []
        for _ in range(30):
            self._spawn_particle()
    
    def _spawn_particle(self):
        """Spawn a new particle at the left edge."""
        p = Particle(
            x=-random.uniform(10, 50),
            y=random.uniform(0.3, 0.7),  # Normalized y position
            speed=random.uniform(2.0, 5.0),
            size=random.uniform(2, 5),
            alpha=random.uniform(0.4, 1.0),
            trail_length=random.uniform(15, 40)
        )
        self._particles.append(p)
    
    def _get_opacity(self):
        return self._opacity
    
    def _set_opacity(self, value):
        self._opacity = value
        self.update()
    
    opacity = Property(float, _get_opacity, _set_opacity)
    
    def start(self):
        """Start the animation."""
        if self._active:
            return
            
        self._active = True
        self._phase = 0.0
        self._total_tokens = 0
        self._elapsed_time = 0.0
        self._tokens_per_sec = 0.0
        
        # Fade in
        if self._fade_anim:
            self._fade_anim.stop()
        self._fade_anim = QPropertyAnimation(self, b"opacity")
        self._fade_anim.setDuration(400)
        self._fade_anim.setStartValue(0.0)
        self._fade_anim.setEndValue(1.0)
        self._fade_anim.setEasingCurve(QEasingCurve.OutCubic)
        self._fade_anim.start()
        
        self._timer.start(16)  # ~60 FPS
        self.show()
        self.update()
    
    def stop(self):
        """Stop the animation with fade out."""
        if not self._active:
            return
            
        self._active = False
        
        # Fade out
        if self._fade_anim:
            self._fade_anim.stop()
        self._fade_anim = QPropertyAnimation(self, b"opacity")
        self._fade_anim.setDuration(600)
        self._fade_anim.setStartValue(self._opacity)
        self._fade_anim.setEndValue(0.0)
        self._fade_anim.setEasingCurve(QEasingCurve.InCubic)
        self._fade_anim.finished.connect(self._on_fade_complete)
        self._fade_anim.start()
    
    def _on_fade_complete(self):
        """Handle fade out completion."""
        self._timer.stop()
        self.hide()
    
    def update_metrics(self, tokens_per_sec: float, total_tokens: int, elapsed: float):
        """Update generation metrics."""
        self._tokens_per_sec = tokens_per_sec
        self._total_tokens = total_tokens
        self._elapsed_time = elapsed
        
        if tokens_per_sec > 0:
            self._speed_label.setText(f"{tokens_per_sec:.1f} tok/s  •  {total_tokens} tokens")
        else:
            self._speed_label.setText(f"{total_tokens} tokens")
    
    def _update_animation(self):
        """Update animation state."""
        # Update phase
        self._phase += 0.03
        if self._phase > 2 * math.pi:
            self._phase -= 2 * math.pi
        
        # Update ring phases
        for i in range(len(self._ring_phases)):
            self._ring_phases[i] += 0.02 + i * 0.005
            if self._ring_phases[i] > 1.0:
                self._ring_phases[i] -= 1.0
        
        # Update particles
        width = self.width()
        for p in self._particles:
            p.x += p.speed
            # Reset particle if it goes off screen
            if p.x > width + 50:
                p.x = -random.uniform(10, 50)
                p.y = random.uniform(0.3, 0.7)
                p.speed = random.uniform(2.0, 5.0)
                p.alpha = random.uniform(0.4, 1.0)
        
        self.update()
    
    def paintEvent(self, event):
        """Paint the animation."""
        if self._opacity <= 0:
            return
            
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setOpacity(self._opacity)
        
        width = self.width()
        height = self.height()
        center_x = width / 2
        center_y = height / 2
        
        # Draw background gradient
        bg_grad = QLinearGradient(0, 0, width, 0)
        bg_grad.setColorAt(0, QColor(15, 15, 18, 200))
        bg_grad.setColorAt(0.5, QColor(20, 18, 30, 220))
        bg_grad.setColorAt(1, QColor(15, 15, 18, 200))
        painter.fillRect(self.rect(), bg_grad)
        
        # Draw particle stream (behind rings)
        self._draw_particles(painter, height)
        
        # Draw pulsing rings
        self._draw_rings(painter, center_x, center_y)
        
        # Draw center glow
        self._draw_center_glow(painter, center_x, center_y)
        
        painter.end()
    
    def _draw_rings(self, painter: QPainter, cx: float, cy: float):
        """Draw pulsing concentric rings."""
        max_radius = min(self.width(), self.height()) * 0.45
        
        for i, phase in enumerate(self._ring_phases):
            # Ring expands from center outward
            radius = max_radius * phase
            if radius < 5:
                continue
            
            # Fade out as ring expands
            alpha = int(255 * (1.0 - phase) * 0.6)
            if alpha < 10:
                continue
            
            # Ring color with gradient
            color = QColor(self._accent_color)
            color.setAlpha(alpha)
            
            # Thicker rings when smaller
            thickness = max(1, int(4 * (1.0 - phase)))
            
            pen = QPen(color)
            pen.setWidth(thickness)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(QPointF(cx, cy), radius, radius)
    
    def _draw_particles(self, painter: QPainter, height: float):
        """Draw flowing particle stream."""
        center_y = height / 2
        stream_height = height * 0.25
        
        for p in self._particles:
            # Calculate y position with wave motion
            wave = math.sin(p.x * 0.02 + self._phase) * 8
            y = center_y + (p.y - 0.5) * stream_height * 2 + wave
            
            # Particle color with alpha
            color = QColor(self._accent_light)
            color.setAlpha(int(p.alpha * 180 * self._opacity))
            
            # Draw particle trail
            trail_color = QColor(self._accent_color)
            trail_color.setAlpha(int(p.alpha * 60 * self._opacity))
            
            trail_pen = QPen(trail_color)
            trail_pen.setWidth(int(p.size * 0.6))
            painter.setPen(trail_pen)
            painter.drawLine(
                QPointF(p.x - p.trail_length, y),
                QPointF(p.x, y)
            )
            
            # Draw particle head
            painter.setPen(Qt.NoPen)
            painter.setBrush(color)
            painter.drawEllipse(QPointF(p.x, y), p.size, p.size)
    
    def _draw_center_glow(self, painter: QPainter, cx: float, cy: float):
        """Draw glowing center point."""
        # Pulsing intensity
        pulse = 0.7 + 0.3 * math.sin(self._phase * 2)
        
        # Radial gradient glow
        glow_radius = 25 * pulse
        grad = QRadialGradient(cx, cy, glow_radius)
        
        center_color = QColor(self._accent_light)
        center_color.setAlpha(int(200 * pulse))
        grad.setColorAt(0, center_color)
        
        mid_color = QColor(self._accent_color)
        mid_color.setAlpha(int(100 * pulse))
        grad.setColorAt(0.5, mid_color)
        
        grad.setColorAt(1, QColor(0, 0, 0, 0))
        
        painter.setPen(Qt.NoPen)
        painter.setBrush(grad)
        painter.drawEllipse(QPointF(cx, cy), glow_radius, glow_radius)
        
        # Bright center dot
        core_color = QColor(255, 255, 255, int(220 * pulse))
        painter.setBrush(core_color)
        painter.drawEllipse(QPointF(cx, cy), 3, 3)
