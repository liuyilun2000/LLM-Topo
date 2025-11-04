#!/usr/bin/env python3
"""
高清晰度大尺寸环形曲面渲染器
- 白色背景
- 黑色描边
- 大尺寸环形曲面
- 高分辨率渲染
"""

# manim -pqk -t --format=png torus.py Canvas


from manim import *
import numpy as np

class Canvas(ThreeDScene):
    def construct(self):
        self.camera.background_color = WHITE
        
        self.set_camera_orientation(phi=60 * DEGREES, theta=15 * DEGREES)
        
        torus = Torus(
            major_radius=3,
            minor_radius=1.2,
            resolution=(12, 8),
            fill_opacity=0.5,
            stroke_color=GRAY_E,
            stroke_width=1
        )
        
        torus.fill_color = WHITE      
        
        self.add(torus)
        
        self.wait(0.1)    
