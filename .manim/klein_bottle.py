#!/usr/bin/env python3
"""
标准克莱因瓶可视化渲染器
- 白色背景
- 黑色描边
- 高分辨率网格
"""

# manim -pqk --format=png klein_bottle.py KleinBottle

from manim import *
import numpy as np

# 设置随机种子保证复现
np.random.seed(42)

class KleinBottle(ThreeDScene):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.u_resolution = 12
        self.v_resolution = 8
    
    def construct(self):
        self.camera.background_color = WHITE
        
        self.set_camera_orientation(phi=60 * DEGREES, theta=-45 * DEGREES)
                
        klein_bottle = Surface(
            func=self.klein_bottle_func,
            u_range=[0, 2*PI],
            v_range=[0, 2*PI],
            resolution=(self.u_resolution, self.v_resolution), 
            fill_opacity=0.8,  
            stroke_color=BLACK,
            stroke_width=1.2
        )
        
        klein_bottle.fill_color = WHITE
        
        self.add(klein_bottle)
        
        # 添加随机游走
        random_walk = self.create_random_walk_on_klein_bottle()
        self.add(random_walk)
        
        axes = ThreeDAxes()
        labels = axes.get_axis_labels(
            Text("x-axis").scale(0.7), Text("y-axis").scale(0.45), Text("z-axis").scale(0.45)
        )
        axes.set_stroke(color=BLACK, width=1.2)
        labels.set_stroke(color=BLACK, width=1.2)
        axes.set_color(GRAY_C)
        labels.set_color(GRAY_C)
        self.add(axes, labels)
        
        self.wait(0.1)
    
    def klein_bottle_func(self, u, v):
        r = 4 - 2 * np.cos(u)
        if (0 <= u) and (u < PI):
            x = 6 * np.cos(u) * (0.8 + np.sin(u)) + r * np.cos(u) * np.cos(v)
            y = 16 * np.sin(u) + r * np.sin(u) * np.cos(v)
        else:
            x = 6 * np.cos(u) * (0.8 + np.sin(u)) + r * np.cos(v + PI)
            y = 16 * np.sin(u)
        z = r * np.sin(v)
        
        scale = .25
        return np.array([x * scale, y * scale, z * scale])
    
    
    def create_random_walk_on_klein_bottle(self):
        """在克莱因瓶上创建的随机游走"""
        walk_group = VGroup()
        
        # 网格参数与Surface的resolution一致
        num_nodes = 8
        
        # 8个邻居方向 (u, v) 的偏移
        neighbors = [
            (1, 0),   # 右
            (-1, 0),  # 左
            (0, 1),   # 上
            (0, -1),  # 下
            (1, 1),   # 右上
            (-1, 1),  # 左上
            (1, -1),  # 右下
            (-1, -1)  # 左下
        ]
        
        # 生成随机游走路径（避免重复访问）
        walk_path = []
        visited = set()
        
        # 随机选择起始位置
        current_u_idx = 7 #np.random.randint(0, self.u_resolution)
        current_v_idx = 3 #np.random.randint(0, self.v_resolution)
        
        walk_path.append((current_u_idx, current_v_idx))
        visited.add((current_u_idx, current_v_idx))
        
        for i in range(num_nodes - 1):
            # 获取所有未访问的邻居
            available_neighbors = []
            for du, dv in neighbors:
                next_u = (current_u_idx + du) % self.u_resolution
                next_v = (current_v_idx + dv) % self.v_resolution
                if (next_u, next_v) not in visited:
                    available_neighbors.append((du, dv))
            
            # 如果没有可用邻居，随机选择一个（避免死锁）
            if not available_neighbors:
                neighbor_idx = np.random.randint(0, 8)
                du, dv = neighbors[neighbor_idx]
            else:
                # 从可用邻居中随机选择一个
                neighbor_idx = np.random.randint(0, len(available_neighbors))
                du, dv = available_neighbors[neighbor_idx]
            
            # 更新位置（考虑周期性边界条件）
            current_u_idx = (current_u_idx + du) % self.u_resolution
            current_v_idx = (current_v_idx + dv) % self.v_resolution
            
            walk_path.append((current_u_idx, current_v_idx))
            visited.add((current_u_idx, current_v_idx))
        
        # 创建节点
        nodes = []
        for u_idx, v_idx in walk_path:
            # 将网格索引映射到uv范围 [0, 2*PI]
            u = u_idx * 2 * PI / self.u_resolution
            v = v_idx * 2 * PI / self.v_resolution
            point = self.klein_bottle_func(u, v)
            
            node = Dot3D(point, radius=0.08, color=RED)
            node.set_z_index(10)  # Set high z-index to appear in front
            node.set_shade_in_3d(False)  # Disable 3D shading
            nodes.append(node)
            walk_group.add(node)
        
        # 创建边
        for i in range(len(walk_path) - 1):
            u1_idx, v1_idx = walk_path[i]
            u2_idx, v2_idx = walk_path[i + 1]
            
            # 将网格索引映射到uv范围
            u1 = u1_idx * 2 * PI / self.u_resolution
            v1 = v1_idx * 2 * PI / self.v_resolution
            u2 = u2_idx * 2 * PI / self.u_resolution
            v2 = v2_idx * 2 * PI / self.v_resolution
            point1 = self.klein_bottle_func(u1, v1)
            point2 = self.klein_bottle_func(u2, v2)
            
            edge = Line3D(point1, point2, color=RED, stroke_width=3)
            edge.set_z_index(20)  # Set high z-index to appear in front
            edge.set_shade_in_3d(False)  # Disable 3D shading
            walk_group.add(edge)
        
        # Ensure the entire walk group appears in front
        walk_group.set_z_index(10)
        return walk_group