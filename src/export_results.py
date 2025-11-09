"""
Export Module
Exports results to various formats: PLY, CSV, GeoJSON, PNG, HTML.
"""
import numpy as np
import json
import csv
from typing import List, Dict, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ResultExporter:
    """Exports fusion and measurement results"""
    
    def __init__(self, output_dir: str, class_names: List[str], colors: Optional[Dict] = None):
        """
        Initialize exporter.
        
        Args:
            output_dir: Output directory path
            class_names: List of class names
            colors: Dictionary mapping class names to RGB colors
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.class_names = class_names
        
        # Default colors
        if colors is None:
            self.colors = {
                'crack': [255, 0, 0],
                'spalling': [0, 255, 0],
                'efflorescence': [0, 0, 255],
                'exposed_rebar': [255, 255, 0],
                'background': [128, 128, 128]
            }
        else:
            self.colors = colors
    
    def export_labeled_cloud_ply(
        self,
        voxel_centers: np.ndarray,
        labels: np.ndarray,
        probabilities: np.ndarray,
        filename: str = 'A_cloud_labeled.ply'
    ):
        """
        Export labeled point cloud to PLY format.
        
        Args:
            voxel_centers: Nx3 coordinates
            labels: N class labels
            probabilities: NxC probability distribution
            filename: Output filename
        """
        output_path = self.output_dir / filename
        
        logger.info(f"Exporting labeled cloud to {output_path}")
        
        # Assign colors based on labels
        colors = np.zeros((len(labels), 3), dtype=np.uint8)
        for i, label in enumerate(labels):
            class_name = self.class_names[label]
            colors[i] = self.colors.get(class_name, [128, 128, 128])
        
        # Write PLY
        with open(output_path, 'w') as f:
            # Header
            f.write('ply\n')
            f.write('format ascii 1.0\n')
            f.write(f'element vertex {len(voxel_centers)}\n')
            f.write('property float x\n')
            f.write('property float y\n')
            f.write('property float z\n')
            f.write('property uchar red\n')
            f.write('property uchar green\n')
            f.write('property uchar blue\n')
            f.write('property int label\n')
            f.write('property float confidence\n')
            f.write('end_header\n')
            
            # Data
            for i in range(len(voxel_centers)):
                x, y, z = voxel_centers[i]
                r, g, b = colors[i]
                label = labels[i]
                conf = probabilities[i, label]
                
                f.write(f'{x:.6f} {y:.6f} {z:.6f} {r} {g} {b} {label} {conf:.6f}\n')
        
        logger.info(f"Exported {len(voxel_centers)} points to PLY")
    
    def export_instances_ply(
        self,
        instances: List['Instance3D'],
        filename: str = 'instances_3d.ply'
    ):
        """
        Export instances to PLY format (colored by instance).
        
        Args:
            instances: List of Instance3D objects
            filename: Output filename
        """
        output_path = self.output_dir / filename
        
        logger.info(f"Exporting instances to {output_path}")
        
        # Collect all points and assign colors
        all_points = []
        all_colors = []
        all_labels = []
        
        # Generate distinct colors for each instance
        np.random.seed(42)
        instance_colors = {}
        
        for inst in instances:
            # Random color for this instance
            inst_color = np.random.randint(50, 256, size=3, dtype=np.uint8)
            instance_colors[inst.instance_id] = inst_color
            
            for point in inst.voxel_centers:
                all_points.append(point)
                all_colors.append(inst_color)
                all_labels.append(inst.class_id)
        
        all_points = np.array(all_points)
        all_colors = np.array(all_colors)
        all_labels = np.array(all_labels)
        
        # Write PLY
        with open(output_path, 'w') as f:
            # Header
            f.write('ply\n')
            f.write('format ascii 1.0\n')
            f.write(f'element vertex {len(all_points)}\n')
            f.write('property float x\n')
            f.write('property float y\n')
            f.write('property float z\n')
            f.write('property uchar red\n')
            f.write('property uchar green\n')
            f.write('property uchar blue\n')
            f.write('property int label\n')
            f.write('end_header\n')
            
            # Data
            for i in range(len(all_points)):
                x, y, z = all_points[i]
                r, g, b = all_colors[i]
                label = all_labels[i]
                
                f.write(f'{x:.6f} {y:.6f} {z:.6f} {r} {g} {b} {label}\n')
        
        logger.info(f"Exported {len(instances)} instances ({len(all_points)} points) to PLY")
    
    def export_measurements_csv(
        self,
        measurements: List[Dict],
        filename: str = 'instances.csv'
    ):
        """
        Export measurements to CSV format.
        
        Args:
            measurements: List of measurement dictionaries
            filename: Output filename
        """
        output_path = self.output_dir / filename
        
        logger.info(f"Exporting measurements to {output_path}")
        
        if not measurements:
            logger.warning("No measurements to export")
            return
        
        # Flatten nested structures for CSV
        flattened = []
        for m in measurements:
            flat = {}
            for key, value in m.items():
                if isinstance(value, (list, np.ndarray)):
                    # Convert arrays to strings
                    if key in ['bbox_min', 'bbox_max', 'bbox_size', 'centroid']:
                        flat[key] = ';'.join([f'{v:.4f}' for v in value])
                    elif key in ['orientation', 'plane_normal']:
                        flat[key] = ';'.join([f'{v:.6f}' for v in value]) if value else ''
                    elif key == 'endpoints':
                        # Format endpoints as "x1,y1,z1;x2,y2,z2"
                        if len(value) > 0:
                            flat[key] = ';'.join([','.join([f'{v:.4f}' for v in pt]) for pt in value])
                        else:
                            flat[key] = ''
                    else:
                        # Generic array handling
                        flat[key] = str(value)
                else:
                    flat[key] = value
            flattened.append(flat)

        # Write CSV
        if flattened:
            # Collect all unique keys from all measurements (handle optional fields)
            all_keys = set()
            for flat in flattened:
                all_keys.update(flat.keys())
            fieldnames = sorted(all_keys)  # Sort for consistent column order

            with open(output_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(flattened)

            logger.info(f"Exported {len(flattened)} measurements to CSV")
    
    def export_instances_geojson(
        self,
        measurements: List[Dict],
        filename: str = 'instances_3d.geojson',
        crs: str = 'EPSG:4326'
    ):
        """
        Export instances to GeoJSON format.
        
        Note: Assumes coordinates are in a projected CRS or need transformation.
        
        Args:
            measurements: List of measurement dictionaries
            filename: Output filename
            crs: Coordinate reference system
        """
        output_path = self.output_dir / filename
        
        logger.info(f"Exporting instances to GeoJSON: {output_path}")
        
        features = []
        
        for m in measurements:
            # Create geometry (Point at centroid)
            centroid = m.get('centroid')
            if centroid is None:
                continue
            
            # Properties (exclude large arrays)
            properties = {}
            for key, value in m.items():
                if key in ['instance_id', 'class_id', 'class_name', 'num_voxels',
                          'volume_m3', 'mean_confidence', 'view_count', 'entropy',
                          'length_m', 'width_m', 'area_m2', 'depth_m',
                          'num_branches', 'num_endpoints', 'skeleton_points']:
                    properties[key] = value
            
            # Create feature
            feature = {
                'type': 'Feature',
                'geometry': {
                    'type': 'Point',
                    'coordinates': centroid  # [x, y, z] or [lon, lat, alt]
                },
                'properties': properties
            }
            
            features.append(feature)
        
        # Create FeatureCollection
        geojson = {
            'type': 'FeatureCollection',
            'crs': {
                'type': 'name',
                'properties': {'name': crs}
            },
            'features': features
        }
        
        # Write GeoJSON
        with open(output_path, 'w') as f:
            json.dump(geojson, f, indent=2)
        
        logger.info(f"Exported {len(features)} features to GeoJSON")
    
    def export_report_markdown(
        self,
        fusion_stats: Dict,
        measurements: List[Dict],
        config: Dict,
        filename: str = 'report.md'
    ):
        """
        Generate summary report in Markdown format.
        
        Args:
            fusion_stats: Statistics from label fusion
            measurements: List of measurement dictionaries
            config: Configuration used
            filename: Output filename
        """
        output_path = self.output_dir / filename
        
        logger.info(f"Generating report: {output_path}")
        
        with open(output_path, 'w') as f:
            f.write('# 3D Defect Fusion Report\n\n')
            
            # Dataset summary
            f.write('## Dataset Summary\n\n')
            f.write(f"- **Number of views**: {fusion_stats.get('num_views', 0)}\n")
            f.write(f"- **Number of masks**: {fusion_stats.get('num_masks', 0)}\n")
            f.write(f"- **Total points projected**: {fusion_stats.get('total_points', 0):,}\n")
            f.write(f"- **Labeled voxels**: {fusion_stats.get('labeled_voxels', 0):,}\n")
            f.write('\n')
            
            # Quality metrics
            f.write('## Quality Metrics\n\n')
            f.write(f"- **Mean entropy**: {fusion_stats.get('mean_entropy', 0):.4f}\n")
            f.write(f"- **Conflict rate**: {fusion_stats.get('conflict_rate', 0):.2%}\n")
            f.write('\n')
            
            # Instance summary
            f.write('## Instance Summary\n\n')
            f.write(f"- **Total instances**: {len(measurements)}\n")
            
            # By class
            class_counts = {}
            for m in measurements:
                class_name = m.get('class_name', 'unknown')
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            f.write('\n### Instances by Class\n\n')
            for class_name, count in sorted(class_counts.items()):
                f.write(f"- **{class_name}**: {count}\n")
            f.write('\n')
            
            # Measurement statistics
            f.write('## Measurement Statistics\n\n')
            
            # Cracks
            crack_measures = [m for m in measurements if 'crack' in m.get('class_name', '').lower()]
            if crack_measures:
                lengths = [m.get('length_m', 0) for m in crack_measures]
                total_length = sum(lengths)
                avg_length = np.mean(lengths)
                max_length = max(lengths)
                
                f.write('### Cracks\n\n')
                f.write(f"- **Total length**: {total_length:.2f} m\n")
                f.write(f"- **Average length**: {avg_length:.2f} m\n")
                f.write(f"- **Maximum length**: {max_length:.2f} m\n")
                f.write('\n')
            
            # Area defects
            area_measures = [m for m in measurements 
                           if any(x in m.get('class_name', '').lower() 
                                 for x in ['spall', 'efflor', 'rebar'])]
            if area_measures:
                areas = [m.get('area_m2', 0) for m in area_measures]
                total_area = sum(areas)
                avg_area = np.mean(areas)
                max_area = max(areas)
                
                f.write('### Area-based Defects\n\n')
                f.write(f"- **Total area**: {total_area:.4f} m²\n")
                f.write(f"- **Average area**: {avg_area:.4f} m²\n")
                f.write(f"- **Maximum area**: {max_area:.4f} m²\n")
                f.write('\n')
            
            # Top instances
            f.write('## Top Instances\n\n')
            
            # Sort by length or area
            sorted_measures = sorted(measurements, 
                                   key=lambda x: x.get('length_m', x.get('area_m2', 0)), 
                                   reverse=True)[:10]
            
            f.write('| Instance ID | Class | Measurement | Confidence | Views |\n')
            f.write('|-------------|-------|-------------|------------|-------|\n')
            
            for m in sorted_measures:
                inst_id = m.get('instance_id', '')
                class_name = m.get('class_name', '')
                
                if 'length_m' in m:
                    measurement = f"{m['length_m']:.2f} m"
                elif 'area_m2' in m:
                    measurement = f"{m['area_m2']:.4f} m²"
                else:
                    measurement = 'N/A'
                
                conf = m.get('mean_confidence', 0)
                views = m.get('view_count', 0)
                
                f.write(f"| {inst_id} | {class_name} | {measurement} | {conf:.2f} | {views} |\n")
            
            f.write('\n')
            
            # Configuration
            f.write('## Configuration\n\n')
            f.write('```yaml\n')
            import yaml
            f.write(yaml.dump(config, default_flow_style=False))
            f.write('```\n')
        
        logger.info("Report generated successfully")

    def export_pointcloud_images(
        self,
        voxel_centers: np.ndarray,
        labels: np.ndarray,
        filename_prefix: str = 'pointcloud',
        angles: List[tuple] = None
    ):
        """
        Render point cloud to PNG images from multiple viewpoints.

        Args:
            voxel_centers: Nx3 coordinates
            labels: N class labels
            filename_prefix: Prefix for output filenames
            angles: List of (azimuth, elevation) tuples in degrees
        """
        try:
            import open3d as o3d
        except ImportError:
            logger.warning("open3d not installed, skipping PNG rendering. Install with: pip install open3d")
            return

        logger.info(f"Rendering point cloud images to {self.output_dir}")

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(voxel_centers)

        # Assign colors based on labels
        colors = np.zeros((len(labels), 3), dtype=np.float64)
        for i, label in enumerate(labels):
            class_name = self.class_names[label]
            rgb = self.colors.get(class_name, [128, 128, 128])
            colors[i] = [rgb[0]/255.0, rgb[1]/255.0, rgb[2]/255.0]

        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Default viewing angles if not specified
        if angles is None:
            angles = [
                (0, 0),      # Front
                (90, 0),     # Right
                (180, 0),    # Back
                (270, 0),    # Left
                (0, 45),     # Top-front
                (0, -45),    # Bottom-front
            ]

        # Render from each angle
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=1920, height=1080)
        vis.add_geometry(pcd)

        render_option = vis.get_render_option()
        render_option.point_size = 3.0
        render_option.background_color = np.array([1, 1, 1])  # White background

        for i, (azimuth, elevation) in enumerate(angles):
            # Set camera view
            ctr = vis.get_view_control()
            ctr.set_front([np.sin(np.radians(azimuth)) * np.cos(np.radians(elevation)),
                          np.sin(np.radians(elevation)),
                          np.cos(np.radians(azimuth)) * np.cos(np.radians(elevation))])
            ctr.set_up([0, 1, 0])

            # Capture image
            output_path = self.output_dir / f"{filename_prefix}_view_{i:02d}_az{azimuth:03d}_el{elevation:+03d}.png"
            vis.capture_screen_image(str(output_path), do_render=True)
            logger.info(f"  Rendered view {i+1}/{len(angles)}: {output_path.name}")

        vis.destroy_window()
        logger.info(f"Rendered {len(angles)} point cloud images")

    def export_interactive_html(
        self,
        voxel_centers: np.ndarray,
        labels: np.ndarray,
        measurements: List[Dict],
        filename: str = 'interactive_3d.html'
    ):
        """
        Create interactive 3D visualization using Plotly.

        Args:
            voxel_centers: Nx3 coordinates
            labels: N class labels
            measurements: List of measurement dictionaries
            filename: Output HTML filename
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            logger.warning("plotly not installed, skipping HTML visualization. Install with: pip install plotly")
            return

        output_path = self.output_dir / filename
        logger.info(f"Creating interactive 3D visualization: {output_path}")

        # Sample points if too many (plotly can be slow with >100k points)
        max_points = 50000
        if len(voxel_centers) > max_points:
            logger.info(f"Sampling {max_points} points from {len(voxel_centers)} for performance")
            indices = np.random.choice(len(voxel_centers), max_points, replace=False)
            voxel_centers = voxel_centers[indices]
            labels = labels[indices]

        # Prepare data by class
        traces = []

        for class_id, class_name in enumerate(self.class_names):
            mask = labels == class_id
            if not np.any(mask):
                continue

            points = voxel_centers[mask]
            rgb = self.colors.get(class_name, [128, 128, 128])
            color_str = f'rgb({rgb[0]}, {rgb[1]}, {rgb[2]})'

            trace = go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode='markers',
                name=f"{class_name} ({np.sum(mask)} points)",
                marker=dict(
                    size=2,
                    color=color_str,
                    opacity=0.8
                ),
                hovertemplate=f"<b>{class_name}</b><br>" +
                             "X: %{x:.3f}<br>" +
                             "Y: %{y:.3f}<br>" +
                             "Z: %{z:.3f}<br>" +
                             "<extra></extra>"
            )
            traces.append(trace)

        # Add instance centroids as larger markers
        if measurements:
            centroids = []
            centroid_labels = []
            centroid_texts = []

            for m in measurements:
                centroid = m.get('centroid')
                if centroid is not None:
                    centroids.append(centroid)
                    class_name = m.get('class_name', 'unknown')
                    centroid_labels.append(class_name)

                    # Create hover text
                    if 'length_m' in m:
                        text = f"{m['instance_id']}<br>Length: {m['length_m']:.2f}m"
                    elif 'area_m2' in m:
                        text = f"{m['instance_id']}<br>Area: {m['area_m2']:.4f}m²"
                    else:
                        text = m['instance_id']
                    centroid_texts.append(text)

            if centroids:
                centroids = np.array(centroids)
                trace_centroids = go.Scatter3d(
                    x=centroids[:, 0],
                    y=centroids[:, 1],
                    z=centroids[:, 2],
                    mode='markers+text',
                    name='Instance Centroids',
                    marker=dict(
                        size=8,
                        color='black',
                        symbol='diamond',
                        line=dict(width=2, color='white')
                    ),
                    text=[str(i+1) for i in range(len(centroids))],
                    textposition='top center',
                    hovertext=centroid_texts,
                    hovertemplate="%{hovertext}<extra></extra>"
                )
                traces.append(trace_centroids)

        # Create figure
        fig = go.Figure(data=traces)

        # Update layout
        fig.update_layout(
            title='3D Defect Visualization (Interactive)',
            scene=dict(
                xaxis_title='X (m)',
                yaxis_title='Y (m)',
                zaxis_title='Z (m)',
                aspectmode='data'
            ),
            width=1400,
            height=900,
            showlegend=True,
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='black',
                borderwidth=1
            )
        )

        # Save HTML
        fig.write_html(str(output_path))
        logger.info(f"Interactive HTML visualization saved")


def export_all_results(
    fusion_result: Dict,
    instances: List['Instance3D'],
    measurements: List[Dict],
    output_dir: str,
    class_names: List[str],
    config: Dict,
    colors: Optional[Dict] = None
):
    """
    Export all results to various formats.
    
    Args:
        fusion_result: Result from LabelFusion
        instances: List of Instance3D objects
        measurements: List of measurement dictionaries
        output_dir: Output directory
        class_names: List of class names
        config: Configuration dictionary
        colors: Color mapping for classes
    """
    exporter = ResultExporter(output_dir, class_names, colors)

    # Export labeled cloud
    exporter.export_labeled_cloud_ply(
        fusion_result['voxel_centers'],
        fusion_result['labels'],
        fusion_result['probabilities']
    )

    # Export instances
    exporter.export_instances_ply(instances)

    # Export measurements
    exporter.export_measurements_csv(measurements)
    exporter.export_instances_geojson(measurements)

    # Export report
    exporter.export_report_markdown(
        fusion_result['stats'],
        measurements,
        config
    )

    # Export visualizations (PNG images from multiple angles)
    logger.info("Generating visualization images...")
    exporter.export_pointcloud_images(
        fusion_result['voxel_centers'],
        fusion_result['labels'],
        filename_prefix='crack_pointcloud'
    )

    # Export interactive HTML
    logger.info("Generating interactive 3D HTML...")
    exporter.export_interactive_html(
        fusion_result['voxel_centers'],
        fusion_result['labels'],
        measurements,
        filename='crack_visualization_interactive.html'
    )

    logger.info(f"All results exported to {output_dir}")


if __name__ == '__main__':
    # Test
    logging.basicConfig(level=logging.INFO)
    
    # Create dummy data
    voxel_centers = np.random.randn(100, 3)
    labels = np.random.randint(0, 5, 100)
    probabilities = np.random.rand(100, 5)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    
    class_names = ['crack', 'spalling', 'efflorescence', 'exposed_rebar', 'background']
    
    exporter = ResultExporter('test_output', class_names)
    
    # Test PLY export
    exporter.export_labeled_cloud_ply(voxel_centers, labels, probabilities, 'test.ply')
    
    print("Export test completed")
