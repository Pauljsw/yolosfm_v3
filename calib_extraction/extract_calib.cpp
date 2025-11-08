#include <libobsensor/ObSensor.hpp>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>

int main() {
    try {
        std::cout << "==========================================================\n";
        std::cout << "Orbbec Femto Bolt Calibration Extractor\n";
        std::cout << "==========================================================\n\n";

        // Create pipeline
        auto pipeline = std::make_shared<ob::Pipeline>();

        // Get device info
        auto device = pipeline->getDevice();
        auto deviceInfo = device->getDeviceInfo();

        std::cout << "✅ Camera found!\n";
        std::cout << "   Name: " << deviceInfo->name() << "\n";
        std::cout << "   Serial: " << deviceInfo->serialNumber() << "\n";
        std::cout << "   Firmware: " << deviceInfo->firmwareVersion() << "\n\n";

        // Configure streams
        auto config = std::make_shared<ob::Config>();

        // Enable color stream (3840x2160)
        auto colorProfiles = pipeline->getStreamProfileList(OB_SENSOR_COLOR);
        if(colorProfiles) {
            std::cout << "Available Color Resolutions:\n";
            for(uint32_t i = 0; i < colorProfiles->count(); i++) {
                auto profile = colorProfiles->getProfile(i)->as<ob::VideoStreamProfile>();
                std::cout << "  - " << profile->width() << "x" << profile->height()
                         << " @ " << profile->fps() << "fps\n";
            }

            std::shared_ptr<ob::StreamProfile> colorProfile = nullptr;
            for(uint32_t i = 0; i < colorProfiles->count(); i++) {
                auto profile = colorProfiles->getProfile(i)->as<ob::VideoStreamProfile>();
                if(profile->width() == 3840 && profile->height() == 2160) {
                    colorProfile = profile;
                    break;
                }
            }
            if(!colorProfile) {
                colorProfile = colorProfiles->getProfile(0);
            }
            config->enableStream(colorProfile);
            auto vp = colorProfile->as<ob::VideoStreamProfile>();
            std::cout << "✅ Selected Color: " << vp->width() << "x" << vp->height() << "\n\n";
        }

        // Enable depth stream (512x512)
        auto depthProfiles = pipeline->getStreamProfileList(OB_SENSOR_DEPTH);
        if(depthProfiles) {
            std::cout << "Available Depth Resolutions:\n";
            for(uint32_t i = 0; i < depthProfiles->count(); i++) {
                auto profile = depthProfiles->getProfile(i)->as<ob::VideoStreamProfile>();
                std::cout << "  - " << profile->width() << "x" << profile->height()
                         << " @ " << profile->fps() << "fps\n";
            }

            std::shared_ptr<ob::StreamProfile> depthProfile = nullptr;
            for(uint32_t i = 0; i < depthProfiles->count(); i++) {
                auto profile = depthProfiles->getProfile(i)->as<ob::VideoStreamProfile>();
                if(profile->width() == 512 && profile->height() == 512) {
                    depthProfile = profile;
                    break;
                }
            }
            if(!depthProfile) {
                depthProfile = depthProfiles->getProfile(0);
            }
            config->enableStream(depthProfile);
            auto vp = depthProfile->as<ob::VideoStreamProfile>();
            std::cout << "✅ Selected Depth: " << vp->width() << "x" << vp->height() << "\n\n";
        }

        // Start pipeline
        std::cout << "==========================================================\n";
        std::cout << "📡 Starting pipeline...\n";
        std::cout << "==========================================================\n";
        pipeline->start(config);
        std::cout << "✅ Pipeline started\n\n";

        // Get calibration parameters
        std::cout << "==========================================================\n";
        std::cout << "📐 Extracting calibration...\n";
        std::cout << "==========================================================\n\n";

        auto calibParam = pipeline->getCalibrationParam(config);

        // ===== RGB Camera Intrinsic =====
        OBCameraIntrinsic colorIntrinsic = calibParam.intrinsics[OB_SENSOR_COLOR];
        OBCameraDistortion colorDistortion = calibParam.distortion[OB_SENSOR_COLOR];

        std::cout << "📷 RGB Camera Intrinsic:\n";
        std::cout << "   Resolution: " << colorIntrinsic.width << "x" << colorIntrinsic.height << "\n";
        std::cout << "   fx: " << colorIntrinsic.fx << "\n";
        std::cout << "   fy: " << colorIntrinsic.fy << "\n";
        std::cout << "   cx: " << colorIntrinsic.cx << "\n";
        std::cout << "   cy: " << colorIntrinsic.cy << "\n";
        std::cout << "   K matrix:\n";
        std::cout << "     [" << std::setw(10) << colorIntrinsic.fx << ", 0, " << std::setw(10) << colorIntrinsic.cx << "]\n";
        std::cout << "     [0, " << std::setw(10) << colorIntrinsic.fy << ", " << std::setw(10) << colorIntrinsic.cy << "]\n";
        std::cout << "     [0, 0, 1]\n";
        std::cout << "   Distortion: [" << colorDistortion.k1 << ", " << colorDistortion.k2 << ", "
                  << colorDistortion.p1 << ", " << colorDistortion.p2 << ", "
                  << colorDistortion.k3 << ", " << colorDistortion.k4 << ", "
                  << colorDistortion.k5 << ", " << colorDistortion.k6 << "]\n\n";

        // ===== Depth Camera Intrinsic =====
        OBCameraIntrinsic depthIntrinsic = calibParam.intrinsics[OB_SENSOR_DEPTH];
        OBCameraDistortion depthDistortion = calibParam.distortion[OB_SENSOR_DEPTH];

        std::cout << "🎯 Depth Camera Intrinsic:\n";
        std::cout << "   Resolution: " << depthIntrinsic.width << "x" << depthIntrinsic.height << "\n";
        std::cout << "   fx: " << depthIntrinsic.fx << "\n";
        std::cout << "   fy: " << depthIntrinsic.fy << "\n";
        std::cout << "   cx: " << depthIntrinsic.cx << "\n";
        std::cout << "   cy: " << depthIntrinsic.cy << "\n";
        std::cout << "   K matrix:\n";
        std::cout << "     [" << std::setw(10) << depthIntrinsic.fx << ", 0, " << std::setw(10) << depthIntrinsic.cx << "]\n";
        std::cout << "     [0, " << std::setw(10) << depthIntrinsic.fy << ", " << std::setw(10) << depthIntrinsic.cy << "]\n";
        std::cout << "     [0, 0, 1]\n";
        std::cout << "   Distortion (k1-k6, p1-p2): [" << depthDistortion.k1 << ", " << depthDistortion.k2 << ", "
                  << depthDistortion.p1 << ", " << depthDistortion.p2 << ", "
                  << depthDistortion.k3 << ", " << depthDistortion.k4 << ", "
                  << depthDistortion.k5 << ", " << depthDistortion.k6 << "]\n\n";

        // ===== Extrinsic: Depth → Color =====
        OBExtrinsic d2c = calibParam.extrinsics[OB_SENSOR_DEPTH][OB_SENSOR_COLOR];

        std::cout << "⭐ Extrinsic Transformation (Depth → Color):\n";
        std::cout << "   Rotation matrix (R):\n";
        for(int i = 0; i < 3; i++) {
            std::cout << "     [";
            for(int j = 0; j < 3; j++) {
                std::cout << std::setw(10) << std::fixed << std::setprecision(6) << d2c.rot[i*3 + j];
                if(j < 2) std::cout << ", ";
            }
            std::cout << "]\n";
        }
        std::cout << "\n   Translation vector (t) [mm]:\n";
        std::cout << "     [" << std::setw(10) << d2c.trans[0] << ", "
                  << std::setw(10) << d2c.trans[1] << ", "
                  << std::setw(10) << d2c.trans[2] << "]\n";
        std::cout << "\n   Translation vector (t) [meters]:\n";
        std::cout << "     [" << std::setw(10) << std::fixed << std::setprecision(6)
                  << d2c.trans[0]/1000.0 << ", "
                  << std::setw(10) << d2c.trans[1]/1000.0 << ", "
                  << std::setw(10) << d2c.trans[2]/1000.0 << "]\n\n";

        // Calculate baseline
        float baseline_mm = std::sqrt(d2c.trans[0]*d2c.trans[0] +
                                      d2c.trans[1]*d2c.trans[1] +
                                      d2c.trans[2]*d2c.trans[2]);
        std::cout << "   Baseline (camera distance): " << baseline_mm << " mm\n\n";

        // ===== Save to file =====
        std::cout << "==========================================================\n";
        std::cout << "💾 Saving calibration...\n";
        std::cout << "==========================================================\n";

        // Create calib directory
        system("mkdir -p ../calib");

        // Save detailed text file
        std::ofstream file("../calib/femto_bolt_calibration.txt");
        file << "# Orbbec Femto Bolt Factory Calibration\n";
        file << "# Device: " << deviceInfo->name() << "\n";
        file << "# Serial: " << deviceInfo->serialNumber() << "\n";
        file << "# Firmware: " << deviceInfo->firmwareVersion() << "\n\n";

        file << "# RGB Camera Intrinsic (" << colorIntrinsic.width << "x" << colorIntrinsic.height << ")\n";
        file << "rgb_K: [" << colorIntrinsic.fx << ", 0, " << colorIntrinsic.cx << "; "
             << "0, " << colorIntrinsic.fy << ", " << colorIntrinsic.cy << "; "
             << "0, 0, 1]\n";
        file << "rgb_D: [" << colorDistortion.k1 << ", " << colorDistortion.k2 << ", "
             << colorDistortion.p1 << ", " << colorDistortion.p2 << ", "
             << colorDistortion.k3 << ", " << colorDistortion.k4 << ", "
             << colorDistortion.k5 << ", " << colorDistortion.k6 << "]\n\n";

        file << "# Depth Camera Intrinsic (" << depthIntrinsic.width << "x" << depthIntrinsic.height << ")\n";
        file << "depth_K: [" << depthIntrinsic.fx << ", 0, " << depthIntrinsic.cx << "; "
             << "0, " << depthIntrinsic.fy << ", " << depthIntrinsic.cy << "; "
             << "0, 0, 1]\n";
        file << "depth_D: [" << depthDistortion.k1 << ", " << depthDistortion.k2 << ", "
             << depthDistortion.p1 << ", " << depthDistortion.p2 << ", "
             << depthDistortion.k3 << ", " << depthDistortion.k4 << ", "
             << depthDistortion.k5 << ", " << depthDistortion.k6 << "]\n\n";

        file << "# Extrinsic: Depth → Color\n";
        file << "R: [";
        for(int i = 0; i < 9; i++) {
            file << d2c.rot[i];
            if(i < 8) file << ", ";
        }
        file << "]\n";
        file << "t_mm: [" << d2c.trans[0] << ", " << d2c.trans[1] << ", " << d2c.trans[2] << "]\n";
        file << "t_m: [" << d2c.trans[0]/1000.0 << ", " << d2c.trans[1]/1000.0 << ", " << d2c.trans[2]/1000.0 << "]\n";
        file.close();
        std::cout << "✅ Saved: ../calib/femto_bolt_calibration.txt\n";

        // Save Python-friendly format
        std::ofstream npyfile("../calib/femto_bolt_calibration_numpy.txt");
        npyfile << "# Python NumPy format for align_depth_to_rgb\n";
        npyfile << "import numpy as np\n\n";

        npyfile << "rgb_K = np.array([\n";
        npyfile << "    [" << colorIntrinsic.fx << ", 0, " << colorIntrinsic.cx << "],\n";
        npyfile << "    [0, " << colorIntrinsic.fy << ", " << colorIntrinsic.cy << "],\n";
        npyfile << "    [0, 0, 1]\n";
        npyfile << "], dtype=np.float64)\n\n";

        npyfile << "rgb_D = np.array(["
                << colorDistortion.k1 << ", " << colorDistortion.k2 << ", "
                << colorDistortion.p1 << ", " << colorDistortion.p2 << ", "
                << colorDistortion.k3 << ", " << colorDistortion.k4 << ", "
                << colorDistortion.k5 << ", " << colorDistortion.k6
                << "], dtype=np.float64)\n\n";

        npyfile << "depth_K = np.array([\n";
        npyfile << "    [" << depthIntrinsic.fx << ", 0, " << depthIntrinsic.cx << "],\n";
        npyfile << "    [0, " << depthIntrinsic.fy << ", " << depthIntrinsic.cy << "],\n";
        npyfile << "    [0, 0, 1]\n";
        npyfile << "], dtype=np.float64)\n\n";

        npyfile << "# FIXED: Include all 8 distortion coefficients (k1-k6, p1-p2)\n";
        npyfile << "depth_D = np.array(["
                << depthDistortion.k1 << ", " << depthDistortion.k2 << ", "
                << depthDistortion.p1 << ", " << depthDistortion.p2 << ", "
                << depthDistortion.k3 << ", " << depthDistortion.k4 << ", "
                << depthDistortion.k5 << ", " << depthDistortion.k6
                << "], dtype=np.float64)\n\n";

        npyfile << "R = np.array([\n";
        for(int i = 0; i < 3; i++) {
            npyfile << "    [" << d2c.rot[i*3] << ", " << d2c.rot[i*3+1] << ", " << d2c.rot[i*3+2] << "]";
            if(i < 2) npyfile << ",";
            npyfile << "\n";
        }
        npyfile << "], dtype=np.float64)\n\n";

        npyfile << "t = np.array([["
                << d2c.trans[0]/1000.0 << "], ["
                << d2c.trans[1]/1000.0 << "], ["
                << d2c.trans[2]/1000.0
                << "]], dtype=np.float64)  # in meters\n\n";

        npyfile << "# Usage:\n";
        npyfile << "# aligned_depth = align_depth_to_rgb(\n";
        npyfile << "#     depth_img,\n";
        npyfile << "#     rgb_K=rgb_K, rgb_D=rgb_D,\n";
        npyfile << "#     depth_K=depth_K, depth_D=depth_D,\n";
        npyfile << "#     rgb_size=(3840, 2160),\n";
        npyfile << "#     T_d2r=(R, t),\n";
        npyfile << "#     depth_unit='m',\n";
        npyfile << "#     use_simple_resize=False\n";
        npyfile << "# )\n";
        npyfile.close();
        std::cout << "✅ Saved: ../calib/femto_bolt_calibration_numpy.txt\n";

        // ===== Save JSON files =====

        // 1. rgb_camera_info.json
        std::ofstream rgb_json("../calib/rgb_camera_info.json");
        rgb_json << "{\n";
        rgb_json << "  \"width\": " << colorIntrinsic.width << ",\n";
        rgb_json << "  \"height\": " << colorIntrinsic.height << ",\n";
        rgb_json << "  \"distortion_model\": \"rational_polynomial\",\n";
        rgb_json << "  \"D\": [\n";
        rgb_json << "    " << colorDistortion.k1 << ",\n";
        rgb_json << "    " << colorDistortion.k2 << ",\n";
        rgb_json << "    " << colorDistortion.p1 << ",\n";
        rgb_json << "    " << colorDistortion.p2 << ",\n";
        rgb_json << "    " << colorDistortion.k3 << ",\n";
        rgb_json << "    " << colorDistortion.k4 << ",\n";
        rgb_json << "    " << colorDistortion.k5 << ",\n";
        rgb_json << "    " << colorDistortion.k6 << "\n";
        rgb_json << "  ],\n";
        rgb_json << "  \"K\": [\n";
        rgb_json << "    [" << colorIntrinsic.fx << ", 0.0, " << colorIntrinsic.cx << "],\n";
        rgb_json << "    [0.0, " << colorIntrinsic.fy << ", " << colorIntrinsic.cy << "],\n";
        rgb_json << "    [0.0, 0.0, 1.0]\n";
        rgb_json << "  ]\n";
        rgb_json << "}\n";
        rgb_json.close();
        std::cout << "✅ Saved: ../calib/rgb_camera_info.json\n";

        // 2. depth_camera_info.json - FIXED: Use all distortion coefficients
        std::ofstream depth_json("../calib/depth_camera_info.json");
        depth_json << "{\n";
        depth_json << "  \"width\": " << depthIntrinsic.width << ",\n";
        depth_json << "  \"height\": " << depthIntrinsic.height << ",\n";
        depth_json << "  \"distortion_model\": \"rational_polynomial\",\n";
        depth_json << "  \"D\": [\n";
        depth_json << "    " << depthDistortion.k1 << ",\n";
        depth_json << "    " << depthDistortion.k2 << ",\n";
        depth_json << "    " << depthDistortion.p1 << ",\n";
        depth_json << "    " << depthDistortion.p2 << ",\n";
        depth_json << "    " << depthDistortion.k3 << ",\n";
        depth_json << "    " << depthDistortion.k4 << ",\n";
        depth_json << "    " << depthDistortion.k5 << ",\n";
        depth_json << "    " << depthDistortion.k6 << "\n";
        depth_json << "  ],\n";
        depth_json << "  \"K\": [\n";
        depth_json << "    [" << depthIntrinsic.fx << ", 0.0, " << depthIntrinsic.cx << "],\n";
        depth_json << "    [0.0, " << depthIntrinsic.fy << ", " << depthIntrinsic.cy << "],\n";
        depth_json << "    [0.0, 0.0, 1.0]\n";
        depth_json << "  ]\n";
        depth_json << "}\n";
        depth_json.close();
        std::cout << "✅ Saved: ../calib/depth_camera_info.json\n";

        // 3. extrinsic_depth_to_color.json
        std::ofstream ext_json("../calib/extrinsic_depth_to_color.json");
        ext_json << "{\n";
        ext_json << "  \"description\": \"Extrinsic transformation from Depth to Color frame (Orbbec Femto Bolt)\",\n";
        ext_json << "  \"camera\": \"" << deviceInfo->name() << "\",\n";
        ext_json << "  \"serial_number\": \"" << deviceInfo->serialNumber() << "\",\n";
        ext_json << "  \"firmware_version\": \"" << deviceInfo->firmwareVersion() << "\",\n";
        ext_json << "  \"R\": [\n";
        for(int i = 0; i < 3; i++) {
            ext_json << "    [" << d2c.rot[i*3] << ", " << d2c.rot[i*3+1] << ", " << d2c.rot[i*3+2] << "]";
            if(i < 2) ext_json << ",";
            ext_json << "\n";
        }
        ext_json << "  ],\n";
        ext_json << "  \"t\": [" << d2c.trans[0]/1000.0 << ", " << d2c.trans[1]/1000.0 << ", " << d2c.trans[2]/1000.0 << "],\n";
        ext_json << "  \"t_mm\": [" << d2c.trans[0] << ", " << d2c.trans[1] << ", " << d2c.trans[2] << "],\n";
        ext_json << "  \"baseline_mm\": " << baseline_mm << ",\n";
        ext_json << "  \"note\": \"This transformation converts points from depth camera coordinates to color camera coordinates. Apply as: P_color = R * P_depth + t\"\n";
        ext_json << "}\n";
        ext_json.close();
        std::cout << "✅ Saved: ../calib/extrinsic_depth_to_color.json\n\n";

        // Stop pipeline
        pipeline->stop();

        std::cout << "==========================================================\n";
        std::cout << "✨ SUCCESS! Calibration extraction complete!\n";
        std::cout << "==========================================================\n";

        return 0;

    } catch(ob::Error &e) {
        std::cerr << "❌ Orbbec Error: " << e.getMessage() << std::endl;
        return 1;
    } catch(std::exception &e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }
}
