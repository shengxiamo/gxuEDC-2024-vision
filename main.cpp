#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

// 绘制四边形的边框和对角线，并计算对角线交点坐标
cv::Point drawQuadrilateral(cv::Mat& frame, const std::vector<cv::Point>& points) 
{
    cv::polylines(frame, points, true, cv::Scalar(0, 0, 255), 2); // 四边形轮廓使用红色线条

    // 绘制四条边
    for (size_t i = 0; i < points.size(); ++i) 
    {
        cv::line(frame, points[i], points[(i + 1) % points.size()], cv::Scalar(0, 0, 255), 2);
    }

    cv::line(frame, points[0], points[2], cv::Scalar(0, 0, 255), 2);
    cv::line(frame, points[1], points[3], cv::Scalar(0, 0, 255), 2);
    
    // 计算对角线交点坐标（也就是中心点）
    cv::Point intersection = (points[0] + points[2]) / 2;
    cv::circle(frame, intersection, 5, cv::Scalar(255, 0, 0), -1); // 在中心点处画一个蓝色的小圆点
    return intersection;
}

int main() 
{
    // 打开默认摄像头
    cv::VideoCapture cap(0);

    // 检查摄像头是否成功打开
    if (!cap.isOpened()) 
    {
        std::cerr << "Error opening video stream or file" << std::endl;
        return -1;
    }

    // 记录时间以计算帧率
    auto startTime = std::chrono::steady_clock::now();
    int frameCounter = 0;

    while (true) 
    {
        cv::Mat frame;
        // 读取摄像头帧
        cap >> frame;

        // 如果帧为空，立即退出循环
        if (frame.empty())
            break;

        // 颜色通道相减法识别绿色平行四边形
        cv::Mat bgrChannels[3];
        cv::split(frame, bgrChannels); // 拆分成BGR通道
        cv::Mat greenComponent = bgrChannels[1] - bgrChannels[0] - bgrChannels[2]; // 提取绿色通道

        // 对绿色通道应用阈值操作，得到绿色平行四边形的掩码
        cv::Mat mask;
        cv::threshold(greenComponent, mask, 30, 255, cv::THRESH_BINARY);

        // 对掩码应用一次形态学开运算去除噪点
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
        cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);

        // 对掩码应用一次形态学闭运算填补内部空洞
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

        // 显示掩码
        cv::imshow("Mask", mask);

        // 寻找轮廓
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // 找到最大的轮廓
        int maxContourIndex = -1;
        double maxContourLength = 0;
        for (size_t i = 0; i < contours.size(); ++i) 
        {
            double contourLength = cv::arcLength(contours[i], true);
            if (contourLength > maxContourLength) 
            {
                maxContourLength = contourLength;
                maxContourIndex = i;
            }
        }

        // 如果找到了最大的轮廓
        if (maxContourIndex != -1) 
        {
            // 获取最大轮廓的近似多边形
            std::vector<cv::Point> approx;
            cv::approxPolyDP(contours[maxContourIndex], approx, cv::arcLength(contours[maxContourIndex], true) * 0.02, true);

            // 如果近似多边形有4个顶点，则认为是平行四边形
            if (approx.size() == 4) 
            {
                // 绘制平行四边形轮廓和中心点
                drawQuadrilateral(frame, approx);
            }
        }

        // 显示处理后的帧和帧率
        auto currentTime = std::chrono::steady_clock::now();
        double elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - startTime).count() / 1000.0;
        double fps = frameCounter / elapsedTime;
        cv::putText(frame, "FPS: " + std::to_string(fps), cv::Point(frame.cols - 150, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
        cv::imshow("Frame", frame);

        // 更新帧数和时间
        ++frameCounter;
        if (elapsedTime >= 1.0) 
        {
            startTime = currentTime;
            frameCounter = 0;
        }

        // 按下ESC键退出
        char c = (char) cv::waitKey(1);
        if (c == 27)
            break;
    }

    // 释放视频捕获对象
    cap.release();

    // 关闭所有窗口
    cv::destroyAllWindows();

    return 0;
}
