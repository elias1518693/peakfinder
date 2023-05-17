/*****************************************************************************
 * Alpine Terrain Renderer
 * Copyright (C) 2022 Adam Celarek
 * Copyright (C) 2023 Jakob Lindner
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *****************************************************************************/

#include <array>

#include <QDebug>
#include <QImage>
#include <QImageWriter>
#include <QMoveEvent>
#include <QOpenGLBuffer>
#include <QOpenGLContext>
#include <QOpenGLDebugLogger>
#include <QOpenGLExtraFunctions>
#include <QOpenGLFramebufferObject>
#include <QOpenGLShaderProgram>
#include <QOpenGLTexture>
#include <QOpenGLVertexArrayObject>
#include <QPropertyAnimation>
#include <QRandomGenerator>
#include <QSequentialAnimationGroup>
#include <QTimer>
#include <glm/glm.hpp>

#include "Atmosphere.h"
#include "DebugPainter.h"
#include "Framebuffer.h"
#include "ShaderManager.h"
#include "ShaderProgram.h"
#include "TileManager.h"
#include "Window.h"
#include "helpers.h"
#include "nucleus/utils/bit_coding.h"
#include "opencv2/opencv.hpp"

using gl_engine::Window;

Window::Window()
    : m_camera({ 1822577.0, 6141664.0 - 500, 171.28 + 500 }, { 1822577.0, 6141664.0, 171.28 }) // should point right at the stephansdom
{
    qDebug("Window::Window()");
    m_tile_manager = std::make_unique<TileManager>();
    QTimer::singleShot(1, [this]() { emit update_requested(); });
}

Window::~Window()
{
    qDebug("~Window::Window()");
}

void Window::initialise_gpu()
{
    QOpenGLDebugLogger* logger = new QOpenGLDebugLogger(this);
    logger->initialize();
    connect(logger, &QOpenGLDebugLogger::messageLogged, [](const auto& message) {
        qDebug() << message;
    });
    logger->disableMessages(QList<GLuint>({ 131185 }));
    logger->startLogging(QOpenGLDebugLogger::SynchronousLogging);

    m_debug_painter = std::make_unique<DebugPainter>();
    m_shader_manager = std::make_unique<ShaderManager>();
    m_atmosphere = std::make_unique<Atmosphere>();

    m_tile_manager->init();
    m_tile_manager->initilise_attribute_locations(m_shader_manager->tile_shader());
    m_screen_quad_geometry = gl_engine::helpers::create_screen_quad_geometry();
    m_framebuffer = std::make_unique<Framebuffer>(Framebuffer::DepthFormat::Int24, std::vector({ Framebuffer::ColourFormat::RGBA8 }));
    m_depth_buffer = std::make_unique<Framebuffer>(Framebuffer::DepthFormat::Int24, std::vector({ Framebuffer::ColourFormat::RGBA8 }));
    emit gpu_ready_changed(true);
}

void Window::resize_framebuffer(int width, int height)
{
    if (width == 0 || height == 0)
        return;

    QOpenGLFunctions* f = QOpenGLContext::currentContext()->functions();
    m_framebuffer->resize({ width, height });
    m_atmosphere->resize({ width, height });
    m_depth_buffer->resize({ width / 4, height / 4 });

    f->glViewport(0, 0, width, height);
}

void Window::paint(QOpenGLFramebufferObject* framebuffer)
{
    m_frame_start = std::chrono::time_point_cast<ClockResolution>(Clock::now());
    QOpenGLExtraFunctions *f = QOpenGLContext::currentContext()->extraFunctions();
    f->glEnable(GL_CULL_FACE);
    f->glCullFace(GL_BACK);

    // DEPTH BUFFER
    m_camera.set_viewport_size(m_depth_buffer->size());
    m_depth_buffer->bind();
    f->glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
    f->glEnable(GL_DEPTH_TEST);
    f->glDepthFunc(GL_LESS);

    m_shader_manager->depth_program()->bind();
    m_tile_manager->draw(m_shader_manager->depth_program(), m_camera);
    m_depth_buffer->unbind();
    // END DEPTH BUFFER

    m_camera.set_viewport_size(m_framebuffer->size());
    m_framebuffer->bind();
    f->glClearColor(1.0, 0.0, 0.5, 1);

    f->glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

    m_shader_manager->atmosphere_bg_program()->bind();
    m_atmosphere->draw(m_shader_manager->atmosphere_bg_program(),
                       m_camera,
                       m_shader_manager->screen_quad_program(),
                       m_framebuffer.get());

    f->glEnable(GL_DEPTH_TEST);
    f->glDepthFunc(GL_LESS);
    f->glEnable(GL_BLEND);
    f->glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    m_shader_manager->tile_shader()->bind();
    m_tile_manager->draw(m_shader_manager->tile_shader(), m_camera);

    m_framebuffer->unbind();
    if (framebuffer)
        framebuffer->bind();

    m_shader_manager->screen_quad_program()->bind();
    m_framebuffer->bind_colour_texture(0);
    m_screen_quad_geometry.draw();

    m_shader_manager->release();

    f->glFinish(); // synchronization
    m_frame_end = std::chrono::time_point_cast<ClockResolution>(Clock::now());
}

void Window::paintPanorama(QOpenGLFramebufferObject* framebuffer){

    m_frame_start = std::chrono::time_point_cast<ClockResolution>(Clock::now());
    QOpenGLExtraFunctions* f = QOpenGLContext::currentContext()->extraFunctions();

    // DEPTH BUFFER
    m_camera.set_viewport_size(m_depth_buffer->size());
    m_depth_buffer->bind();
    f->glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
    f->glEnable(GL_DEPTH_TEST);
    f->glDepthFunc(GL_LESS);

    m_shader_manager->depth_program()->bind();
    m_tile_manager->draw(m_shader_manager->depth_program(), m_camera);
    m_depth_buffer->unbind();
    // END DEPTH BUFFER

    m_camera.set_viewport_size(m_framebuffer->size());

    m_shader_manager->tile_shader()->bind();
    f->glClearColor(1.0, 0.0, 0.5, 1);
    std::vector<glm::dmat4>vp = m_camera.local_view_projection_matrix_cube(m_camera.position());
    std::vector<std::unique_ptr<Framebuffer>>fb(6);
    for(uint i = 0; i < 6; i++){
                fb[i] = std::make_unique<Framebuffer>(Framebuffer::DepthFormat::Int24, std::vector({ Framebuffer::ColourFormat::RGBA8 }));
                fb[i]->resize(m_framebuffer->size());
                fb[i]->bind();
                f->glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
                f->glEnable(GL_DEPTH_TEST);
                f->glDepthFunc(GL_LESS);
                m_tile_manager->draw_view(m_shader_manager->tile_shader(), m_camera, vp[i]);
                fb[i]->unbind();
    }
    if (framebuffer)
        framebuffer->bind();
    m_shader_manager->panorama_program()->bind();
    for(uint i = 0; i < 6; i++){
        fb[i]->bind_colour_texture_to_binding(0, i);
        f->glUniform1i(m_shader_manager->panorama_program()->uniform_location("texture_sampler"+std::to_string(i)), i);
    }
    m_shader_manager->panorama_program()->set_uniform("fov",40.0f);
    m_screen_quad_geometry.draw();
    /*
        std::unique_ptr<Framebuffer> big_framebuffer = std::make_unique<Framebuffer>(Framebuffer::DepthFormat::Int24, std::vector({ Framebuffer::ColourFormat::RGBA8 }));
        big_framebuffer->resize({4000,4000});
        big_framebuffer->bind();
        for(uint i = 0; i < 6; i++){
            fb[i]->bind_colour_texture_to_binding(0, i);
            f->glUniform1i(m_shader_manager->panorama_program()->uniform_location("texture_sampler"+std::to_string(i)), i);
        }
        m_screen_quad_geometry.draw();
        QString imagePath(QStringLiteral("image.jpeg"));
        QImage image = big_framebuffer->read_colour_attachment(0);
        {
            QImageWriter writer(imagePath);
            if(!writer.write(image))
                qDebug() << writer.errorString();
        }
        big_framebuffer->unbind();
        captureFrame = false;
    */
    m_shader_manager->release();
    f->glFinish(); // synchronization
    m_frame_end = std::chrono::time_point_cast<ClockResolution>(Clock::now());
}


cv::Mat ComputeVCC(const cv::Mat& SP, const cv::Mat& DP, const cv::Mat& SR, const cv::Mat& DR)
{
    cv::Size dimP = SP.size();
    cv::Size dimR = SR.size();

    cv::Mat paddedSR, paddedDR;
    cv::copyMakeBorder(SR, paddedSR, 0, dimP.height, 0, dimP.width, cv::BORDER_CONSTANT, cv::Scalar(0));
    cv::copyMakeBorder(DR, paddedDR, 0, dimR.height, 0, dimR.width, cv::BORDER_CONSTANT, cv::Scalar(0));
qDebug()<<"here";
    cv::Mat COMP = paddedSR;
    qDebug()<<"here";
    cv::Mat COMR = paddedSR;
    DP.convertTo(DP,  SP.type());
    paddedSR.convertTo(paddedSR,  SP.type());
    paddedDR.convertTo(paddedDR,  SP.type());
    COMP.convertTo(COMP,  SP.type());
    COMR.convertTo(COMR,  SP.type());
    for (int i = 0; i < SP.rows; i++)
    {
        for (int j = 0; j < SP.cols; j++)
        {
            COMP.at<double>(i, j) = SP.at<double>(i, j) * cos(DP.at<double>(i, j));
            COMR.at<double>(i, j) = paddedSR.at<double>(i, j) * cos(paddedDR.at<double>(i, j));

        }
    }
    qDebug()<<"did it #############";
    cv::Mat COMR_squared, COMP_squared;
    cv::multiply(COMR, COMR, COMR_squared);
    cv::multiply(COMP, COMP, COMP_squared);
qDebug()<<"here";
    cv::dft(COMP_squared, COMP_squared, cv::DFT_COMPLEX_OUTPUT);
    cv::dft(COMR_squared, COMR_squared, cv::DFT_COMPLEX_OUTPUT);
qDebug()<<"here";
    cv::Mat VCC;
    cv::mulSpectrums(COMP_squared, COMR_squared, VCC, cv::DFT_ROWS | cv::DFT_COMPLEX_OUTPUT, true);
    cv::idft(VCC, VCC, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
qDebug()<<"here";
    cv::flip(VCC, VCC, 1);
    VCC = VCC(cv::Rect(0, dimP.height, dimP.width, dimP.height));
qDebug()<<"here";
    return VCC;
}




QImage keepFirstNonZeroPixels(const QImage& inputImage)
{
    QImage outputImage(inputImage.size(), inputImage.format());
    outputImage.fill(qRgb(0, 0, 0)); // Fill the output image with black

    for (int x = 0; x < inputImage.width(); ++x) {

        for (int y = 0; y < inputImage.height(); ++y) {
            QRgb pixel = inputImage.pixel(x, inputImage.height()-1-y);
            if (qRed(pixel) > 0 || qGreen(pixel) > 0 || qBlue(pixel) > 0) {
                    outputImage.setPixel(x, y, qRgb(255,255,255));
                    //break;
            }
        }
    }

    return outputImage;
}
QImage mat_to_qimage(cv::Mat const& mat)
{

    // Create a QImage from the Mat data
    QImage qImage(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_ARGB32);
    qImage.bits(); // enforce deep copy

    return qImage;
}

cv::Mat qimage_to_mat(QImage image){

   return cv::Mat(image.height(), image.width(), CV_8UC4, (void*) image.constBits(), image.bytesPerLine());
 }
void detectAndMatchSIFTFeatures(const QImage& image1, const QImage& image2, std::vector<std::vector<cv::DMatch>>& matches)
{
    cv::Mat mat1 = cv::Mat(image1.height(), image1.width(), CV_8UC4, (void*) image1.constBits(), image1.bytesPerLine());
    cv::Mat mat2 = cv::Mat(image2.height(), image2.width(), CV_8UC4, (void*) image2.constBits(), image2.bytesPerLine());

    // Convert the images to grayscale
    cv::Mat gray1, gray2;
    cv::cvtColor(mat1, gray1, cv::COLOR_BGRA2GRAY);
    cv::cvtColor(mat2, gray2, cv::COLOR_BGRA2GRAY);
    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat des1, des2;
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
    sift->detectAndCompute(mat1, cv::noArray(), kp1, des1);
    sift->detectAndCompute(mat2, cv::noArray(), kp2, des2);

    // BFMatcher with default params
    cv::BFMatcher bf(cv::NORM_L2);
    bf.knnMatch(des1, des2, matches, 2);

    // Apply ratio test
    std::vector<cv::DMatch> good;
    for(size_t i = 0; i < matches.size(); i++) {
        if(matches[i][0].distance < 0.6 * matches[i][1].distance) {
            good.push_back(matches[i][0]);
        }
    }

    // Draw matches
    cv::Mat matchedImage;
    cv::drawMatches(mat1, kp1, mat2, kp2, good, matchedImage, cv::Scalar::all(-1), cv::Scalar::all(-1),
        std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    matchedImage.convertTo(matchedImage, CV_8UC4);
    QImage qmatchedImage = mat_to_qimage(matchedImage);
    qDebug() << matches.size();
    qDebug() << good.size();
    //qDebug() << matchesMask.size();

    QString imagePath(QStringLiteral("matchedimage.jpeg"));
    {
        QImageWriter writer(imagePath);
        if(!writer.write(qmatchedImage))
            qDebug() << writer.errorString();
    }
  }

void detectAndMatchORBFeatures(const QImage& image1, const QImage& image2, std::vector<cv::DMatch>& matches)
{
    // Convert the QImage objects to cv::Mat objects
    cv::Mat mat1 = cv::Mat(image1.height(), image1.width(), CV_8UC4, (void*) image1.constBits(), image1.bytesPerLine());
    cv::Mat mat2 = cv::Mat(image2.height(), image2.width(), CV_8UC4, (void*) image2.constBits(), image2.bytesPerLine());

    // Convert the images to grayscale
    cv::Mat gray1, gray2;
    cv::cvtColor(mat1, gray1, cv::COLOR_BGRA2GRAY);
    cv::cvtColor(mat2, gray2, cv::COLOR_BGRA2GRAY);

    // Detect keypoints and compute descriptors using ORB
    cv::Ptr<cv::ORB> orb = cv::ORB::create(2000);
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;
    orb->detectAndCompute(mat1, cv::Mat(), keypoints1, descriptors1);
    orb->detectAndCompute(mat2, cv::Mat(), keypoints2, descriptors2);

    // Match features using Brute-Force Matcher
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<cv::DMatch> bf_matches;
    matcher.match(descriptors1, descriptors2, bf_matches);


    const float dist_ratio = 0.8;
    for (const cv::DMatch& match : bf_matches) {
        //if (match.distance < dist_ratio * bf_matches[match.queryIdx].distance) {
            matches.push_back(match);
        //}
    }
    // Sort matches by score
     std::sort(matches.begin(), matches.end());

     // Remove not so good matches
     const float GOOD_MATCH_PERCENT = 0.01f;
     const int numGoodMatches = matches.size() * GOOD_MATCH_PERCENT;
     matches.erase(matches.begin()+numGoodMatches, matches.end());
     /*
    cv::Mat mask;
    std::vector<cv::Point2f> points1, points2;
    for( size_t i = 0; i < matches.size(); i++ )
      {
        points1.push_back( keypoints1[ matches[i].queryIdx ].pt );
        points2.push_back( keypoints2[ matches[i].trainIdx ].pt );
      }
    cv::Mat h = cv::findHomography(points1, points2, cv::RANSAC, 3,mask);
    std::vector<cv::DMatch> matchesMask(mask.rows * mask.cols);
    for(int i = 0; i < mask.rows; i++) {
        for(int j = 0; j < mask.cols; j++) {
            matchesMask[i * mask.cols + j] = mask.at<cv::DMatch>(i, j);
        }
    }
    */
    // Draw matches between the images
        cv::Mat matchedImage;
        cv::drawMatches(mat1, keypoints1, mat2, keypoints2, matches, matchedImage);

        // Convert the matched image back to QImage and display it
        matchedImage.convertTo(matchedImage, CV_8UC4);
        QImage qmatchedImage = mat_to_qimage(matchedImage);
        qDebug() << matches.size();
        //qDebug() << matchesMask.size();

        QString imagePath(QStringLiteral("matchedimage.jpeg"));
        {
            QImageWriter writer(imagePath);
            if(!writer.write(qmatchedImage))
                qDebug() << writer.errorString();
        }
}

void Window::process_image(const QImage& image){
    //QImage greyscale = image.convertToFormat(QImage::Format_Grayscale8);
    QOpenGLExtraFunctions* f = QOpenGLContext::currentContext()->extraFunctions();
    cv::Mat debugImage;

    //SOBEL FILTER INPUT IMAGE
    /*
    std::unique_ptr<Framebuffer> framebuffer = std::make_unique<Framebuffer>(image, Framebuffer::DepthFormat::None);
    qDebug()<<framebuffer->size().x << framebuffer->size().y;
    m_shader_manager->sobel_program()->bind();
    framebuffer->bind();
    f->glDisable(GL_DEPTH_TEST);
    f->glDisable(GL_BLEND);
    framebuffer->bind_colour_texture(0);
    m_screen_quad_geometry.draw();
*/
    //input = keepFirstNonZeroPixels(input);
    cv::Mat image_real = qimage_to_mat(image);
    cv::Mat gray;
    cv::cvtColor(image_real, gray, cv::COLOR_BGR2GRAY);
    cv::Sobel(gray,gray, CV_32F, 1, 0);
    cv::Sobel(gray,gray, CV_32F, 0, 1);


    //SOBEL FILTER PANORAMA IMAGE
    /*
    m_framebuffer->bind();
    f->glDisable(GL_DEPTH_TEST);
    f->glDisable(GL_BLEND);
    m_framebuffer->bind_colour_texture(0);
    m_screen_quad_geometry.draw();
    m_shader_manager->sobel_program()->release();
*/
    cv::Mat panorama;
    qimage_to_mat(m_framebuffer->read_colour_attachment(0)).convertTo(panorama, CV_32FC4,  1.0/255.0);

    cv::Mat out_mat;
    int w = panorama.cols;
    int h = panorama.rows;
    cv::Mat matching_image;
    image_real.convertTo(matching_image, CV_32FC4, 1.0/255.0);

    //debug images
    cv::resize(image_real,debugImage, cv::Size(960, 540));
    cv::imshow("input real",debugImage);
    cv::waitKey(0);
    cv::destroyAllWindows();

    cv::resize(panorama,debugImage, cv::Size(960, 540));
    cv::imshow("input panroama",debugImage);
    cv::waitKey(0);
    cv::destroyAllWindows();

    //template matching
    cv::matchTemplate(matching_image,panorama,out_mat,cv::TM_CCORR);
    double minVal = 0;
    double maxVal = 0;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc(out_mat,&minVal,&maxVal, &minLoc, &maxLoc);

    //Draw rectangle
    cv::Point bottomRight(maxLoc.x + w, maxLoc.y + h);
    cv::rectangle(image_real, maxLoc, bottomRight, cv::Scalar(255, 255, 255), 5);
    cv::resize(image_real,image_real, cv::Size(960, 540));
    cv::imshow("matched",image_real);
    cv::waitKey(0);
    cv::destroyAllWindows();

    //debug images
    /*
    QImage out_image = mat_to_qimage(image_real);
    QString imagePath(QStringLiteral("sobel.jpeg"));
    {
        QImageWriter writer(imagePath);
        if(!writer.write(out_image))
            qDebug() << writer.errorString();
    }

    QString realPath(QStringLiteral("real_image.jpeg"));
    {
        QImageWriter writer(realPath);
        if(!writer.write(input))
            qDebug() << writer.errorString();
    }
    QString panoramaPath(QStringLiteral("panorama.jpeg"));
    {
        QImageWriter writer(panoramaPath);
        if(!writer.write(mat_to_qimage(panorama)))
            qDebug() << writer.errorString();
    }
    */
}
void Window::paintOverGL(QPainter* painter)
{
    const auto frame_duration = (m_frame_end - m_frame_start);
    const auto frame_duration_float = double(frame_duration.count()) / 1000.;
    const auto frame_duration_text = QString("Last frame: %1ms, draw indicator: ")
                                         .arg(QString::asprintf("%04.1f", frame_duration_float));

    const auto random_u32 = QRandomGenerator::global()->generate();

    painter->setFont(QFont("Helvetica", 12));
    painter->setPen(Qt::white);
    QRect text_bb = painter->boundingRect(10, 20, 1, 15, Qt::TextSingleLine, frame_duration_text);
    painter->drawText(10, 20, frame_duration_text);
    painter->drawText(10, 40, m_debug_scheduler_stats);
    painter->drawText(10, 60, m_debug_text);
    painter->setBrush(QBrush(QColor(random_u32)));
    painter->drawRect(int(text_bb.right()) + 5, 8, 12, 12);
}

void Window::keyPressEvent(QKeyEvent* e)
{
    if (e->key() == Qt::Key::Key_F5) {
        m_shader_manager->reload_shaders();
        qDebug("all shaders reloaded");
        emit update_requested();
    }
    if (e->key() == Qt::Key::Key_F11
        || (e->key() == Qt::Key_P && e->modifiers() == Qt::ControlModifier)
        || (e->key() == Qt::Key_F5 && e->modifiers() == Qt::ControlModifier)) {
        e->ignore();
    }

    emit key_pressed(e->keyCombination());
}

void Window::keyReleaseEvent(QKeyEvent* e)
{
    emit key_released(e->keyCombination());
}

void Window::updateCameraEvent()
{
    emit update_camera_requested();
}

void Window::set_permissible_screen_space_error(float new_error)
{
    if (m_tile_manager)
        m_tile_manager->set_permissible_screen_space_error(new_error);
}

void Window::update_camera(const nucleus::camera::Definition& new_definition)
{
    //    qDebug("void Window::update_camera(const nucleus::camera::Definition& new_definition)");
    m_camera = new_definition;
    emit update_requested();
}

void Window::update_debug_scheduler_stats(const QString& stats)
{
    m_debug_scheduler_stats = stats;
    emit update_requested();
}

void Window::update_gpu_quads(const std::vector<nucleus::tile_scheduler::tile_types::GpuTileQuad>& new_quads, const std::vector<tile::Id>& deleted_quads)
{
    assert(m_tile_manager);
    m_tile_manager->update_gpu_quads(new_quads, deleted_quads);
}

float Window::depth(const glm::dvec2& normalised_device_coordinates)
{
    const auto read_float = float(m_depth_buffer->read_colour_attachment_pixel(0, normalised_device_coordinates)[0]) / 255.f;
    //    const auto read_float = nucleus::utils::bit_coding::to_f16f16(m_depth_buffer->read_colour_attachment_pixel(0, normalised_device_coordinates))[0];
    const auto depth = std::exp(read_float * 13.f);
    return depth;
}

glm::dvec3 Window::position(const glm::dvec2& normalised_device_coordinates)
{
    return m_camera.position() + m_camera.ray_direction(normalised_device_coordinates) * (double)depth(normalised_device_coordinates);
}

void Window::deinit_gpu()
{
    emit gpu_ready_changed(false);
    m_tile_manager.reset();
    m_debug_painter.reset();
    m_atmosphere.reset();
    m_shader_manager.reset();
    m_framebuffer.reset();
    m_depth_buffer.reset();
    m_screen_quad_geometry = {};
}

void Window::set_aabb_decorator(const nucleus::tile_scheduler::utils::AabbDecoratorPtr& new_aabb_decorator)
{
    assert(m_tile_manager);
    m_tile_manager->set_aabb_decorator(new_aabb_decorator);
}

void Window::add_tile(const std::shared_ptr<nucleus::Tile>& tile)
{
    assert(m_tile_manager);
    m_tile_manager->add_tile(tile);
}

void Window::remove_tile(const tile::Id& id)
{
    assert(m_tile_manager);
    m_tile_manager->remove_tile(id);
}


nucleus::camera::AbstractDepthTester* Window::depth_tester()
{
    return this;
}
