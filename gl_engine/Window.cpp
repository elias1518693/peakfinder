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
    current_image = framebuffer->toImage();
    paintPanorama();
    f->glFinish(); // synchronization
    m_frame_end = std::chrono::time_point_cast<ClockResolution>(Clock::now());
}
cv::Mat QImageToMat(const QImage& image)
{
    // Convert QImage to QPixmap
    QPixmap pixmap = QPixmap::fromImage(image);

    // Convert QPixmap to cv::Mat
    cv::Mat mat;
    cv::Mat alpha;

    // QImage format conversion to support different pixel formats

    switch (image.format())
    {
        case QImage::Format_ARGB32:
        case QImage::Format_ARGB32_Premultiplied:
            mat = cv::Mat(image.height(), image.width(), CV_8UC4, (void*) image.bits(), image.bytesPerLine());
            break;
        case QImage::Format_RGB32:
            mat = cv::Mat(image.height(), image.width(), CV_8UC4, (void*) image.bits(), image.bytesPerLine());
            cv::cvtColor(mat, mat, cv::COLOR_BGRA2RGBA);
            break;
        case QImage::Format_RGB888:
            mat = cv::Mat(image.height(), image.width(), CV_8UC3, (void*) image.bits(), image.bytesPerLine());
            cv::cvtColor(mat, mat, cv::COLOR_BGR2RGB);
            break;
        case QImage::Format_RGBA8888:
        mat = cv::Mat(image.height(), image.width(), CV_8UC4, (void*) image.bits(), image.bytesPerLine());
        cv::cvtColor(mat, mat, cv::COLOR_BGRA2RGBA);
        break;
        case QImage::Format_Grayscale8:
            mat = cv::Mat(image.height(), image.width(), CV_8UC1, (void*) image.bits(), image.bytesPerLine());
            break;
        default:
            // Unsupported image format
            throw std::runtime_error("Unsupported image format");
    }

    return mat.clone();  // Return a cloned copy of the converted cv::Mat
}
void Window::paintPanorama(QOpenGLFramebufferObject* framebuffer){
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
    glm::dvec2 size = glm::dvec2(glm::max(m_framebuffer->size().x, m_framebuffer->size().y));
    m_camera.set_viewport_size(size);
    m_camera.set_field_of_view(90);

    m_shader_manager->tile_shader()->bind();
    f->glClearColor(1.0, 0.0, 0.5, 1);
    std::vector<glm::dmat4>vp = m_camera.local_view_projection_matrix_cube(m_camera.position());
    std::vector<std::unique_ptr<Framebuffer>>fb(6);

    for(uint i = 0; i < 6; i++){
                fb[i] = std::make_unique<Framebuffer>(Framebuffer::DepthFormat::Int24, std::vector({ Framebuffer::ColourFormat::RGBA8 }));
                fb[i]->resize(size);
                fb[i]->bind();
                f->glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
                f->glEnable(GL_DEPTH_TEST);
                f->glDepthFunc(GL_LESS);
                m_tile_manager->draw_view(m_shader_manager->tile_shader(), m_camera, vp[i]);
                fb[i]->unbind();
    }
    std::unique_ptr<Framebuffer> transferBuffer = std::make_unique<Framebuffer>(Framebuffer::DepthFormat::Int24, std::vector({ Framebuffer::ColourFormat::RGBA8 }));;
    if (framebuffer)
        framebuffer->bind();
    else{
        transferBuffer->resize(glm::dvec2(8000,8000));
        transferBuffer->bind();
        f->glDisable(GL_DEPTH_TEST);
    }
    m_shader_manager->panorama_program()->bind();
    for(uint i = 0; i < 6; i++){
        fb[i]->bind_colour_texture_to_binding(0, i);
        f->glUniform1i(m_shader_manager->panorama_program()->uniform_location("texture_sampler"+std::to_string(i)), i);
    }
    m_shader_manager->panorama_program()->set_uniform("fov",34.2054579f);
    m_screen_quad_geometry.draw();

    m_shader_manager->release();
    if(framebuffer)
        current_image = framebuffer->toImage();
    else{
        current_image = transferBuffer->read_colour_attachment(0);
    }


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



void Window::process_image(const QImage& image){
    m_processImage = true;
    //QImage greyscale = image.convertToFormat(QImage::Format_Grayscale8);
    QOpenGLExtraFunctions* f = QOpenGLContext::currentContext()->extraFunctions();
    cv::Mat debugImage;

    //this->paintPanorama();
    std::unique_ptr<Framebuffer> framebuffer = std::make_unique<Framebuffer>(image, Framebuffer::DepthFormat::None);
    float fov = glm::radians(34.2054579f);
    float k = fov * current_image.width()/(2* glm::pi<float>() * framebuffer->size().x);
    float test = fov * current_image.height()/(2* glm::pi<float>() * framebuffer->size().y);
    qDebug()<<k;
    glm::vec2 imageSize = glm::vec2(image.width() * k , image.height() * test );
    framebuffer->resize(glm::vec2(image.width() * k  , image.height() * test ));
     std::unique_ptr<Framebuffer> framebuffer_out = std::make_unique<Framebuffer>(image, Framebuffer::DepthFormat::None);
    qDebug()<<framebuffer->size().x << framebuffer->size().y;
        m_shader_manager->sobel_program()->bind();
        f->glUniform2f(m_shader_manager->sobel_program()->uniform_location("imageSize"),framebuffer->size().x, framebuffer->size().y);
        f->glUniform1f(m_shader_manager->sobel_program()->uniform_location("fov"), fov);
        framebuffer->bind();
        f->glClearColor(0.0, 0.0, 0.0, 1);
        f->glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
        f->glDisable(GL_DEPTH_TEST);
        f->glDisable(GL_BLEND);
        framebuffer_out->bind_colour_texture(0);
        m_screen_quad_geometry.draw();
    QImage input = framebuffer->read_colour_attachment(0);

    cv::Mat image_real = QImageToMat(input);
    cv::cvtColor(image_real, image_real, cv::COLOR_BGR2GRAY);
    image_real.convertTo(image_real,CV_32F, 1.0/255.0);
     //cv::GaussianBlur(image_real, image_real, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);
    cv::Mat sX, sY, mag1, phase;
    cv::Sobel(image_real, sX, CV_32F, 1, 0);
    cv::Sobel(image_real, sY, CV_32F, 0, 1);
    cv::magnitude(sX, sY, mag1);
    cv::phase(sX, sY, phase, true);

    cv::resize(mag1,debugImage, cv::Size(mag1.cols, mag1.rows));
    cv::imshow("input real",debugImage);
    cv::waitKey(0);
    cv::destroyAllWindows();
    cv::Mat panorama = QImageToMat(current_image);
    //cv::Mat panorama = QImageToMat(current_image);
    cv::cvtColor(panorama, panorama, cv::COLOR_BGR2GRAY);
    panorama.convertTo(panorama,CV_32F, 1.0/255.0);
    //cv::GaussianBlur(panorama, panorama, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);
    cv::Mat sX2, sY2, mag2, phase2;
    cv::Sobel(panorama, sX2, CV_32F, 1, 0);
    cv::Sobel(panorama, sY2, CV_32F, 0, 1);
    cv::magnitude(sX2, sY2, mag2);
    cv::phase(sX2, sY2, phase2, true);
    cv::Mat out_mat;
    int w = image_real.cols;
    int h = image_real.rows;

    cv::resize(mag2,debugImage, cv::Size(mag2.cols/10,mag2.rows/10));
    cv::imshow("input panroama",debugImage);
    cv::waitKey(0);
    cv::destroyAllWindows();

    //template matching
    cv::matchTemplate(phase2.mul(mag2),phase.mul(mag1),out_mat,cv::TM_CCORR);
    double minVal = 0;
    double maxVal = 0;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc(out_mat,&minVal,&maxVal, &minLoc, &maxLoc);

    //Draw rectangle
    cv::Point bottomRight(maxLoc.x + w, maxLoc.y + h);
    cv::rectangle(panorama, maxLoc, bottomRight, cv::Scalar(255, 255, 255), 12);
    image_real.copyTo(panorama.rowRange(maxLoc.y, bottomRight.y).colRange(maxLoc.x, bottomRight.x), image_real != 0);
    cv::resize(panorama,panorama, cv::Size(panorama.cols/8,panorama.rows/8));
    cv::imshow("matched",panorama);
    cv::waitKey(0);
    cv::destroyAllWindows();
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


