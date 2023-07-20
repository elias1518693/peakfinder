/*****************************************************************************
 * Alpine Terrain Builder
 * Copyright (C) 2022 alpinemaps.org
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

#include "ShaderManager.h"

#include <QOpenGLContext>

#include "ShaderProgram.h"

using gl_engine::ShaderManager;
using gl_engine::ShaderProgram;

static const char* const debugVertexShaderSource = R"(
  layout(location = 0) in vec4 a_position;
  uniform highp mat4 matrix;
  void main() {
    gl_Position = matrix * a_position;
  })";

static const char* const debugFragmentShaderSource = R"(
  out lowp vec4 out_Color;
  void main() {
     out_Color = vec4(1.0, 0.0, 0.0, 1.0);
  })";

ShaderManager::ShaderManager()
{
    m_tile_program = std::make_unique<ShaderProgram>(
        ShaderProgram::Files({ "tile.vert" }),
        ShaderProgram::Files({ "atmosphere_implementation.frag", "tile.frag" }));
    m_debug_program = std::make_unique<ShaderProgram>(debugVertexShaderSource, debugFragmentShaderSource);
    m_screen_quad_program = std::make_unique<ShaderProgram>(
        ShaderProgram::Files({ "screen_pass.vert" }),
        ShaderProgram::Files({ "screen_copy.frag" }));
    m_atmosphere_bg_program = std::make_unique<ShaderProgram>(
        ShaderProgram::Files({ "screen_pass.vert" }),
        ShaderProgram::Files({ "atmosphere_implementation.frag", "atmosphere_bg.frag" }));
    m_depth_program
        = std::make_unique<ShaderProgram>(ShaderProgram::Files({ "tile.vert" }),
            ShaderProgram::Files({ "encoder.glsl",
                "depth.frag" }));
    m_panorama_program = std::make_unique<ShaderProgram>(
        ShaderProgram::Files({ "panorama.vert" }),
        ShaderProgram::Files({ "panorama.frag" }));
    m_sobel_program = std::make_unique<ShaderProgram>(
        ShaderProgram::Files({ "screen_pass.vert" }),
        ShaderProgram::Files({ "sobel.frag" }));
    m_cylinder_program = std::make_unique<ShaderProgram>(
        ShaderProgram::Files({ "panorama.vert" }),
        ShaderProgram::Files({ "cylinder.frag" }));
    m_program_list.push_back(m_tile_program.get());
    m_program_list.push_back(m_debug_program.get());
    m_program_list.push_back(m_screen_quad_program.get());
    m_program_list.push_back(m_atmosphere_bg_program.get());
    m_program_list.push_back(m_depth_program.get());
    m_program_list.push_back(m_panorama_program.get());
    m_program_list.push_back(m_sobel_program.get());
    m_program_list.push_back(m_cylinder_program.get());
}

ShaderManager::~ShaderManager() = default;

ShaderProgram* ShaderManager::tile_shader() const
{
    return m_tile_program.get();
}

ShaderProgram* ShaderManager::debug_shader() const
{
    return m_debug_program.get();
}

ShaderProgram* ShaderManager::screen_quad_program() const
{
    return m_screen_quad_program.get();
}

ShaderProgram* ShaderManager::atmosphere_bg_program() const
{
    return m_atmosphere_bg_program.get();
}

ShaderProgram* ShaderManager::depth_program() const
{
    return m_depth_program.get();
}

ShaderProgram* ShaderManager::panorama_program() const
{
    return m_panorama_program.get();
}

ShaderProgram* ShaderManager::sobel_program() const
{
    return m_sobel_program.get();
}

ShaderProgram* ShaderManager::cylinder_program() const
{
    return m_cylinder_program.get();
}


void ShaderManager::release()
{
    m_tile_program->release();
}

void ShaderManager::reload_shaders()
{
    for (auto* program : m_program_list) {
        program->reload();
    }
}
