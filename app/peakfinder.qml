/*****************************************************************************
 * Alpine Terrain Renderer
 * Copyright (C) 2023 Adam Celarek
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

import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import Alpine

Rectangle {
    id: root
    color: "#00FFFFFF"

    Rectangle {
        anchors {
            centerIn: root
        }
        width: 400
        height: 400

        Column {
            spacing: 10
            anchors.centerIn: parent

            TextField {
                id: latitudeTextField
                placeholderText: "Latitude"
            }

            TextField {
                id: longitudeTextField
                placeholderText: "Longitude"
            }

            TextField {
                id: heightTextField
                placeholderText: "Height"
            }

            TextField {
                id: imageUrlTextField
                placeholderText: "Image URL"
            }

            Button {
                text: "Send"
                onClicked: {
                    var latitude = latitudeTextField.text
                    var longitude = longitudeTextField.text
                    var height = heightTextField.text
                    var imageUrl = imageUrlTextField.text

                    // Call your function with the text field values as parameters
                    map.set_position(latitude, longitude, height)
                    map.load_image(imageUrl)
                }
            }
        }
    }
}
