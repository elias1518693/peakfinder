import QtQuick
import QtQuick.Controls.Material
import QtQuick.Layouts
import QtQuick.Dialogs
import Alpine

import "components"

Item {
    id: main
    property int theme: Material.Light //Material.System
    property int accent: Material.Green

    Rectangle {
        id: tool_bar
        height: 60
        color: main_stack_view.depth === 1 ? "#01FF00FF" : Qt.alpha(Material.backgroundColor, 0.7)
        anchors {
            left: parent.left
            right: parent.right
            top: parent.top
        }

        RowLayout {
            anchors.fill: parent
            anchors.margins: 6
            Rectangle {
                width: 48
                height: 48
                color: "#00FF0000"
                Image {
                    source: "icons/menu.svg"
                    width: parent.width / 2
                    height: parent.height / 2
                    anchors.centerIn: parent
                }
                MouseArea {
                    anchors.fill: parent
                    onClicked: {
                        menu.open()
                        search_results.visible = false
                    }
                }
            }
            Label {
                id: page_title
                text: ""
                wrapMode: Label.Wrap
                font.pointSize: 24
                Layout.fillWidth: true
            }
            SearchBox {
                id: search
                search_results: search_results
            }
        }
        z: 100
    }

    SearchResults {
        id: search_results
        map: map
        search_height: search.height
        anchors {
            top: tool_bar.bottom
            bottom: parent.bottom
            left: parent.left
            right: parent.right
            margins: 10
        }
    }

    PageDrawer {
        id: menu

        bannerTitle: "Alpine Maps"
        bannerIconSource: "../icons/favicon_256.png"
        bannerSubtitle: qsTr ("Version 0.8 Alpha")
        selectedButtonId: 0

        DrawerSeparator {}

        DrawerButton {
            bid: 0
            text: qsTr ("Map")
            iconSource: "../icons/minimal/mountain.png"
            onClicked: change_page("map", qsTr("Map"))
        }

        DrawerButton {
            text: qsTr ("Coordinates")
            iconSource: "../icons/minimal/location.png"
            onClicked: change_page("Coordinates.qml", qsTr("Coordinates"))
        }

        DrawerButton {
            text: qsTr ("Settings")
            iconSource: "../icons/minimal/settings.png"
            onClicked: change_page("Settings.qml", qsTr("Settings"))
        }

        DrawerSeparator {}

        DrawerButton {
            text: stats_window.visible ? qsTr ("Hide Statistics") : qsTr("Statistics")
            iconSource: "../icons/minimal/charts.png"
            selectable: false
            onClicked: stats_window.visible = !stats_window.visible
        }

        DrawerSpacer {}

        DrawerSeparator {}

        DrawerButton {
            text: qsTr ("About")
            iconSource: "../icons/minimal/information.png"
            onClicked: change_page("About.qml", qsTr("About"))
        }

    }

    function change_page(source, title) {
        if (source === "map") {
            if (main_stack_view.depth >= 1) main_stack_view.pop()
            page_title.visible = false
            search.visible = true
            return
        }
        if (main_stack_view.depth === 1)
            main_stack_view.push(_qmlPath + source)
        else
            main_stack_view.replace(_qmlPath + source)
        page_title.visible = true
        search.visible = false
        page_title.text = title
    }


    TerrainRenderer {
        id: map
        focus: true
        anchors.fill: parent
        onHud_visible_changed: function(new_hud_visible) {
            tool_bar.visible = new_hud_visible;
        }

        Keys.onPressed: {
            if (event.key === Qt.Key_S) {
                stats_window.visible = !stats_window.visible
            }
        }
    }

    StackView {
        id: main_stack_view
        anchors {
            top: tool_bar.bottom
            bottom: parent.bottom
            left: parent.left
            right: parent.right
        }

        initialItem: Map {
            renderer: map
        }
    }


    //DebugWindow {}

    StatsWindow {
        id: stats_window
        visible: false
    }

     //property TerrainRenderer renderer
    Component.onCompleted: {
        change_page("map")
    }

}

