/*****************************************************************************
 * Alpine Terrain Renderer
 * Copyright (C) 2023 Gerald Kimmersdorfer
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

#pragma once

#include <QObject>
#include <QMap>
#include <QList>
#include <QString>

#include "nucleus/timing/TimerManager.h"
#include "TimerFrontendObject.h"

class TimerFrontendManager : public QObject
{
    Q_OBJECT

public:
    TimerFrontendManager(const TimerFrontendManager& src);
    ~TimerFrontendManager();
    TimerFrontendManager(QObject* parent = nullptr);

public slots:
    void receive_measurements(QList<nucleus::timing::TimerReport> values);

signals:
    void updateTimingList(QList<TimerFrontendObject*> data);

private:
    QList<TimerFrontendObject*> m_timer;
    QMap<QString, TimerFrontendObject*> m_timer_map;
    static int current_frame;

};
